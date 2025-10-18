"""
DEPRECATED: This file contains the original, custom implementation of ViT-GPT2.
It is kept for reference only. The main implementation has been replaced by the
HuggingFace VisionEncoderDecoderModel in `src/models/vit_gpt2.py`.
"""
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.models.eva_vit import create_eva_vit_g
from src.models.medllm import LayerNorm


class ViT_GPT2(nn.Module):
    """
    A simple Vision Transformer + GPT-2 model for image captioning.
    - Vision Encoder: EVA-ViT-g
    - Language Model: GPT-2
    - Connector: A linear projection layer
    """

    def __init__(
        self,
        vit_model: str = "eva_clip_g",
        img_size: int = 224,
        patch_size: int = 32,
        drop_path_rate: float = 0.0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        gpt2_model: str = "gpt2",
        max_txt_len: int = 100,
    ):
        super().__init__()

        # --- Vision Encoder ---
        print(f"INFO: Loading vision encoder: {vit_model}")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model,
            img_size=(1, img_size, img_size),
            patch_size=(1, patch_size, patch_size),
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            precision=vit_precision,
        )
        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
            print("INFO: Vision encoder is frozen.")

        # --- Language Model ---
        print(f"INFO: Loading language model: {gpt2_model}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm_model = GPT2LMHeadModel.from_pretrained(gpt2_model)

        # --- Connector ---
        vit_embed_dim = self.visual_encoder.num_features
        lm_embed_dim = self.lm_model.config.n_embd
        self.vision_proj = nn.Linear(vit_embed_dim, lm_embed_dim)

        self.max_txt_len = max_txt_len

    def init_vision_encoder(
        self,
        model_name,
        img_size,
        patch_size,
        drop_path_rate,
        use_grad_checkpoint,
        precision,
    ):
        visual_encoder = create_eva_vit_g(
            img_size, patch_size, drop_path_rate, use_grad_checkpoint, precision
        )
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with passed dtype
        if torch.cuda.is_available():
            return torch.amp.autocast("cuda", dtype=dtype)
        else:
            return nn.Identity()

    def forward(self, samples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            samples (dict): A dictionary containing:
                - 'images' (torch.Tensor): Input images of shape (B, C, H, W).
                - 'reports' (List[str]): A list of ground truth text reports.

        Returns:
            A dictionary containing the 'loss'.
        """
        images = samples["images"].cuda()
        reports = samples["reports"]

        # Add a depth dimension for the 3D ViT
        if images.ndim == 4:
            images = images.unsqueeze(2)

        # 1. Get visual embeddings
        with torch.no_grad():
            with self.maybe_autocast():
                image_embeds = self.ln_vision(
                    self.visual_encoder(images)
                )  # (B, N_patches + 1, D_vit)

        # 2. Project visual embeddings to LM embedding space
        image_embeds_proj = self.vision_proj(image_embeds)  # (B, N_patches + 1, D_lm)

        # 3. Tokenize text reports
        self.tokenizer.padding_side = "right"
        text_tokens = self.tokenizer(
            reports,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(images.device)

        # 4. Prepare combined inputs for LM
        input_embeds = self.lm_model.transformer.wte(
            text_tokens.input_ids
        )  # (B, L, D_lm)

        # Concatenate image and text embeddings: [IMG, TEXT]
        combined_embeds = torch.cat([image_embeds_proj, input_embeds], dim=1)

        # Create corresponding attention mask
        img_attns = torch.ones(
            image_embeds_proj.size()[:-1], dtype=torch.long, device=images.device
        )
        combined_attns = torch.cat([img_attns, text_tokens.attention_mask], dim=1)

        # 5. Prepare labels for LM, ignoring the image part
        img_labels = torch.full(
            image_embeds_proj.size()[:-1], -100, dtype=torch.long, device=images.device
        )
        text_labels = text_tokens.input_ids.masked_fill(
            text_tokens.attention_mask == 0, -100
        )
        combined_labels = torch.cat([img_labels, text_labels], dim=1)

        # 6. Pass to LM and compute loss
        outputs = self.lm_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attns,
            labels=combined_labels,
            return_dict=True,
        )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(self, samples: Dict[str, Any], **kwargs) -> List[str]:
        """
        Generate text captions for images.

        Args:
            samples (dict): A dictionary containing 'images'.
            **kwargs: Additional arguments for lm_model.generate().

        Returns:
            A list of generated captions.
        """
        images = samples["images"].cuda()

        # Add a depth dimension for the 3D ViT
        if images.ndim == 4:
            images = images.unsqueeze(2)

        # 1. Get visual embeddings
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(images))

        # 2. Project visual embeddings to LM embedding space
        image_embeds_proj = self.vision_proj(image_embeds)
        img_attns = torch.ones(
            image_embeds_proj.size()[:-1], dtype=torch.long, device=images.device
        )

        # 3. Set default generation parameters
        kwargs.setdefault("max_new_tokens", self.max_txt_len)
        kwargs.setdefault("num_beams", 5)
        kwargs.setdefault("do_sample", False)

        # 4. Generate text directly from the image embeddings
        # The model was trained to treat image embeddings as a prefix.
        # By passing them as `inputs_embeds`, we ask the LM to generate
        # the sequence that follows this prefix.
        outputs = self.lm_model.generate(
            inputs_embeds=image_embeds_proj,
            attention_mask=img_attns,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [text.strip() for text in generated_text]
