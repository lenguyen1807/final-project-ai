# src/models/vit_gpt2.py

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BioGptForCausalLM,  # Import specifically to check for it
    VisionEncoderDecoderModel,
)

# PEFT might not be installed, so handle import error
try:
    from peft import LoraConfig, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: PEFT not found. LoRA will not be available.")


class ViT_GPT2(nn.Module):
    """
    HuggingFace-based Vision Transformer + GPT-2 model for image captioning.
    - Now supports ANY vision model compatible with VisionEncoderDecoderModel.
    - Supports LoRA fine-tuning.
    - Handles decoders that don't support cross-attention (e.g., BioGPT)
      by using a prefix-based approach.
    """

    def __init__(
        self,
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        encoder_model_name=None,
        decoder_model_name=None,
        **kwargs,  # to accept other params from old signature
    ):
        super().__init__()
        self.is_prefix_lm = False  # Flag to track the model type

        # For backward compatibility with old trainer configs
        if "gpt2_model" in kwargs and decoder_model_name is None:
            decoder_model_name = kwargs["gpt2_model"]

        if encoder_model_name is None and decoder_model_name is None:
            # Default to original model if no names are provided
            model_name = "nlpconnect/vit-gpt2-image-captioning"
            print(f"INFO: Loading default HuggingFace model: {model_name}")

            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            _encoder_model_name = encoder_model_name or "google/vit-base-patch16-224"
            _decoder_model_name = decoder_model_name or "gpt2"

            print(f"INFO: Loading encoder from: {_encoder_model_name}")
            print(f"INFO: Loading decoder from: {_decoder_model_name}")

            encoder = AutoModel.from_pretrained(_encoder_model_name)
            decoder_config = AutoConfig.from_pretrained(_decoder_model_name)

            # --- KEY CHANGE: Check if decoder supports cross-attention ---
            if "biogpt" in _decoder_model_name.lower():
                print("INFO: BioGPT detected. Using prefix-based LM approach.")
                self.is_prefix_lm = True
                # We do NOT set add_cross_attention=True for BioGPT
                decoder = BioGptForCausalLM.from_pretrained(
                    _decoder_model_name, config=decoder_config
                )
                # Add a projection layer from ViT's hidden size to BioGPT's embedding size
                self.vision_proj = nn.Linear(
                    encoder.config.hidden_size, decoder.config.hidden_size
                )
            else:
                print("INFO: Using cross-attention based approach.")
                if (
                    not hasattr(decoder_config, "add_cross_attention")
                    or not decoder_config.add_cross_attention
                ):
                    print(
                        "INFO: Setting `add_cross_attention=True` for decoder config."
                    )
                    decoder_config.add_cross_attention = True
                decoder = AutoModelForCausalLM.from_pretrained(
                    _decoder_model_name, config=decoder_config
                )

            self.model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
            self.feature_extractor = AutoImageProcessor.from_pretrained(
                _encoder_model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(_decoder_model_name)

        if self.tokenizer.pad_token is None:
            print("INFO: Tokenizer has no pad_token. Setting pad_token = eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model.config.decoder.pad_token_id is None:
            self.model.config.decoder.pad_token_id = self.tokenizer.pad_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.config.decoder_start_token_id = (
            self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        )

        self.use_lora = use_lora
        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT library not available.")
            print("INFO: Applying LoRA configuration...")
            target_modules = [
                "decoder.block.*.attn.c_attn",  # GPT-2
                "decoder.layers.*.self_attn.q_proj",  # BioGPT
                "decoder.layers.*.self_attn.v_proj",  # BioGPT
            ]
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)
            print("INFO: LoRA applied.")
            print(
                f"Number of trainable parameters: {self.model.print_trainable_parameters()}"
            )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, samples):
        images = samples["images"]
        reports = samples["reports"]
        image_list = [img for img in images]
        pixel_values = self.feature_extractor(
            images=image_list, return_tensors="pt"
        ).pixel_values.to(self.device)

        # Tokenize reports for labels
        self.tokenizer.padding_side = "right"
        text_tokens = self.tokenizer(reports, padding="longest", return_tensors="pt")
        labels = text_tokens.input_ids.to(self.device)

        if self.is_prefix_lm:
            # --- Prefix-based approach for BioGPT ---
            # 1. Get visual embeddings from the encoder
            image_embeds = self.model.encoder(
                pixel_values=pixel_values
            ).last_hidden_state

            # 2. Project visual embeddings to the decoder's embedding space
            image_embeds_proj = self.vision_proj(image_embeds)

            # 3. Get text embeddings
            input_embeds = self.model.decoder.get_input_embeddings()(labels)

            # 4. Concatenate image and text embeddings: [IMG, TEXT]
            combined_embeds = torch.cat([image_embeds_proj, input_embeds], dim=1)

            # 5. Create corresponding attention mask
            img_attns = torch.ones(
                image_embeds_proj.size()[:-1], dtype=torch.long, device=self.device
            )
            combined_attns = torch.cat(
                [img_attns, text_tokens.attention_mask.to(self.device)], dim=1
            )

            # 6. Prepare labels, ignoring the image part
            img_labels = torch.full(
                image_embeds_proj.size()[:-1],
                -100,
                dtype=torch.long,
                device=self.device,
            )
            text_labels = labels.masked_fill(
                text_tokens.attention_mask.to(self.device) == 0, -100
            )
            combined_labels = torch.cat([img_labels, text_labels], dim=1)

            # 7. Pass directly to the decoder
            outputs = self.model.decoder(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attns,
                labels=combined_labels,
                return_dict=True,
            )
        else:
            # --- Cross-attention approach for standard models (e.g., GPT-2) ---
            labels[labels == self.tokenizer.pad_token_id] = -100
            outputs = self.model(
                pixel_values=pixel_values, labels=labels, return_dict=True
            )

        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(self, samples, **kwargs):
        """
        Generate captions for inference.
        """
        images = samples["images"]

        image_list = [img for img in images]
        pixel_values = self.feature_extractor(
            images=image_list, return_tensors="pt"
        ).pixel_values.to(self.device)

        kwargs.setdefault("max_length", 100)
        kwargs.setdefault("num_beams", 5)
        kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        kwargs.setdefault(
            "decoder_start_token_id", self.model.config.decoder_start_token_id
        )

        if self.is_prefix_lm:
            # --- Prefix-based generation for BioGPT ---
            image_embeds = self.model.encoder(
                pixel_values=pixel_values
            ).last_hidden_state
            image_embeds_proj = self.vision_proj(image_embeds)
            img_attns = torch.ones(
                image_embeds_proj.size()[:-1], dtype=torch.long, device=self.device
            )

            # --- KEY CHANGE: Manually create the BOS token as the starting point ---
            # 1. Get the batch size from the image embeddings
            batch_size = image_embeds_proj.shape[0]

            # 2. Create a tensor for the BOS token ID for each sample in the batch
            bos_tokens = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=self.device,
            )

            # 3. Get the embedding for the BOS token
            bos_embeds = self.model.decoder.get_input_embeddings()(bos_tokens)

            # 4. Concatenate the image embeddings and the BOS token embedding
            inputs_embeds = torch.cat([image_embeds_proj, bos_embeds], dim=1)

            # 5. Create the corresponding attention mask
            bos_attn = torch.ones(
                bos_embeds.size()[:-1], dtype=torch.long, device=self.device
            )
            attention_mask = torch.cat([img_attns, bos_attn], dim=1)

            # Generate text starting from this combined prefix
            # We don't need to pass input_ids because inputs_embeds is the full prompt now
            generated_ids = self.model.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )
            # The generated IDs will not include the prompt, so we don't need to slice them.

        else:
            # --- Cross-attention generation (no changes needed here) ---
            generated_ids = self.model.generate(pixel_values, **kwargs)

        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [text.strip() for text in generated_text]

    def save_pretrained(self, path):
        print(f"INFO: Saving model to {path}")
        self.model.save_pretrained(path)
        # Also save the projection layer if it exists
        if hasattr(self, "vision_proj"):
            torch.save(self.vision_proj.state_dict(), f"{path}/vision_proj.pt")
