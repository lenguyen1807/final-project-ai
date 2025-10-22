from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BioGptForCausalLM,
    VisionEncoderDecoderModel,
)

# PEFT might not be installed, so handle import error
try:
    from peft import LoraConfig, PeftModel, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: PEFT not found. LoRA will not be available.")

    # Define dummy classes for type hinting if PEFT is not available
    class LoraConfig:
        pass

    class PeftModel(nn.Module):
        pass


class ViT_GPT2(nn.Module):
    """
    HuggingFace-based Vision Transformer + Decoder model for image captioning.
    - Supports ANY vision model and ANY causal LM.
    - Supports robust LoRA fine-tuning via PEFT.
    - Handles decoders that don't support cross-attention (e.g., BioGPT)
      by using a prefix-based approach.
    """

    def __init__(
        self,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        encoder_model_name: Optional[str] = None,
        decoder_model_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the model, tokenizer, and feature extractor.

        Args:
            use_lora: Whether to apply LoRA for parameter-efficient fine-tuning.
            lora_rank: The rank 'r' for the LoRA update matrices.
            lora_alpha: The alpha parameter for LoRA scaling.
            lora_dropout: Dropout probability for LoRA layers.
            encoder_model_name: HuggingFace name/path for the vision encoder.
            decoder_model_name: HuggingFace name/path for the language decoder.
            **kwargs: Catches legacy parameters (e.g., 'gpt2_model').
        """
        super().__init__()
        self.is_prefix_lm = False  # Flag to track the model type
        self.use_lora = use_lora

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
            _decoder_model_name = "gpt2"  # To help LoRA target selection
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
                decoder = BioGptForCausalLM.from_pretrained(
                    _decoder_model_name, config=decoder_config
                )
                self.model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

                # Add and register the projection layer
                self.vision_proj = nn.Linear(
                    encoder.config.hidden_size, decoder.config.hidden_size
                )
                self.model.vision_proj = self.vision_proj
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

            # Load tokenizer and feature extractor AFTER model creation
            self.feature_extractor = AutoImageProcessor.from_pretrained(
                _encoder_model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(_decoder_model_name)

        # --- Tokenizer and Config setup ---
        if self.tokenizer.pad_token is None:
            print("INFO: Tokenizer has no pad_token. Setting pad_token = eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.decoder.config.pad_token_id is None:
                self.model.decoder.config.pad_token_id = self.tokenizer.eos_token_id

        if self.model.config.decoder.pad_token_id is None:
            self.model.config.decoder.pad_token_id = self.tokenizer.pad_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if bos_token_id is None:
            raise ValueError("Tokenizer must have a BOS or EOS token.")

        self.model.config.decoder_start_token_id = bos_token_id
        self.model.decoder.config.decoder_start_token_id = bos_token_id

        # --- BEST PRACTICE: LoRA Configuration ---
        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "PEFT library not available. Please install `peft` to use LoRA."
                )
            print("INFO: Applying LoRA configuration...")

            # --- BEST PRACTICE: DYNAMIC TARGET MODULES ---
            # We build the list of targets based on the *actual* models loaded.
            # We use wildcard matching (`*`) which PEFT supports.

            # 1. Add Encoder (ViT / DINO) targets
            # These are common for all model combinations
            for i in range(12):
                target_modules = [
                    f"encoder.encoder.layer.{i}.attention.attention.query",
                    f"encoder.encoder.layer.{i}.attention.attention.key",
                    f"encoder.encoder.layer.{i}.attention.attention.value",
                    f"encoder.encoder.layer.{i}.attention.output.dense",
                    f"encoder.encoder.layer.{i}.intermediate.dense",
                    f"encoder.encoder.layer.{i}.output.dense",
                ]

            # 2. Add Decoder targets (conditionally)
            if self.is_prefix_lm:
                # This means we are using BioGPT
                print("INFO: Setting LoRA targets for BioGPT decoder.")
                target_modules.extend(
                    [
                        "decoder.layers.[0-9]+.self_attn.q_proj",
                        "decoder.layers.[0-9]+.self_attn.k_proj",
                        "decoder.layers.[0-9]+.self_attn.v_proj",
                        "decoder.layers.[0-9]+.self_attn.out_proj",
                        "decoder.layers.[0-9]+.fc1",  # MLP layer
                        "decoder.layers.[0-9]+.fc2",  # MLP layer
                    ]
                )
            else:
                # This means we are using GPT-2 (or a compatible model)
                print("INFO: Setting LoRA targets for GPT-2 decoder.")
                for i in range(12):  # 0-11 layers
                    target_modules.extend(
                        [
                            f"decoder.transformer.h.{i}.attn.c_attn",
                            f"decoder.transformer.h.{i}.attn.c_proj",
                            f"decoder.transformer.h.{i}.mlp.c_fc",
                            f"decoder.transformer.h.{i}.mlp.c_proj",
                        ]
                    )

            # 3. Handle the custom projection layer (for BioGPT)
            # Best practice is to full-finetune new layers like this one.
            modules_to_save = []
            if self.is_prefix_lm and hasattr(self.model, "vision_proj"):
                modules_to_save.append("vision_proj")

            print(f"INFO: LoRA Target Modules: {target_modules}")
            print(f"INFO: LoRA Modules to Save: {modules_to_save}")

            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save,  # Tell PEFT to full-finetune this
                bias="none",  # Standard practice for LoRA
            )

            # Wrap the model with PEFT
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            trainable_params_count = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f"Total number of trainable parameters: {trainable_params_count}")

    @property
    def device(self) -> torch.device:
        """Returns the device of the underlying model."""
        return next(self.model.parameters()).device

    def forward(self, samples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass for training.
        """
        images = samples["images"]
        reports = samples["reports"]

        image_list = [img for img in images]
        try:
            pixel_values = self.feature_extractor(
                images=image_list, return_tensors="pt"
            ).pixel_values.to(self.device)
        except Exception as e:
            print(f"Error processing images: {e}")
            if not image_list:
                return {
                    "loss": torch.tensor(0.0, device=self.device, requires_grad=True)
                }
            raise e

        self.tokenizer.padding_side = "right"  # Important for training
        text_tokens = self.tokenizer(
            reports,
            padding="longest",
            truncation=True,
            max_length=512,  # Add explicit max_length
            return_tensors="pt",
        )
        labels = text_tokens.input_ids.to(self.device)
        attention_mask = text_tokens.attention_mask.to(self.device)

        if self.is_prefix_lm:
            # --- Prefix-based approach for BioGPT ---
            image_embeds = self.model.encoder(
                pixel_values=pixel_values
            ).last_hidden_state

            # Call the module from self.model, which is PEFT-aware
            image_embeds_proj = self.model.vision_proj(image_embeds)

            input_embeds = self.model.decoder.get_input_embeddings()(labels)
            combined_embeds = torch.cat([image_embeds_proj, input_embeds], dim=1)

            img_attns = torch.ones(
                image_embeds_proj.size()[:-1], dtype=torch.long, device=self.device
            )
            combined_attns = torch.cat([img_attns, attention_mask], dim=1)

            img_labels = torch.full(
                image_embeds_proj.size()[:-1],
                -100,
                dtype=torch.long,
                device=self.device,
            )
            text_labels = labels.masked_fill(attention_mask == 0, -100)
            combined_labels = torch.cat([img_labels, text_labels], dim=1)

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
                pixel_values=pixel_values,
                labels=labels,
                decoder_attention_mask=attention_mask,
                return_dict=True,
            )

        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(self, samples: Dict[str, Any], **kwargs) -> List[str]:
        """
        Generate captions for inference.
        """
        self.model.eval()  # Ensure model is in eval mode
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

            image_embeds_proj = self.model.vision_proj(image_embeds)

            img_attns = torch.ones(
                image_embeds_proj.size()[:-1], dtype=torch.long, device=self.device
            )

            batch_size = image_embeds_proj.shape[0]

            bos_tokens = torch.full(
                (batch_size, 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long,
                device=self.device,
            )
            bos_embeds = self.model.decoder.get_input_embeddings()(bos_tokens)
            inputs_embeds = torch.cat([image_embeds_proj, bos_embeds], dim=1)

            bos_attn = torch.ones(
                bos_embeds.size()[:-1], dtype=torch.long, device=self.device
            )
            attention_mask = torch.cat([img_attns, bos_attn], dim=1)

            generated_ids = self.model.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )

        else:
            # --- Cross-attention generation (standard) ---
            generated_ids = self.model.generate(pixel_values, **kwargs)

        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [text.strip() for text in generated_text]

    def save_pretrained(self, path: str):
        """
        Saves the model. If LoRA is used, this saves the adapter.
        If not, this saves the full model.
        """
        print(f"INFO: Saving model components to {path}")

        # PEFT's save_pretrained will handle saving the adapter_config.json
        # and adapter_model.bin (which includes LoRA weights AND
        # the full weights for 'vision_proj' if it's in modules_to_save).
        self.model.save_pretrained(path)

        # Always save the tokenizer and feature extractor
        self.tokenizer.save_pretrained(path)
        self.feature_extractor.save_pretrained(path)

        print(f"INFO: Model components saved to {path}")
