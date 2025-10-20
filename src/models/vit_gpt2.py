import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
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

        # For backward compatibility with old trainer configs
        if "gpt2_model" in kwargs and decoder_model_name is None:
            decoder_model_name = kwargs["gpt2_model"]

        if encoder_model_name is None and decoder_model_name is None:
            # Default to original model if no names are provided
            model_name = "nlpconnect/vit-gpt2-image-captioning"
            print(f"INFO: Loading default HuggingFace model: {model_name}")

            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            # Use AutoImageProcessor for compatibility
            self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            # Build model from specified encoder/decoder
            _encoder_model_name = (
                encoder_model_name or "google/vit-base-patch16-224"  # Default encoder
            )
            _decoder_model_name = decoder_model_name or "gpt2"  # Default decoder

            print(f"INFO: Loading encoder from: {_encoder_model_name}")
            print(f"INFO: Loading decoder from: {_decoder_model_name}")

            # Load the decoder config and enable cross-attention
            decoder_config = AutoConfig.from_pretrained(_decoder_model_name)
            if (
                not hasattr(decoder_config, "add_cross_attention")
                or not decoder_config.add_cross_attention
            ):
                print(
                    "INFO: Setting `add_cross_attention=True` for GPT-2 decoder config."
                )
                decoder_config.add_cross_attention = True

            # Load models separately and combine
            # This is more robust than from_encoder_decoder_pretrained for custom combinations
            encoder = AutoModel.from_pretrained(_encoder_model_name)
            decoder = AutoModelForCausalLM.from_pretrained(
                _decoder_model_name, config=decoder_config
            )
            self.model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

            # Load the correct processor for the chosen encoder
            self.feature_extractor = AutoImageProcessor.from_pretrained(  # <<< CHANGED
                _encoder_model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(_decoder_model_name)

        # Set pad token for tokenizer and model config
        if self.tokenizer.pad_token is None:
            print("INFO: Tokenizer has no pad_token. Setting pad_token = eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure the model's decoder configuration is consistent
        if self.model.config.decoder.pad_token_id is None:
            self.model.config.decoder.pad_token_id = self.tokenizer.pad_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Link the decoder's config to the main model config for generation
        self.model.config.decoder_start_token_id = (
            self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        )

        self.use_lora = use_lora
        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "PEFT library is not available. Please install it via `pip install peft` to use LoRA."
                )
            print("INFO: Applying LoRA configuration...")
            # Modules to apply LoRA to. These are common for ViT/DINO and GPT2.
            # DINOv2 uses "query", "value" like ViT.
            target_modules = [
                "encoder.layer.*.attention.attention.query",
                "encoder.layer.*.attention.attention.value",
                "decoder.block.*.attn.c_attn",
            ]

            # We can also target the linear projection layers if needed
            # "encoder.layer.*.intermediate.dense",
            # "encoder.layer.*.output.dense",
            # "decoder.block.*.mlp.c_fc",
            # "decoder.block.*.mlp.c_proj",

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
        # Helper property to get model device
        return next(self.model.parameters()).device

    def forward(self, samples):
        """
        Forward pass for training.
        No changes needed here.
        """
        images = samples["images"]
        reports = samples["reports"]

        # The feature_extractor (now AutoImageProcessor) expects a list of images.
        # The input `images` is a batch tensor (B, C, H, W).
        # We convert it to a list of tensors, which the processor handles.
        image_list = [img for img in images]

        try:
            pixel_values = self.feature_extractor(
                images=image_list, return_tensors="pt"
            ).pixel_values.to(self.device)
        except Exception as e:
            print(f"ERROR during feature extraction: {e}")
            print(f"Image tensor shape: {images.shape}, dtype: {images.dtype}")
            # The processor for RadDino (ConvNextImageProcessor) might expect
            # images in a different format/range than ViTImageProcessor.
            # The XRayDataset loads images and applies a transform.
            # Ensure the transform just converts to tensor, as the
            # feature_extractor handles resizing and normalization.
            raise e

        # Tokenize reports for labels
        labels = self.tokenizer(
            reports, padding="longest", return_tensors="pt"
        ).input_ids.to(self.device)

        # For training, labels should be -100 for padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Forward pass through VisionEncoderDecoderModel
        outputs = self.model(pixel_values=pixel_values, labels=labels, return_dict=True)
        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(self, samples, **kwargs):
        """
        Generate captions for inference.
        No changes needed here.
        """
        images = samples["images"]

        # The feature extractor expects a list of images.
        image_list = [img for img in images]
        pixel_values = self.feature_extractor(
            images=image_list, return_tensors="pt"
        ).pixel_values.to(self.device)

        # Set default generation parameters if not provided
        kwargs.setdefault("max_length", 100)
        kwargs.setdefault("num_beams", 5)

        # Ensure pad_token_id is set for generation
        kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        # Ensure bos_token_id is set to start generation
        kwargs.setdefault(
            "decoder_start_token_id", self.model.config.decoder_start_token_id
        )

        generated_ids = self.model.generate(pixel_values, **kwargs)

        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [text.strip() for text in generated_text]

    def save_pretrained(self, path):
        """
        Save model checkpoint.
        No changes needed here.
        """
        # If using PEFT, the peft model handles saving correctly.
        # If not, the underlying transformers model handles it.
        print(f"INFO: Saving model to {path}")
        self.model.save_pretrained(path)
