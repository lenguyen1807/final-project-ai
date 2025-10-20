import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

# PEFT might not be installed, so handle import error
try:
    from peft import LoraConfig, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class ViT_GPT2(nn.Module):
    """
    HuggingFace-based Vision Transformer + GPT-2 model for image captioning.
    - Uses `nlpconnect/vit-gpt2-image-captioning`
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
            model_name = "nlpconnect/vit-gpt2-image-captioning"

            print(f"INFO: Loading HuggingFace model: {model_name}")
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            _encoder_model_name = (
                encoder_model_name or "nlpconnect/vit-gpt2-image-captioning"
            )
            _decoder_model_name = decoder_model_name or "gpt2"

            print(f"INFO: Loading encoder from: {_encoder_model_name}")
            print(f"INFO: Loading decoder from: {_decoder_model_name}")

            self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_pretrained_model_name_or_path=_encoder_model_name,
                decoder_pretrained_model_name_or_path=_decoder_model_name,
            )
            self.feature_extractor = ViTImageProcessor.from_pretrained(
                _encoder_model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(_decoder_model_name)

        # Set pad token for tokenizer and model config
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.decoder.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.use_lora = use_lora
        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "PEFT library is not available. Please install it via `pip install peft` to use LoRA."
                )
            print("INFO: Applying LoRA configuration...")
            # Modules to apply LoRA to. These are common for ViT and GPT2.
            target_modules = [
                "encoder.layer.*.attention.attention.query",
                "encoder.layer.*.attention.attention.value",
                "decoder.block.*.attn.c_attn",
            ]
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)

        print(
            f"Number of trainable parameters: {self.model.print_trainable_parameters()}"
        )

    def forward(self, samples):
        images = samples["images"]
        reports = samples["reports"]

        # The feature_extractor expects a list of images.
        # The input `images` is a batch tensor (B, C, H, W).
        # We convert it to a list of tensors, which the processor handles.
        image_list = [img for img in images]
        pixel_values = self.feature_extractor(
            images=image_list, return_tensors="pt"
        ).pixel_values.to(self.model.device)

        # Tokenize reports for labels
        labels = self.tokenizer(
            reports, padding="longest", return_tensors="pt"
        ).input_ids.to(self.model.device)

        # For training, labels should be -100 for padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(pixel_values=pixel_values, labels=labels, return_dict=True)
        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(self, samples, **kwargs):
        images = samples["images"]

        # The feature extractor expects a list of images.
        image_list = [img for img in images]
        pixel_values = self.feature_extractor(
            images=image_list, return_tensors="pt"
        ).pixel_values.to(self.model.device)

        # Set default generation parameters if not provided
        kwargs.setdefault("max_length", 100)
        kwargs.setdefault("num_beams", 5)

        generated_ids = self.model.generate(pixel_values, **kwargs)

        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [text.strip() for text in generated_text]

    # Add save_pretrained for compatibility with the trainer's checkpointing
    def save_pretrained(self, path):
        # If using PEFT, the peft model handles saving correctly.
        # If not, the underlying transformers model handles it.
        print(f"INFO: Saving model to {path}")
        self.model.save_pretrained(path)
