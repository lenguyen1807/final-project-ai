"""
Medical Image Captioning Pipeline
A modular and extensible framework for medical image captioning with config-driven architecture.
"""

__version__ = "1.0.0"
__author__ = "Medical AI Team"

from .data_loader import MedicalImageDataset, create_data_loaders
from .encoders import ImageEncoder, ViTEncoder, SwinEncoder, ResNetEncoder
from .decoders import TextDecoder, GPT2Decoder, T5Decoder, LLaMADecoder
from .utils import Config, setup_logging, set_seed
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "MedicalImageDataset",
    "create_data_loaders", 
    "ImageEncoder",
    "ViTEncoder",
    "SwinEncoder",
    "ResNetEncoder",
    "TextDecoder",
    "GPT2Decoder",
    "T5Decoder",
    "LLaMADecoder",
    "Config",
    "setup_logging",
    "set_seed"
]
