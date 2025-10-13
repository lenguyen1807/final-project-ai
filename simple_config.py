"""
Simple configuration for medical image captioning training.
Focus on decoder changes for captioning task.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SimpleConfig:
    """Simple configuration for training captioning models."""
    
    # Model settings
    decoder_type: str = "gpt2"  # gpt2, t5, llama
    decoder_model: str = "gpt2"  # specific model name
    encoder_type: str = "vit"  # vit, swin, resnet
    encoder_model: str = "google/vit-base-patch16-224"
    
    # Training settings
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 5e-5
    max_length: int = 128
    
    # Device and system
    device: str = "auto"  # auto, cuda, cpu
    num_workers: int = 2
    seed: int = 42
    
    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    
    # Generation settings
    temperature: float = 0.8
    top_p: float = 0.9
    do_sample: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set decoder model if not specified
        if self.decoder_type == "gpt2" and self.decoder_model == "gpt2":
            self.decoder_model = "gpt2"
        elif self.decoder_type == "t5" and self.decoder_model == "gpt2":
            self.decoder_model = "t5-small"
        elif self.decoder_type == "llama" and self.decoder_model == "gpt2":
            self.decoder_model = "meta-llama/Llama-2-7b-hf"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimpleConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'decoder_type': self.decoder_type,
            'decoder_model': self.decoder_model,
            'encoder_type': self.encoder_type,
            'encoder_model': self.encoder_model,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'max_length': self.max_length,
            'device': self.device,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'do_sample': self.do_sample
        }
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("🔧 Simple Training Configuration")
        print("=" * 40)
        print(f"Decoder: {self.decoder_type} ({self.decoder_model})")
        print(f"Encoder: {self.encoder_type} ({self.encoder_model})")
        print(f"Training: {self.num_epochs} epochs, batch_size={self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"Max length: {self.max_length}")
        print("=" * 40)


# Default configurations for different scenarios
def get_default_config() -> SimpleConfig:
    """Get default configuration for GPT-2 captioning."""
    return SimpleConfig()

def get_t5_config() -> SimpleConfig:
    """Get configuration for T5 captioning."""
    return SimpleConfig(
        decoder_type="t5",
        decoder_model="t5-small",
        learning_rate=3e-4
    )

def get_llama_config() -> SimpleConfig:
    """Get configuration for LLaMA captioning."""
    return SimpleConfig(
        decoder_type="llama",
        decoder_model="meta-llama/Llama-2-7b-hf",
        batch_size=4,  # Smaller batch for LLaMA
        learning_rate=2e-5
    )

def get_quick_config() -> SimpleConfig:
    """Get configuration for quick testing."""
    return SimpleConfig(
        batch_size=2,
        num_epochs=2,
        max_length=64
    )
