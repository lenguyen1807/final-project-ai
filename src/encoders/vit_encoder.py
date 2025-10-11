"""
ViT (Vision Transformer) encoder for medical image captioning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import ViTModel, ViTConfig

from .base_encoder import ImageEncoder


class ViTEncoder(ImageEncoder):
    """ViT encoder for medical image captioning.
    
    This encoder uses Vision Transformer to encode medical images into
    visual features that can be used by text decoders.
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        output_dim: Optional[int] = None,
        freeze_encoder: bool = False,
        device: str = "cpu",
        projection_dim: Optional[int] = None
    ):
        """Initialize ViT encoder.
        
        Args:
            model_name: Name of the ViT model to load
            output_dim: Output dimension (if None, uses model's hidden size)
            freeze_encoder: Whether to freeze encoder parameters
            device: Device to run on
            projection_dim: Dimension for projection layer (if needed)
        """
        # Load ViT model
        self.vit = ViTModel.from_pretrained(model_name)
        self.model_name = model_name
        
        # Get model's hidden size
        hidden_size = self.vit.config.hidden_size
        
        # Set output dimension
        if output_dim is None:
            output_dim = hidden_size
        
        # Initialize base class
        super().__init__(output_dim, freeze_encoder, device)
        
        # Add projection layer if needed
        if projection_dim is not None and projection_dim != hidden_size:
            self.projection = nn.Linear(hidden_size, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
        
        # Freeze parameters if needed
        if freeze_encoder:
            self.freeze()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT encoder.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            
        Returns:
            Encoded features [batch_size, seq_len, hidden_dim]
        """
        # Forward pass through ViT
        outputs = self.vit(pixel_values=images)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply projection if needed
        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
        
        return hidden_states
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self.output_dim
    
    def get_sequence_length(self) -> int:
        """Get the sequence length of the encoded features."""
        # ViT sequence length = (image_size / patch_size)² + 1 (for CLS token)
        # For 224x224 with 16x16 patches: (224/16)² + 1 = 14² + 1 = 197
        return 197
    
    def get_model(self) -> ViTModel:
        """Get the underlying ViT model."""
        return self.vit
    
    def get_config(self) -> ViTConfig:
        """Get the ViT configuration."""
        return self.vit.config
