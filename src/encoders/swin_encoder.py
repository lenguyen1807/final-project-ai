"""
Swin Transformer encoder for medical image captioning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import SwinModel, SwinConfig

from .base_encoder import ImageEncoder


class SwinEncoder(ImageEncoder):
    """Swin Transformer encoder for medical image captioning.
    
    This encoder uses Swin Transformer to encode medical images into
    visual features with hierarchical representation.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/swin-tiny-patch4-window7-224",
        output_dim: Optional[int] = None,
        freeze_encoder: bool = False,
        device: str = "cpu",
        projection_dim: Optional[int] = None
    ):
        """Initialize Swin encoder.
        
        Args:
            model_name: Name of the Swin model to load
            output_dim: Output dimension (if None, uses model's hidden size)
            freeze_encoder: Whether to freeze encoder parameters
            device: Device to run on
            projection_dim: Dimension for projection layer (if needed)
        """
        # Load Swin model
        self.swin = SwinModel.from_pretrained(model_name)
        self.model_name = model_name
        
        # Get model's hidden size
        hidden_size = self.swin.config.hidden_size
        
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
        """Forward pass through Swin encoder.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            
        Returns:
            Encoded features [batch_size, seq_len, hidden_dim]
        """
        # Forward pass through Swin
        outputs = self.swin(pixel_values=images)
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
        # Swin sequence length depends on the model configuration
        # For swin-tiny: typically 49 (7x7) for 224x224 input
        return 49
    
    def get_model(self) -> SwinModel:
        """Get the underlying Swin model."""
        return self.swin
    
    def get_config(self) -> SwinConfig:
        """Get the Swin configuration."""
        return self.swin.config
