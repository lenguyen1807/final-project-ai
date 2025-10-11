"""
ResNet encoder for medical image captioning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import torchvision.models as models

from .base_encoder import ImageEncoder


class ResNetEncoder(ImageEncoder):
    """ResNet encoder for medical image captioning.
    
    This encoder uses ResNet to encode medical images into
    visual features with CNN-based representation.
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        output_dim: Optional[int] = None,
        freeze_encoder: bool = False,
        device: str = "cpu",
        projection_dim: Optional[int] = None
    ):
        """Initialize ResNet encoder.
        
        Args:
            model_name: Name of the ResNet model to load
            output_dim: Output dimension (if None, uses model's feature size)
            freeze_encoder: Whether to freeze encoder parameters
            device: Device to run on
            projection_dim: Dimension for projection layer (if needed)
        """
        # Load ResNet model
        if model_name == "resnet18":
            self.resnet = models.resnet18(pretrained=True)
            feature_dim = 512
        elif model_name == "resnet34":
            self.resnet = models.resnet34(pretrained=True)
            feature_dim = 512
        elif model_name == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
            feature_dim = 2048
        elif model_name == "resnet101":
            self.resnet = models.resnet101(pretrained=True)
            feature_dim = 2048
        elif model_name == "resnet152":
            self.resnet = models.resnet152(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        self.model_name = model_name
        
        # Set output dimension
        if output_dim is None:
            output_dim = feature_dim
        
        # Initialize base class
        super().__init__(output_dim, freeze_encoder, device)
        
        # Add projection layer if needed
        if projection_dim is not None and projection_dim != feature_dim:
            self.projection = nn.Linear(feature_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
        
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze parameters if needed
        if freeze_encoder:
            self.freeze()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet encoder.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            
        Returns:
            Encoded features [batch_size, seq_len, hidden_dim]
        """
        # Forward pass through ResNet
        features = self.resnet(images)  # [batch_size, feature_dim, 1, 1]
        
        # Reshape to sequence format
        batch_size = features.size(0)
        features = features.view(batch_size, -1)  # [batch_size, feature_dim]
        
        # Add sequence dimension for compatibility with transformer decoders
        features = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Apply projection if needed
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self.output_dim
    
    def get_sequence_length(self) -> int:
        """Get the sequence length of the encoded features."""
        # ResNet outputs a single feature vector
        return 1
    
    def get_model(self) -> nn.Module:
        """Get the underlying ResNet model."""
        return self.resnet
