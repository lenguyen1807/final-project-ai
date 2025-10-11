"""
Base encoder class for image encoding in medical image captioning.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class ImageEncoder(nn.Module, ABC):
    """Abstract base class for image encoders.
    
    This class defines the interface that all image encoders must implement
    for medical image captioning tasks.
    """
    
    def __init__(
        self,
        output_dim: int,
        freeze_encoder: bool = False,
        device: str = "cpu"
    ):
        """Initialize the encoder.
        
        Args:
            output_dim: Output dimension of the encoder
            freeze_encoder: Whether to freeze encoder parameters
            device: Device to run on
        """
        super().__init__()
        self.output_dim = output_dim
        self.freeze_encoder = freeze_encoder
        self.device = device
    
    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            
        Returns:
            Encoded features [batch_size, seq_len, hidden_dim]
        """
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        pass
    
    @abstractmethod
    def get_sequence_length(self) -> int:
        """Get the sequence length of the encoded features."""
        pass
    
    def freeze(self):
        """Freeze encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.freeze_encoder = True
    
    def unfreeze(self):
        """Unfreeze encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
