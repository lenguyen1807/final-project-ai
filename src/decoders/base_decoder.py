"""
Base decoder class for text generation in medical image captioning.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from transformers import PreTrainedTokenizer


class TextDecoder(nn.Module, ABC):
    """Abstract base class for text decoders.
    
    This class defines the interface that all text decoders must implement
    for medical image captioning tasks.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        device: str = "cpu"
    ):
        """Initialize the decoder.
        
        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            device: Device to run on
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the decoder.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input tokens
            encoder_hidden_states: Hidden states from image encoder
            labels: Target labels for training
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1
    ) -> str:
        """Generate text from encoder hidden states.
        
        Args:
            encoder_hidden_states: Hidden states from image encoder
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            
        Returns:
            Generated text string
        """
        pass
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def get_pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_token_id
    
    def get_eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id
    
    def get_bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
