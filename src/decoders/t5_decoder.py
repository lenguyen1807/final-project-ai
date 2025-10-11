"""
T5 decoder for medical image captioning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import T5ForConditionalGeneration, T5Config, PreTrainedTokenizer

from .base_decoder import TextDecoder


class T5Decoder(TextDecoder):
    """T5 decoder for medical image captioning.
    
    This decoder uses T5 (Text-to-Text Transfer Transformer) for generating
    medical image captions. T5 treats all NLP tasks as text-to-text problems.
    """
    
    def __init__(
        self,
        model_name: str = "t5-small",
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 128,
        device: str = "cpu",
        freeze_decoder: bool = False
    ):
        """Initialize T5 decoder.
        
        Args:
            model_name: Name of the T5 model to load
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            device: Device to run on
            freeze_decoder: Whether to freeze decoder parameters
        """
        super().__init__(tokenizer, max_length, device)
        
        # Load T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Freeze parameters if needed
        if freeze_decoder:
            for param in self.t5.parameters():
                param.requires_grad = False
        
        self.model_name = model_name
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through T5 decoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for input tokens [batch_size, seq_len]
            encoder_hidden_states: Hidden states from image encoder [batch_size, seq_len, hidden_size]
            labels: Target labels for training [batch_size, seq_len]
            
        Returns:
            Dictionary containing model outputs
        """
        # For T5, we need to encode the visual features as "input" and text as "labels"
        # This is a simplified approach - in practice, you might need to adapt this
        
        # Prepare inputs
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Add labels for training
        if labels is not None:
            inputs["labels"] = labels
        
        # Forward pass through T5
        outputs = self.t5(**inputs)
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None,
            "hidden_states": outputs.decoder_hidden_states if hasattr(outputs, 'decoder_hidden_states') else None,
            "attentions": outputs.decoder_attentions if hasattr(outputs, 'decoder_attentions') else None
        }
    
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
            encoder_hidden_states: Hidden states from image encoder [1, seq_len, hidden_size]
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            
        Returns:
            Generated text string
        """
        self.eval()
        
        with torch.no_grad():
            # For T5, we need to create a task prefix
            # In medical captioning, we can use "generate medical report:"
            task_prefix = "generate medical report:"
            prefix_ids = self.tokenizer.encode(task_prefix, return_tensors="pt").to(self.device)
            
            # Generate using T5
            generation_config = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "early_stopping": True
            }
            
            generated_ids = self.t5.generate(
                input_ids=prefix_ids,
                **generation_config
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Remove task prefix if present
            if task_prefix in generated_text:
                generated_text = generated_text.replace(task_prefix, "").strip()
            
            return generated_text
    
    def get_model(self) -> T5ForConditionalGeneration:
        """Get the underlying T5 model."""
        return self.t5
    
    def get_config(self) -> T5Config:
        """Get the T5 configuration."""
        return self.t5.config
