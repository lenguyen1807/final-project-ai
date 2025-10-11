"""
GPT-2 decoder for medical image captioning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizer

from .base_decoder import TextDecoder


class GPT2Decoder(TextDecoder):
    """GPT-2 decoder with cross-attention for medical image captioning.
    
    This decoder extends GPT-2 with cross-attention capabilities to incorporate
    visual features from image encoders for generating medical image captions.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 128,
        device: str = "cpu",
        add_cross_attention: bool = True,
        freeze_decoder: bool = False
    ):
        """Initialize GPT-2 decoder.
        
        Args:
            model_name: Name of the GPT-2 model to load
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            device: Device to run on
            add_cross_attention: Whether to add cross-attention layers
            freeze_decoder: Whether to freeze decoder parameters
        """
        super().__init__(tokenizer, max_length, device)
        
        # Load GPT-2 configuration
        gpt2_config = GPT2Config.from_pretrained(model_name)
        
        # Add cross-attention if needed
        if add_cross_attention:
            gpt2_config.add_cross_attention = True
        
        # Load GPT-2 model
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name, config=gpt2_config)
        
        # Freeze parameters if needed
        if freeze_decoder:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        self.add_cross_attention = add_cross_attention
        self.model_name = model_name
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through GPT-2 decoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for input tokens [batch_size, seq_len]
            encoder_hidden_states: Hidden states from image encoder [batch_size, seq_len, hidden_size]
            labels: Target labels for training [batch_size, seq_len]
            
        Returns:
            Dictionary containing model outputs
        """
        # Prepare inputs for GPT-2
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Add cross-attention if encoder states are provided
        if encoder_hidden_states is not None and self.add_cross_attention:
            inputs["encoder_hidden_states"] = encoder_hidden_states
        
        # Add labels for training
        if labels is not None:
            inputs["labels"] = labels
        
        # Forward pass through GPT-2
        outputs = self.gpt2(**inputs)
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None
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
            # Start with BOS token
            input_ids = torch.tensor([[self.get_bos_token_id()]], device=self.device)
            generated_tokens = [self.get_bos_token_id()]
            
            for _ in range(max_length):
                # Prepare inputs
                inputs = {
                    "input_ids": input_ids,
                    "encoder_hidden_states": encoder_hidden_states
                }
                
                # Forward pass
                outputs = self.gpt2(**inputs)
                logits = outputs.logits[0, -1, :]  # Get last token logits
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Generate next token
                if do_sample:
                    # Apply nucleus sampling
                    probs = torch.softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    probs = probs / torch.sum(probs)
                    
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1).item()
                
                # Check for EOS token
                if next_token == self.get_eos_token_id():
                    break
                
                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
    
    def get_model(self) -> GPT2LMHeadModel:
        """Get the underlying GPT-2 model."""
        return self.gpt2
    
    def get_config(self) -> GPT2Config:
        """Get the GPT-2 configuration."""
        return self.gpt2.config
