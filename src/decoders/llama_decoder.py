"""
LLaMA decoder for medical image captioning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import LlamaForCausalLM, LlamaConfig, PreTrainedTokenizer

from .base_decoder import TextDecoder


class LLaMADecoder(TextDecoder):
    """LLaMA decoder for medical image captioning.
    
    This decoder uses LLaMA (Large Language Model Meta AI) for generating
    medical image captions. LLaMA provides strong language understanding
    capabilities for medical text generation.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 128,
        device: str = "cpu",
        freeze_decoder: bool = False
    ):
        """Initialize LLaMA decoder.
        
        Args:
            model_name: Name of the LLaMA model to load
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            device: Device to run on
            freeze_decoder: Whether to freeze decoder parameters
        """
        super().__init__(tokenizer, max_length, device)
        
        # Load LLaMA model
        self.llama = LlamaForCausalLM.from_pretrained(model_name)
        
        # Freeze parameters if needed
        if freeze_decoder:
            for param in self.llama.parameters():
                param.requires_grad = False
        
        self.model_name = model_name
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through LLaMA decoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for input tokens [batch_size, seq_len]
            encoder_hidden_states: Hidden states from image encoder [batch_size, seq_len, hidden_size]
            labels: Target labels for training [batch_size, seq_len]
            
        Returns:
            Dictionary containing model outputs
        """
        # Prepare inputs
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Add labels for training
        if labels is not None:
            inputs["labels"] = labels
        
        # Forward pass through LLaMA
        outputs = self.llama(**inputs)
        
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
            # Create a medical prompt for LLaMA
            medical_prompt = "Generate a medical report for this chest X-ray image:"
            prompt_ids = self.tokenizer.encode(medical_prompt, return_tensors="pt").to(self.device)
            
            # Generate using LLaMA
            generation_config = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "early_stopping": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            generated_ids = self.llama.generate(
                input_ids=prompt_ids,
                **generation_config
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Remove prompt if present
            if medical_prompt in generated_text:
                generated_text = generated_text.replace(medical_prompt, "").strip()
            
            return generated_text
    
    def get_model(self) -> LlamaForCausalLM:
        """Get the underlying LLaMA model."""
        return self.llama
    
    def get_config(self) -> LlamaConfig:
        """Get the LLaMA configuration."""
        return self.llama.config
