"""
Factory for creating decoder instances.
Fixed version for Python compatibility.
"""

from transformers import PreTrainedTokenizer

from .base_decoder_fixed import TextDecoder
from .gpt2_decoder_fixed import GPT2Decoder


def create_decoder(
    decoder_type: str,
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    device: str = "cpu",
    **kwargs
) -> TextDecoder:
    """Create a decoder instance based on type.
    
    Args:
        decoder_type: Type of decoder ('gpt2', 't5', 'llama')
        model_name: Name of the model to load
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        device: Device to run on
        **kwargs: Additional arguments for decoder initialization
        
    Returns:
        Decoder instance
        
    Raises:
        ValueError: If decoder_type is not supported
    """
    decoder_type = decoder_type.lower()
    
    if decoder_type == "gpt2":
        return GPT2Decoder(
            model_name=model_name,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}. "
                       f"Supported types: gpt2")


def get_supported_decoders():
    """Get information about supported decoders.
    
    Returns:
        Dictionary mapping decoder types to their information
    """
    return {
        "gpt2": {
            "name": "GPT-2",
            "description": "Generative Pre-trained Transformer 2 with cross-attention",
            "models": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
            "supports_cross_attention": True,
            "medical_specialized": False
        }
    }


def get_decoder_recommendations():
    """Get decoder recommendations for medical image captioning.
    
    Returns:
        Dictionary with recommendations for different use cases
    """
    return {
        "best_overall": "gpt2",
        "best_quality": "gpt2", 
        "best_speed": "gpt2",
        "best_medical": "gpt2",
        "best_resource_efficient": "gpt2"
    }
