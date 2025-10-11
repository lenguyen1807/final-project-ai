"""
Demo script showing how to use different decoders for medical image captioning.
"""

import torch
from transformers import GPT2Tokenizer, T5Tokenizer, LlamaTokenizer
from src.decoders import GPT2Decoder, T5Decoder, LLaMADecoder
from src.decoders.decoder_factory import create_decoder, get_supported_decoders


def demo_gpt2_decoder():
    """Demo GPT-2 decoder usage."""
    print("=== GPT-2 Decoder Demo ===")
    
    # Initialize tokenizer and decoder
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    decoder = GPT2Decoder(
        model_name="gpt2",
        tokenizer=tokenizer,
        max_length=128,
        device="cpu",
        add_cross_attention=True,
        freeze_decoder=False
    )
    
    # Create dummy encoder hidden states (simulating image features)
    batch_size, seq_len, hidden_size = 1, 197, 768  # ViT patch size
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create dummy input
    input_ids = torch.tensor([[tokenizer.bos_token_id, tokenizer.eos_token_id]])
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    outputs = decoder.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states
    )
    
    print(f"GPT-2 Output shape: {outputs['logits'].shape}")
    print(f"GPT-2 Loss: {outputs['loss']}")
    
    # Generate text
    generated_text = decoder.generate(
        encoder_hidden_states=encoder_hidden_states,
        max_length=50,
        temperature=0.8,
        top_p=0.9
    )
    
    print(f"Generated text: {generated_text}")
    print()


def demo_t5_decoder():
    """Demo T5 decoder usage."""
    print("=== T5 Decoder Demo ===")
    
    # Initialize tokenizer and decoder
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    decoder = T5Decoder(
        model_name="t5-small",
        tokenizer=tokenizer,
        max_length=128,
        device="cpu",
        freeze_decoder=False
    )
    
    # Create dummy encoder hidden states
    batch_size, seq_len, hidden_size = 1, 197, 512  # T5 hidden size
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create dummy input
    input_ids = torch.tensor([[tokenizer.bos_token_id, tokenizer.eos_token_id]])
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    outputs = decoder.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states
    )
    
    print(f"T5 Output shape: {outputs['logits'].shape}")
    print(f"T5 Loss: {outputs['loss']}")
    
    # Generate text
    generated_text = decoder.generate(
        encoder_hidden_states=encoder_hidden_states,
        max_length=50,
        temperature=0.8,
        top_p=0.9
    )
    
    print(f"Generated text: {generated_text}")
    print()


def demo_llama_decoder():
    """Demo LLaMA decoder usage."""
    print("=== LLaMA Decoder Demo ===")
    
    try:
        # Initialize tokenizer and decoder
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        decoder = LLaMADecoder(
            model_name="meta-llama/Llama-2-7b-hf",
            tokenizer=tokenizer,
            max_length=128,
            device="cpu",
            freeze_decoder=False
        )
        
        # Create dummy encoder hidden states
        batch_size, seq_len, hidden_size = 1, 197, 4096  # LLaMA hidden size
        encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create dummy input
        input_ids = torch.tensor([[tokenizer.bos_token_id, tokenizer.eos_token_id]])
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass
        outputs = decoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states
        )
        
        print(f"LLaMA Output shape: {outputs['logits'].shape}")
        print(f"LLaMA Loss: {outputs['loss']}")
        
        # Generate text
        generated_text = decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            max_length=50,
            temperature=0.8,
            top_p=0.9
        )
        
        print(f"Generated text: {generated_text}")
        
    except Exception as e:
        print(f"LLaMA demo failed (likely due to model access): {e}")
    
    print()


def demo_factory_pattern():
    """Demo factory pattern for creating decoders."""
    print("=== Decoder Factory Demo ===")
    
    # Show supported decoders
    supported_decoders = get_supported_decoders()
    print("Supported decoders:")
    for decoder_type, info in supported_decoders.items():
        print(f"  {decoder_type}: {info['name']} - {info['description']}")
    
    print()
    
    # Create decoder using factory
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        decoder = create_decoder(
            decoder_type="gpt2",
            model_name="gpt2",
            tokenizer=tokenizer,
            max_length=128,
            device="cpu",
            add_cross_attention=True
        )
        
        print(f"Created decoder: {type(decoder).__name__}")
        print(f"Model name: {decoder.model_name}")
        print(f"Vocab size: {decoder.get_vocab_size()}")
        
    except Exception as e:
        print(f"Factory demo failed: {e}")
    
    print()


def demo_medical_specialization():
    """Demo medical specialization features."""
    print("=== Medical Specialization Demo ===")
    
    # Medical-specific prompts
    medical_prompts = [
        "Generate a medical report for this chest X-ray image:",
        "Describe the findings in this medical image:",
        "Provide a clinical interpretation of this radiology image:"
    ]
    
    print("Medical prompts for different decoders:")
    for prompt in medical_prompts:
        print(f"  - {prompt}")
    
    print()
    
    # Show how different decoders handle medical text
    print("Decoder characteristics for medical use:")
    print("  GPT-2: Good for general medical text, supports cross-attention")
    print("  T5: Excellent for structured medical reports, text-to-text format")
    print("  LLaMA: Strong language understanding, good for complex medical descriptions")
    
    print()


if __name__ == "__main__":
    print("Medical Image Captioning - Decoder Module Demo")
    print("=" * 50)
    print()
    
    # Run demos
    demo_gpt2_decoder()
    demo_t5_decoder()
    demo_llama_decoder()
    demo_factory_pattern()
    demo_medical_specialization()
    
    print("Demo completed!")
