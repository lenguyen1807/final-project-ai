"""
Simple runner script for quick execution.
Usage: python run.py
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def quick_run():
    """Quick run of the medical image captioning pipeline."""
    print("🚀 Quick Run - Medical Image Captioning")
    print("=" * 50)
    
    try:
        from src.decoders import GPT2Decoder
        from src.decoders.decoder_factory import create_decoder
        from transformers import GPT2Tokenizer
        import torch
        
        print("✅ Imports successful!")
        
        # Initialize tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create decoder
        print("🏗️  Creating decoder...")
        decoder = create_decoder(
            decoder_type="gpt2",
            model_name="gpt2",
            tokenizer=tokenizer,
            max_length=128,
            device="cpu",
            add_cross_attention=True
        )
        
        # Create dummy chest X-ray features
        print("🖼️  Creating dummy chest X-ray features...")
        encoder_hidden_states = torch.randn(1, 197, 768)  # ViT features
        
        # Generate medical caption
        print("📝 Generating medical caption...")
        generated_text = decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            max_length=100,
            temperature=0.8,
            top_p=0.9
        )
        
        print(f"\n🎉 SUCCESS!")
        print(f"📄 Generated medical caption:")
        print(f"   '{generated_text}'")
        print(f"\n✅ Medical image captioning pipeline is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_run()
    sys.exit(0 if success else 1)
