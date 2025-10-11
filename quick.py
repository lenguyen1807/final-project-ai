"""
Quick script to run medical image captioning with minimal setup.
Usage: python quick.py
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Quick run of medical image captioning."""
    print("🚀 Quick Medical Image Captioning")
    print("=" * 40)
    
    try:
        # Import required modules
        from src.decoders import GPT2Decoder
        from transformers import GPT2Tokenizer
        
        print("✅ Imports successful!")
        
        # Initialize tokenizer
        print("📝 Loading GPT-2 tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create decoder
        print("🏗️  Creating GPT-2 decoder...")
        decoder = GPT2Decoder(
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
        
        # Display results
        print(f"\n📄 Generated Medical Caption:")
        print(f"   '{generated_text}'")
        
        # Save to file
        with open("generated_caption.txt", "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"✅ Caption saved to generated_caption.txt")
        
        print(f"\n🎉 Medical image captioning completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
