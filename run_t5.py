"""
Run medical image captioning with T5 decoder.
Usage: python run_t5.py
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run T5 medical image captioning."""
    print("🔄 T5 Medical Image Captioning")
    print("=" * 40)
    
    try:
        # Import required modules
        from src.decoders import T5Decoder
        from transformers import T5Tokenizer
        
        print("✅ Imports successful!")
        
        # Initialize tokenizer
        print("📝 Loading T5 tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        # Create decoder
        print("🏗️  Creating T5 decoder...")
        decoder = T5Decoder(
            model_name="t5-small",
            tokenizer=tokenizer,
            max_length=128,
            device="cpu"
        )
        
        # Create dummy visual features
        print("🖼️  Creating dummy visual features...")
        encoder_hidden_states = torch.randn(1, 197, 512)  # T5 hidden size
        
        # Generate medical report
        print("📝 Generating medical report...")
        generated_text = decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            max_length=100,
            temperature=0.8,
            top_p=0.9
        )
        
        # Display results
        print(f"\n📄 Generated Medical Report:")
        print(f"   '{generated_text}'")
        
        # Save to file
        with open("t5_generated_report.txt", "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"✅ Report saved to t5_generated_report.txt")
        
        print(f"\n🎉 T5 medical image captioning completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
