"""
Run medical image captioning using factory pattern.
Usage: python run_factory.py --decoder gpt2
"""

import argparse
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run medical image captioning using factory pattern."""
    parser = argparse.ArgumentParser(description="Medical Image Captioning with Factory Pattern")
    parser.add_argument("--decoder", choices=["gpt2", "t5", "llama"], default="gpt2", 
                       help="Choose decoder type")
    parser.add_argument("--model", default=None, help="Specific model name")
    parser.add_argument("--output", default="factory_output.txt", help="Output file")
    
    args = parser.parse_args()
    
    print("🏭 Factory Pattern Medical Image Captioning")
    print("=" * 50)
    print(f"Decoder: {args.decoder}")
    print(f"Output: {args.output}")
    print()
    
    try:
        # Import required modules
        from src.decoders.decoder_factory import create_decoder, get_supported_decoders
        from transformers import GPT2Tokenizer, T5Tokenizer, LlamaTokenizer
        
        print("✅ Imports successful!")
        
        # Show supported decoders
        print("📋 Supported decoders:")
        supported = get_supported_decoders()
        for decoder_type, info in supported.items():
            print(f"   - {decoder_type}: {info['name']}")
        
        # Set model name if not provided
        if args.model is None:
            if args.decoder == "gpt2":
                args.model = "gpt2"
            elif args.decoder == "t5":
                args.model = "t5-small"
            elif args.decoder == "llama":
                args.model = "meta-llama/Llama-2-7b-hf"
        
        print(f"\n🏗️  Creating {args.decoder.upper()} decoder with model: {args.model}")
        
        # Initialize appropriate tokenizer
        if args.decoder == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(args.model)
            tokenizer.pad_token = tokenizer.eos_token
        elif args.decoder == "t5":
            tokenizer = T5Tokenizer.from_pretrained(args.model)
        elif args.decoder == "llama":
            tokenizer = LlamaTokenizer.from_pretrained(args.model)
        
        # Create decoder using factory
        decoder = create_decoder(
            decoder_type=args.decoder,
            model_name=args.model,
            tokenizer=tokenizer,
            max_length=128,
            device="cpu",
            add_cross_attention=(args.decoder == "gpt2")
        )
        
        print(f"✅ {args.decoder.upper()} decoder created successfully!")
        
        # Create dummy visual features
        print("🖼️  Creating dummy visual features...")
        if args.decoder == "gpt2":
            hidden_size = 768
        elif args.decoder == "t5":
            hidden_size = 512
        elif args.decoder == "llama":
            hidden_size = 4096
        
        encoder_hidden_states = torch.randn(1, 197, hidden_size)
        
        # Generate caption
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
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"✅ Caption saved to {args.output}")
        
        print(f"\n🎉 Factory pattern medical image captioning completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
