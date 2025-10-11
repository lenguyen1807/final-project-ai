"""
Run medical image captioning with encoder and decoder.
Usage: python run_with_encoder.py --encoder vit --decoder gpt2
"""

import argparse
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run medical image captioning with encoder and decoder."""
    parser = argparse.ArgumentParser(description="Medical Image Captioning with Encoder and Decoder")
    parser.add_argument("--encoder", choices=["vit", "swin", "resnet"], default="vit", 
                       help="Choose encoder type")
    parser.add_argument("--decoder", choices=["gpt2", "t5", "llama"], default="gpt2", 
                       help="Choose decoder type")
    parser.add_argument("--encoder_model", default=None, help="Specific encoder model name")
    parser.add_argument("--decoder_model", default=None, help="Specific decoder model name")
    parser.add_argument("--output", default="encoder_decoder_output.txt", help="Output file")
    
    args = parser.parse_args()
    
    print("🏥 Medical Image Captioning with Encoder and Decoder")
    print("=" * 60)
    print(f"Encoder: {args.encoder}")
    print(f"Decoder: {args.decoder}")
    print(f"Output: {args.output}")
    print()
    
    try:
        # Import required modules
        from src.encoders.encoder_factory import create_encoder
        from src.decoders.decoder_factory import create_decoder
        from transformers import GPT2Tokenizer, T5Tokenizer, LlamaTokenizer
        from torch.utils.tensorboard import SummaryWriter

        print("✅ Imports successful!")
        
        # Set model names if not provided
        if args.encoder_model is None:
            if args.encoder == "vit":
                args.encoder_model = "google/vit-base-patch16-224"
            elif args.encoder == "swin":
                args.encoder_model = "microsoft/swin-tiny-patch4-window7-224"
            elif args.encoder == "resnet":
                args.encoder_model = "resnet50"
        
        if args.decoder_model is None:
            if args.decoder == "gpt2":
                args.decoder_model = "gpt2"
            elif args.decoder == "t5":
                args.decoder_model = "t5-small"
            elif args.decoder == "llama":
                args.decoder_model = "meta-llama/Llama-2-7b-hf"
        
        # Create encoder
        print(f"🏗️  Creating {args.encoder.upper()} encoder with model: {args.encoder_model}")
        encoder = create_encoder(
            encoder_type=args.encoder,
            model_name=args.encoder_model,
            freeze_encoder=False,
            device="cpu"
        )
        print(f"✅ {args.encoder.upper()} encoder created successfully!")
        print(f"   - Output dim: {encoder.get_output_dim()}")
        print(f"   - Sequence length: {encoder.get_sequence_length()}")
        
        # Create decoder
        print(f"🏗️  Creating {args.decoder.upper()} decoder with model: {args.decoder_model}")
        
        # Initialize appropriate tokenizer
        if args.decoder == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(args.decoder_model)
            tokenizer.pad_token = tokenizer.eos_token
        elif args.decoder == "t5":
            tokenizer = T5Tokenizer.from_pretrained(args.decoder_model)
        elif args.decoder == "llama":
            tokenizer = LlamaTokenizer.from_pretrained(args.decoder_model)
        
        decoder = create_decoder(
            decoder_type=args.decoder,
            model_name=args.decoder_model,
            tokenizer=tokenizer,
            max_length=128,
            device="cpu",
            add_cross_attention=(args.decoder == "gpt2")
        )
        print(f"✅ {args.decoder.upper()} decoder created successfully!")
        
        # Create dummy chest X-ray image
        print("🖼️  Creating dummy chest X-ray image...")
        dummy_image = torch.randn(1, 3, 224, 224)  # Single chest X-ray
        print(f"   Image shape: {dummy_image.shape}")
        
        # Encode image
        print("🔍 Encoding image...")
        encoder_features = encoder(dummy_image)
        print(f"✅ Image encoded! Features shape: {encoder_features.shape}")
        
        # Generate medical caption
        print("📝 Generating medical caption...")
        generated_caption = decoder.generate(
            encoder_hidden_states=encoder_features,
            max_length=100,
            temperature=0.8,
            top_p=0.9
        )
        
        # Display results
        print(f"\n📄 Generated Medical Caption:")
        print(f"   '{generated_caption}'")
        
        # Save results
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"Medical Image Captioning Pipeline Result\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Encoder: {args.encoder.upper()} ({args.encoder_model})\n")
            f.write(f"Decoder: {args.decoder.upper()} ({args.decoder_model})\n")
            f.write(f"Image shape: {dummy_image.shape}\n")
            f.write(f"Features shape: {encoder_features.shape}\n")
            f.write(f"Generated caption: {generated_caption}\n")
        
        print(f"✅ Results saved to {args.output}")
        
        print(f"\n🎉 Medical image captioning with encoder and decoder completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
