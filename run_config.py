"""
Run medical image captioning with configuration file.
Usage: python run_config.py --configs configs/default_config.yaml
"""

import argparse
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run medical image captioning with configuration."""
    parser = argparse.ArgumentParser(description="Medical Image Captioning with Config")
    parser.add_argument("--configs", required=True, help="Path to configuration YAML file")
    parser.add_argument("--output", default="config_output.txt", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("⚙️  Config-Driven Medical Image Captioning")
    print("=" * 50)
    print(f"Config file: {args.configs}")
    print(f"Output file: {args.output}")
    print()
    
    try:
        # Import required modules
        from src.utils.config import Config
        from src.decoders.decoder_factory import create_decoder
        from transformers import GPT2Tokenizer, T5Tokenizer, LlamaTokenizer
        
        print("✅ Imports successful!")
        
        # Load configuration
        print(f"📄 Loading configuration from {args.configs}...")
        config = Config.from_yaml(args.configs)
        print(f"✅ Configuration loaded successfully!")
        
        if args.verbose:
            print(f"   - Dataset: {config.dataset.name}")
            print(f"   - Model: {config.model.name}")
            print(f"   - Encoder: {config.model.encoder.type}")
            print(f"   - Decoder: {config.model.decoder.type}")
            print(f"   - Batch size: {config.training.batch_size}")
            print(f"   - Learning rate: {config.training.learning_rate}")
        
        # Get decoder configuration
        decoder_type = config.model.decoder.type
        model_name = config.model.decoder.model_name
        
        print(f"\n🏗️  Creating {decoder_type.upper()} decoder with model: {model_name}")
        
        # Initialize appropriate tokenizer
        if decoder_type == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
        elif decoder_type == "t5":
            tokenizer = T5Tokenizer.from_pretrained(model_name)
        elif decoder_type == "llama":
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        
        # Create decoder using factory
        decoder = create_decoder(
            decoder_type=decoder_type,
            model_name=model_name,
            tokenizer=tokenizer,
            max_length=config.dataset.max_caption_length,
            device=config.system.device,
            add_cross_attention=config.model.decoder.add_cross_attention,
            freeze_decoder=config.model.decoder.freeze_decoder
        )
        
        print(f"✅ {decoder_type.upper()} decoder created successfully!")
        
        # Create dummy visual features based on decoder type
        print("🖼️  Creating dummy visual features...")
        if decoder_type == "gpt2":
            hidden_size = 768
        elif decoder_type == "t5":
            hidden_size = 512
        elif decoder_type == "llama":
            hidden_size = 4096
        else:
            hidden_size = 768
        
        encoder_hidden_states = torch.randn(1, 197, hidden_size)
        
        # Generate caption using config parameters
        print("📝 Generating medical caption...")
        generated_text = decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            max_length=config.generation.max_length,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            do_sample=config.generation.do_sample,
            num_beams=config.generation.num_beams
        )
        
        # Display results
        print(f"\n📄 Generated Medical Caption:")
        print(f"   '{generated_text}'")
        
        # Save to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"✅ Caption saved to {args.output}")
        
        print(f"\n🎉 Config-driven medical image captioning completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
