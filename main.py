"""
Main script to run medical image captioning pipeline directly from command line.
Usage: python main.py --configs configs/default_config.yaml
"""

import argparse
import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        from src.utils.config import Config
        config = Config.from_yaml(config_path)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return None

def create_model(config):
    """Create model based on configuration."""
    try:
        from src.decoders.decoder_factory import create_decoder
        from transformers import GPT2Tokenizer, T5Tokenizer, LlamaTokenizer
        
        # Get decoder type and model name from config
        decoder_type = config.model.decoder.type
        model_name = config.model.decoder.model_name
        
        print(f"🏗️  Creating {decoder_type.upper()} decoder with model: {model_name}")
        
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
        
        # Create decoder
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
        return decoder, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None, None

def generate_caption(decoder, config):
    """Generate caption using the decoder."""
    try:
        # Create dummy visual features (simulating image encoder output)
        if config.model.decoder.type == "gpt2":
            hidden_size = 768
        elif config.model.decoder.type == "t5":
            hidden_size = 512
        elif config.model.decoder.type == "llama":
            hidden_size = 4096
        else:
            hidden_size = 768
        
        encoder_hidden_states = torch.randn(1, 197, hidden_size)
        
        print("📝 Generating medical caption...")
        generated_text = decoder.generate(
            encoder_hidden_states=encoder_hidden_states,
            max_length=config.generation.max_length,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            do_sample=config.generation.do_sample,
            num_beams=config.generation.num_beams
        )
        
        return generated_text
        
    except Exception as e:
        print(f"❌ Failed to generate caption: {e}")
        return None

def main():
    """Main function to run the medical image captioning pipeline."""
    parser = argparse.ArgumentParser(description="Medical Image Captioning Pipeline")
    parser.add_argument("--configs", required=True, help="Path to configuration YAML file")
    parser.add_argument("--output", default="output.txt", help="Output file for generated caption")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🏥 Medical Image Captioning Pipeline")
    print("=" * 50)
    print(f"Config file: {args.configs}")
    print(f"Output file: {args.output}")
    print()
    
    # Load configuration
    config = load_config(args.configs)
    if config is None:
        return False
    
    # Create model
    decoder, tokenizer = create_model(config)
    if decoder is None or tokenizer is None:
        return False
    
    # Generate caption
    generated_text = generate_caption(decoder, config)
    if generated_text is None:
        return False
    
    # Display results
    print(f"\n📄 Generated Medical Caption:")
    print(f"   '{generated_text}'")
    
    # Save to output file
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"✅ Caption saved to {args.output}")
    except Exception as e:
        print(f"❌ Failed to save output: {e}")
        return False
    
    print(f"\n🎉 Medical image captioning completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
