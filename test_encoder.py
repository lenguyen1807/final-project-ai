"""
Test script for encoder module.
Usage: python test_encoder.py
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_vit_encoder():
    """Test ViT encoder."""
    print("🔍 Testing ViT Encoder")
    print("=" * 30)
    
    try:
        from src.encoders import ViTEncoder
        
        # Create ViT encoder
        encoder = ViTEncoder(
            model_name="google/vit-base-patch16-224",
            freeze_encoder=False,
            device="cpu"
        )
        
        print(f"✅ ViT encoder created successfully!")
        print(f"   - Output dim: {encoder.get_output_dim()}")
        print(f"   - Sequence length: {encoder.get_sequence_length()}")
        
        # Test forward pass
        dummy_images = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        features = encoder(dummy_images)
        
        print(f"✅ Forward pass successful!")
        print(f"   - Input shape: {dummy_images.shape}")
        print(f"   - Output shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ViT encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_swin_encoder():
    """Test Swin encoder."""
    print("\n🔍 Testing Swin Encoder")
    print("=" * 30)
    
    try:
        from src.encoders import SwinEncoder
        
        # Create Swin encoder
        encoder = SwinEncoder(
            model_name="microsoft/swin-tiny-patch4-window7-224",
            freeze_encoder=False,
            device="cpu"
        )
        
        print(f"✅ Swin encoder created successfully!")
        print(f"   - Output dim: {encoder.get_output_dim()}")
        print(f"   - Sequence length: {encoder.get_sequence_length()}")
        
        # Test forward pass
        dummy_images = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        features = encoder(dummy_images)
        
        print(f"✅ Forward pass successful!")
        print(f"   - Input shape: {dummy_images.shape}")
        print(f"   - Output shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Swin encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resnet_encoder():
    """Test ResNet encoder."""
    print("\n🔍 Testing ResNet Encoder")
    print("=" * 30)
    
    try:
        from src.encoders import ResNetEncoder
        
        # Create ResNet encoder
        encoder = ResNetEncoder(
            model_name="resnet50",
            freeze_encoder=False,
            device="cpu"
        )
        
        print(f"✅ ResNet encoder created successfully!")
        print(f"   - Output dim: {encoder.get_output_dim()}")
        print(f"   - Sequence length: {encoder.get_sequence_length()}")
        
        # Test forward pass
        dummy_images = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        features = encoder(dummy_images)
        
        print(f"✅ Forward pass successful!")
        print(f"   - Input shape: {dummy_images.shape}")
        print(f"   - Output shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ResNet encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder_factory():
    """Test encoder factory."""
    print("\n🏭 Testing Encoder Factory")
    print("=" * 30)
    
    try:
        from src.encoders.encoder_factory import create_encoder, get_supported_encoders
        
        # Show supported encoders
        print("📋 Supported encoders:")
        supported = get_supported_encoders()
        for encoder_type, info in supported.items():
            print(f"   - {encoder_type}: {info['name']}")
        
        # Test factory creation
        encoder = create_encoder(
            encoder_type="vit",
            model_name="google/vit-base-patch16-224",
            freeze_encoder=False,
            device="cpu"
        )
        
        print(f"✅ Factory pattern successful! Created: {type(encoder).__name__}")
        
        # Test forward pass
        dummy_images = torch.randn(1, 3, 224, 224)
        features = encoder(dummy_images)
        print(f"✅ Forward pass successful! Output shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Encoder factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder_decoder_integration():
    """Test encoder-decoder integration."""
    print("\n🔗 Testing Encoder-Decoder Integration")
    print("=" * 40)
    
    try:
        from src.encoders import ViTEncoder
        from src.decoders import GPT2Decoder
        from transformers import GPT2Tokenizer
        
        # Create encoder
        encoder = ViTEncoder(
            model_name="google/vit-base-patch16-224",
            freeze_encoder=False,
            device="cpu"
        )
        
        # Create decoder
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        decoder = GPT2Decoder(
            model_name="gpt2",
            tokenizer=tokenizer,
            max_length=128,
            device="cpu",
            add_cross_attention=True
        )
        
        print("✅ Encoder and decoder created successfully!")
        
        # Test integration
        dummy_images = torch.randn(1, 3, 224, 224)
        
        # Encode images
        encoder_features = encoder(dummy_images)
        print(f"✅ Image encoding successful! Features shape: {encoder_features.shape}")
        
        # Generate caption
        generated_text = decoder.generate(
            encoder_hidden_states=encoder_features,
            max_length=50,
            temperature=0.8
        )
        
        print(f"✅ Caption generation successful!")
        print(f"   Generated: '{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Encoder-decoder integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all encoder tests."""
    print("🧪 Medical Image Captioning - Encoder Module Tests")
    print("=" * 60)
    
    tests = [
        test_vit_encoder,
        test_swin_encoder,
        test_resnet_encoder,
        test_encoder_factory,
        test_encoder_decoder_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All encoder tests passed! The encoder module is working correctly.")
    else:
        print("⚠️  Some encoder tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
