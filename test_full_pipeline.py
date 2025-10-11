"""
Test script for full medical image captioning pipeline.
Usage: python test_full_pipeline.py
"""
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_full_pipeline():
    """Test full encoder-decoder pipeline."""
    print("🏥 Full Medical Image Captioning Pipeline Test")
    print("=" * 50)
    
    try:
        # Import required modules
        from src.encoders import ViTEncoder
        from src.decoders import GPT2Decoder
        from transformers import GPT2Tokenizer
        
        print("✅ Imports successful!")
        
        # Create encoder
        print("🏗️  Creating ViT encoder...")
        encoder = ViTEncoder(
            model_name="google/vit-base-patch16-224",
            freeze_encoder=False,
            device="cpu"
        )
        print(f"✅ ViT encoder created! Output dim: {encoder.get_output_dim()}")
        
        # Create decoder
        print("🏗️  Creating GPT-2 decoder...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        decoder = GPT2Decoder(
            model_name="gpt2",
            tokenizer=tokenizer,
            max_length=128,
            device="cpu",
            add_cross_attention=True
        )
        print(f"✅ GPT-2 decoder created!")
        
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
        with open("full_pipeline_test_result.txt", "w", encoding="utf-8") as f:
            f.write(f"Medical Image Captioning Pipeline Test Result\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Encoder: ViT (google/vit-base-patch16-224)\n")
            f.write(f"Decoder: GPT-2 (gpt2)\n")
            f.write(f"Image shape: {dummy_image.shape}\n")
            f.write(f"Features shape: {encoder_features.shape}\n")
            f.write(f"Generated caption: {generated_caption}\n")
        
        print(f"✅ Results saved to full_pipeline_test_result.txt")
        
        print(f"\n🎉 Full pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_encoders():
    """Test different encoder types."""
    print("\n🔍 Testing Different Encoders")
    print("=" * 40)
    
    try:
        from src.encoders import ViTEncoder, SwinEncoder, ResNetEncoder
        from src.decoders import GPT2Decoder
        from transformers import GPT2Tokenizer
        
        # Create tokenizer and decoder
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        decoder = GPT2Decoder(
            model_name="gpt2",
            tokenizer=tokenizer,
            max_length=128,
            device="cpu",
            add_cross_attention=True
        )
        
        # Test different encoders
        encoders = [
            ("ViT", ViTEncoder("google/vit-base-patch16-224")),
            ("Swin", SwinEncoder("microsoft/swin-tiny-patch4-window7-224")),
            ("ResNet", ResNetEncoder("resnet50"))
        ]
        
        dummy_image = torch.randn(1, 3, 224, 224)
        
        for encoder_name, encoder in encoders:
            print(f"\n🔍 Testing {encoder_name} encoder...")
            
            # Encode image
            features = encoder(dummy_image)
            print(f"   Features shape: {features.shape}")
            
            # Generate caption
            caption = decoder.generate(
                encoder_hidden_states=features,
                max_length=50,
                temperature=0.8
            )
            
            print(f"   Generated: '{caption[:50]}...'")
        
        print(f"\n✅ All encoder types tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Different encoders test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run full pipeline tests."""
    print("🧪 Medical Image Captioning - Full Pipeline Tests")
    print("=" * 60)
    
    tests = [
        test_full_pipeline,
        test_different_encoders
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All full pipeline tests passed!")
        print("✅ Encoder and decoder modules are working correctly together!")
    else:
        print("⚠️  Some full pipeline tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
