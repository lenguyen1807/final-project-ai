"""
Quick test script for the end-to-end pipeline.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_loading():
    """Test data loading functionality."""
    print("🔍 Testing data loading...")
    
    try:
        from end_to_end_pipeline import EndToEndPipeline
        
        pipeline = EndToEndPipeline()
        train_df, val_df, test_df = pipeline.load_data()
        
        if train_df is not None:
            print("✅ Data loading successful!")
            print(f"   Train: {len(train_df)} samples")
            print(f"   Val: {len(val_df)} samples")
            print(f"   Test: {len(test_df)} samples")
            return True
        else:
            print("❌ Data loading failed!")
            return False
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_model_setup():
    """Test model setup functionality."""
    print("\n🔍 Testing model setup...")
    
    try:
        from end_to_end_pipeline import EndToEndPipeline
        
        pipeline = EndToEndPipeline()
        success = pipeline.setup_model(decoder_type="gpt2", epochs=1, batch_size=2)
        
        if success:
            print("✅ Model setup successful!")
            return True
        else:
            print("❌ Model setup failed!")
            return False
            
    except Exception as e:
        print(f"❌ Model setup test failed: {e}")
        return False

def test_quick_pipeline():
    """Test quick pipeline run."""
    print("\n🔍 Testing quick pipeline...")
    
    try:
        from end_to_end_pipeline import EndToEndPipeline
        
        pipeline = EndToEndPipeline()
        
        # Load data
        train_df, val_df, test_df = pipeline.load_data()
        if train_df is None:
            return False
        
        # Setup model
        if not pipeline.setup_model(decoder_type="gpt2", epochs=1, batch_size=2):
            return False
        
        # Test inference only (skip training for quick test)
        print("📝 Testing inference...")
        inference_results = pipeline.inference(test_df, num_samples=2)
        
        if inference_results:
            print("✅ Quick pipeline test successful!")
            return True
        else:
            print("❌ Quick pipeline test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Quick pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing End-to-End Pipeline")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Setup", test_model_setup),
        ("Quick Pipeline", test_quick_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Pipeline is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
