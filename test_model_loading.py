#!/usr/bin/env python3

import torch
from transformers import AutoProcessor, AutoTokenizer, AutoConfig

model_path = "/home/june/Code/new_llamafactory/saves/huggingface_origin/InternVL3-8B"

print("=" * 60)
print("Testing InternVL3-8B Model and Processor Loading")
print("=" * 60)

# Test 1: Load config
print("\n1. Loading config...")
try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"✅ Config loaded successfully: {config.__class__.__name__}")
    print(f"   Model type: {config.model_type}")
    print(f"   Architecture: {config.architectures}")
except Exception as e:
    print(f"❌ Failed to load config: {e}")

# Test 2: Load tokenizer
print("\n2. Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"✅ Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"❌ Failed to load tokenizer: {e}")

# Test 3: Load processor
print("\n3. Loading processor...")
try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"✅ Processor loaded successfully: {processor.__class__.__name__}")
    
    # Check processor components
    if hasattr(processor, 'tokenizer'):
        print(f"   Has tokenizer: {processor.tokenizer.__class__.__name__}")
    
    if hasattr(processor, 'image_processor'):
        print(f"   Has image_processor: {processor.image_processor.__class__.__name__}")
    
    if hasattr(processor, 'feature_extractor'):
        print(f"   Has feature_extractor: {processor.feature_extractor.__class__.__name__}")
        
    # List all attributes of processor
    print(f"   Processor attributes: {[attr for attr in dir(processor) if not attr.startswith('_')]}")
    
except Exception as e:
    print(f"❌ Failed to load processor: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test processor with a simple text
print("\n4. Testing processor with text...")
try:
    if 'processor' in locals():
        text = "Hello, how are you?"
        inputs = processor(text=text, return_tensors="pt")
        print(f"✅ Text processing successful")
        print(f"   Input keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"   {key} shape: {value.shape}")
except Exception as e:
    print(f"❌ Failed to process text: {e}")

# Test 5: Check if this is a vision-language model
print("\n5. Checking vision capabilities...")
try:
    if 'config' in locals():
        # Check for vision-related attributes
        vision_attrs = ['vision_config', 'vision_tower', 'mm_vision_tower']
        for attr in vision_attrs:
            if hasattr(config, attr):
                print(f"   Found vision attribute: {attr}")
        
        # Check model architecture
        if hasattr(config, 'architectures'):
            arch = config.architectures[0] if config.architectures else "Unknown"
            print(f"   Model architecture: {arch}")
            
except Exception as e:
    print(f"❌ Failed to check vision capabilities: {e}")

print("\n" + "=" * 60)
print("Model loading test completed!")
print("=" * 60)
