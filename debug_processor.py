#!/usr/bin/env python3

import sys
sys.path.append('/home/june/Code/LLaMA-Factory-Latest/src')

from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from transformers import AutoProcessor

# Test loading with the same parameters as your script
model_args, data_args, _, generating_args = get_infer_args(
    dict(
        model_name_or_path="/home/june/Code/new_llamafactory/saves/huggingface_origin/InternVL3-8B",
        dataset="coco_test",
        template="intern_vl",
        trust_remote_code=True,  # Make sure this is True, not "true" string
    )
)

print(f"model_args.trust_remote_code: {model_args.trust_remote_code}")
print(f"model_args.model_name_or_path: {model_args.model_name_or_path}")

# Try to load processor directly
print("\n=== Testing direct processor loading ===")
try:
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    print(f"Direct processor loading successful: {type(processor)}")
except Exception as e:
    print(f"Direct processor loading failed: {e}")
    import traceback
    traceback.print_exc()

# Let's manually test the processor loading logic from load_tokenizer
print("\n=== Manual processor loading test ===")
from llamafactory.model.loader import _get_init_kwargs

init_kwargs = _get_init_kwargs(model_args)
print(f"init_kwargs: {init_kwargs}")

try:
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        **init_kwargs,
    )
    print(f"Processor loaded with use_fast=True: {type(processor)}")
except ValueError as ve:
    print(f"ValueError with use_fast=True: {ve}")
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
        print(f"Processor loaded with use_fast=False: {type(processor)}")
    except Exception as e2:
        print(f"Failed with use_fast=False: {e2}")
        processor = None
except Exception as e:
    print(f"Failed to load processor: {e}")
    processor = None

if processor is not None and "Processor" not in processor.__class__.__name__:
    print(f"Dropping processor because it's not a Processor: {type(processor)}")
    processor = None

print(f"Final processor result: {processor is not None}")

print("\n=== Testing load_tokenizer function ===")
try:
    tokenizer_module = load_tokenizer(model_args)
    print(f"Tokenizer loaded: {tokenizer_module['tokenizer'] is not None}")
    print(f"Processor loaded: {tokenizer_module['processor'] is not None}")
    if tokenizer_module['processor'] is not None:
        print(f"Processor type: {type(tokenizer_module['processor'])}")
    else:
        print("ERROR: Processor is None!")
except Exception as e:
    print(f"ERROR loading tokenizer/processor: {e}")
    import traceback
    traceback.print_exc()
