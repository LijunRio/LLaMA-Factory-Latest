#!/usr/bin/env python3

import sys
sys.path.append('/home/june/Code/LLaMA-Factory-Latest/src')

from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

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
