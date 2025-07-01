# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
我新改的vllm_infer_new.py 使用批处理来open文件，但是回报错，没办法正常运行：
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 67.86it/s, est. speed input: 26644.89 toks/s, output: 4335.03 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 74.83it/s, est. speed input: 29423.29 toks/s, output: 4605.91 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 40.76it/s, est. speed input: 16003.92 toks/s, output: 2778.55 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 51.86it/s, est. speed input: 20376.46 toks/s, output: 3407.86 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 75.40it/s, est. speed input: 29625.84 toks/s, output: 4851.91 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 37.81it/s, est. speed input: 14845.53 toks/s, output: 2406.87 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 51.54it/s, est. speed input: 20225.43 toks/s, output: 3422.93 toks/s]
Resource status after batch 30:██████████████████████████▉ | 31/32 [00:00<00:00, 60.25it/s, est. speed input: 20296.11 toks/s, output: 3379.85 toks/s]
Open files: 0
Memory usage: 6674.49 MB
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 39.50it/s, est. speed input: 15498.70 toks/s, output: 2523.02 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 50.58it/s, est. speed input: 19882.54 toks/s, output: 3341.37 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 45.62it/s, est. speed input: 17928.94 toks/s, output: 3209.34 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 42.53it/s, est. speed input: 16690.54 toks/s, output: 2756.87 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 42.85it/s, est. speed input: 16822.76 toks/s, output: 2732.11 toks/s]
Processed prompts: 100%|███████████████████████████████████| 32/32 [00:00<00:00, 47.96it/s, est. speed input: 18820.08 toks/s, output: 3048.46 toks/s]
Processing batched inference:  56%|██████████████████████████████████████████████▌                                    | 37/66 [02:08<01:16,  2.64s/it]
ERROR 07-01 17:00:59 [core.py:387] EngineCore hit an exception: Traceback (most recent call last):
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/cachetools/__init__.py", line 68, in __getitem__
ERROR 07-01 17:00:59 [core.py:387]     return self.__data[key]
ERROR 07-01 17:00:59 [core.py:387]            ~~~~~~~~~~~^^^^^
ERROR 07-01 17:00:59 [core.py:387] KeyError: '1ec007c256916d72e00681199e5e467a489a2b307e30b6dbbf168dc8d54f85bd'
ERROR 07-01 17:00:59 [core.py:387] 
ERROR 07-01 17:00:59 [core.py:387] During handling of the above exception, another exception occurred:
ERROR 07-01 17:00:59 [core.py:387] 
ERROR 07-01 17:00:59 [core.py:387] Traceback (most recent call last):
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 380, in run_engine_core
ERROR 07-01 17:00:59 [core.py:387]     engine_core.run_busy_loop()
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 400, in run_busy_loop
ERROR 07-01 17:00:59 [core.py:387]     self._process_input_queue()
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 425, in _process_input_queue
ERROR 07-01 17:00:59 [core.py:387]     self._handle_client_request(*req)
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 441, in _handle_client_request
ERROR 07-01 17:00:59 [core.py:387]     self.add_request(request)
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 177, in add_request
ERROR 07-01 17:00:59 [core.py:387]     request.mm_inputs = self.mm_input_cache_server.get_and_update_p1(
ERROR 07-01 17:00:59 [core.py:387]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/vllm/v1/engine/mm_input_cache.py", line 76, in get_and_update_p1
ERROR 07-01 17:00:59 [core.py:387]     mm_input = self.mm_cache[mm_hash]
ERROR 07-01 17:00:59 [core.py:387]                ~~~~~~~~~~~~~^^^^^^^^^
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/cachetools/__init__.py", line 211, in __getitem__
ERROR 07-01 17:00:59 [core.py:387]     value = cache_getitem(self, key)
ERROR 07-01 17:00:59 [core.py:387]             ^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/cachetools/__init__.py", line 70, in __getitem__
ERROR 07-01 17:00:59 [core.py:387]     return self.__missing__(key)
ERROR 07-01 17:00:59 [core.py:387]            ^^^^^^^^^^^^^^^^^^^^^
ERROR 07-01 17:00:59 [core.py:387]   File "/root/miniconda3/envs/llamafactory/lib/python3.11/site-packages/cachetools/__init__.py", line 97, in __missing__
ERROR 07-01 17:00:59 [core.py:387]     raise KeyError(key)
ERROR 07-01 17:00:59 [core.py:387] KeyError: '1ec007c256916d72e00681199e5e467a489a2b307e30b6dbbf168dc8d54f85bd'
ERROR 07-01 17:00:59 [core.py:387] 
CRITICAL 07-01 17:00:59 [core_client.py:359] Got fatal signal from worker processes, shutting down. See stack trace above for root cause issue.
Killed
(llamafactory) root@mcml-dgx-002:/home/june/Code/LLaMA-Factory-Latest# 
(llamafactory) root@mcml-dgx-002:/home/june/Code/LLaMA-Factory-Latest# 

你能帮我分析错误，并觉得应该如何解决呢？
"""

import gc
import json
import os
import resource
from typing import Optional
import warnings
try:
    import psutil
except ImportError:
    warnings.warn("psutil not installed, resource monitoring will be limited")

import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 1024,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)

    # load datasets
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # Set file descriptor limits
    try:
        # Get current limits
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current file descriptor limits: soft={soft_limit}, hard={hard_limit}")
        
        # Try to increase soft limit if needed
        if soft_limit < 4096:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard_limit), hard_limit))
                print(f"Increased file descriptor limit to {min(4096, hard_limit)}")
            except ValueError:
                print(f"Could not increase file descriptor limit, keeping at {soft_limit}")
    except Exception as e:
        print(f"Warning: Could not check/set file descriptor limits: {e}")

    # Monitor initial resources
    print("Initial resource status:")
    monitor_resources()

    # Reduce batch size for multimodal data to avoid too many open files
    effective_batch_size = min(batch_size, 32) if any(train_dataset[0][key] is not None for key in ["images", "videos", "audios"] if key in train_dataset.features) else batch_size
    print(f"Using effective batch size: {effective_batch_size}")

    # Store all results in these lists
    all_prompts, all_preds, all_labels = [], [], []

    # Add batch process to avoid the issue of too many files opened
    for i in tqdm(range(0, len(train_dataset), effective_batch_size), desc="Processing batched inference"):
        vllm_inputs, prompts, labels = [], [], []
        batch = train_dataset[i : min(i + effective_batch_size, len(train_dataset))]

        try:
            successful_inputs = 0
            for j in range(len(batch["input_ids"])):
                try:
                    multi_modal_data = None
                    
                    # Process multimodal data with proper resource management
                    if batch["images"][j] is not None:
                        multi_modal_data = safe_process_multimodal_data(
                            template_obj, batch["images"][j], "image",
                            image_max_pixels=image_max_pixels,
                            image_min_pixels=image_min_pixels
                        )
                    elif batch["videos"][j] is not None:
                        multi_modal_data = safe_process_multimodal_data(
                            template_obj, batch["videos"][j], "video",
                            image_max_pixels=image_max_pixels,
                            image_min_pixels=image_min_pixels,
                            video_fps=video_fps,
                            video_maxlen=video_maxlen
                        )
                    elif batch["audios"][j] is not None:
                        multi_modal_data = safe_process_multimodal_data(
                            template_obj, batch["audios"][j], "audio",
                            sampling_rate=16000
                        )

                    # Only add the input if multimodal processing was successful or not needed
                    if multi_modal_data is not None or all(batch[key][j] is None for key in ["images", "videos", "audios"]):
                        vllm_inputs.append({"prompt_token_ids": batch["input_ids"][j], "multi_modal_data": multi_modal_data})
                        prompts.append(tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens))
                        labels.append(
                            tokenizer.decode(
                                list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                                skip_special_tokens=skip_special_tokens,
                            )
                        )
                        successful_inputs += 1
                        
                    # Force cleanup after each sample
                    gc.collect()
                    
                except Exception as e:
                    print(f"Warning: Failed to process sample {i+j}: {e}")
                    continue

            # Generate predictions for this batch if we have any valid inputs
            if vllm_inputs:
                try:
                    results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
                    preds = [result.outputs[0].text for result in results]
                    
                    # Accumulate results
                    all_prompts.extend(prompts)
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                    
                    print(f"Successfully processed {successful_inputs}/{len(batch['input_ids'])} samples in batch {i//effective_batch_size}")
                    
                except Exception as e:
                    print(f"Error during generation for batch {i//effective_batch_size}: {e}")
                    
            else:
                print(f"No valid inputs in batch {i//effective_batch_size}")
                
        except Exception as e:
            print(f"Error processing batch {i//effective_batch_size}: {e}")
            continue
            
        finally:
            # Force cleanup after each batch
            del vllm_inputs, prompts, labels
            if 'results' in locals():
                del results
            if 'preds' in locals():
                del preds
            gc.collect()
            
            # Monitor resources periodically
            if i % (effective_batch_size * 10) == 0:
                print(f"\nResource status after batch {i//effective_batch_size}:")
                monitor_resources()
                print()

    # Write all results at once outside the loop
    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    print("*" * 70)


def safe_process_multimodal_data(template_obj, sample_data, sample_type, **kwargs):
    """
    Safely process multimodal data with proper resource management.
    
    Args:
        template_obj: Template object for processing
        sample_data: The data to process (image/video/audio)
        sample_type: Type of data ('image', 'video', 'audio')
        **kwargs: Additional arguments for processing
    
    Returns:
        Processed multimodal data or None if processing fails
    """
    if sample_data is None:
        return None
        
    try:
        # Clear some memory before processing
        gc.collect()
        
        if sample_type == "image":
            processed = template_obj.mm_plugin._regularize_images(
                sample_data, 
                image_max_pixels=kwargs.get('image_max_pixels', 768*768), 
                image_min_pixels=kwargs.get('image_min_pixels', 32*32)
            )
            return {"image": processed["images"]} if "images" in processed else None
            
        elif sample_type == "video":
            processed = template_obj.mm_plugin._regularize_videos(
                sample_data,
                image_max_pixels=kwargs.get('image_max_pixels', 768*768),
                image_min_pixels=kwargs.get('image_min_pixels', 32*32),
                video_fps=kwargs.get('video_fps', 2.0),
                video_maxlen=kwargs.get('video_maxlen', 128),
            )
            return {"video": processed["videos"]} if "videos" in processed else None
            
        elif sample_type == "audio":
            processed = template_obj.mm_plugin._regularize_audios(
                sample_data,
                sampling_rate=kwargs.get('sampling_rate', 16000),
            )
            if "audios" in processed and "sampling_rates" in processed:
                # Return list instead of zip object to avoid potential resource issues
                return {"audio": list(zip(processed["audios"], processed["sampling_rates"]))}
            return None
            
    except Exception as e:
        print(f"Warning: Failed to process {sample_type} data: {e}")
        # Force garbage collection after failure
        gc.collect()
        return None
    
    finally:
        # Ensure we clean up any temporary resources
        if 'processed' in locals():
            del processed
        gc.collect()


def monitor_resources():
    """Monitor system resources and print current status."""
    try:
        import psutil
        process = psutil.Process()
        print(f"Open files: {len(process.open_files())}")
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        print(f"CPU usage: {process.cpu_percent()}%")
    except ImportError:
        try:
            # Fallback to basic resource monitoring if psutil is not available
            with open('/proc/self/status') as f:
                for line in f:
                    if 'VmRSS:' in line:
                        memory_usage = float(line.split()[1]) / 1024  # Convert KB to MB
                        print(f"Memory usage: {memory_usage:.2f} MB")
                        break
            
            open_files = os.popen('lsof -p %d | wc -l' % os.getpid()).read().strip()
            print(f"Open files: {open_files}")
        except Exception as e:
            print(f"Could not monitor resources: {e}")


if __name__ == "__main__":
    fire.Fire(vllm_infer)
