llamafactory-cli train \
    --stage sft \
    --model_name_or_path /home/june/Code/new_llamafactory/saves/huggingface_origin/InternVL3-8B-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template intern_vl \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset vinder_adkg_test \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --report_to none \
    --max_new_tokens 1024 \
    --top_p 0.7 \
    --temperature 0.2 \
    --output_dir evaluate_outputs/InternVL3-8B-hf/vindr_test_set \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --do_predict True 



llamafactory-cli train \
    --stage sft \
    --model_name_or_path /home/june/Code/new_llamafactory/saves/huggingface_origin/Qwen2-VL-7B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset vinder_adkg_test \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --report_to none \
    --max_new_tokens 1024 \
    --top_p 0.7 \
    --temperature 0.2 \
    --output_dir evaluate_outputs/Qwen2-VL-7B-Instruct/vindr_test_set \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --do_predict True 


llamafactory-cli train \
    --stage sft \
    --model_name_or_path /home/june/Code/new_llamafactory/saves/huggingface_origin/InternVL3-2B-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template intern_vl \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset vinder_adkg_test \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate True \
    --report_to none \
    --max_new_tokens 1024 \
    --top_p 0.7 \
    --temperature 0.2 \
    --output_dir evaluate_outputs/Qwen2-VL-2B-Instruct/InternVL3-2B-hf \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --do_predict True 


llamafactory-cli train \
    --stage sft \
    --model_name_or_path /home/june/Code/new_llamafactory/saves/huggingface_origin/Qwen2-VL-2B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset vinder_adkg_test \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate True \
    --report_to none \
    --max_new_tokens 1024 \
    --top_p 0.7 \
    --temperature 0.2 \
    --output_dir evaluate_outputs/Qwen2-VL-2B-Instruct/vindr_test_set \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --do_predict True 