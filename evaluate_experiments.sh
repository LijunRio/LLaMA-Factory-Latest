#!/bin/bash

# Common parameters configuration
DATASET_NAME="padchest_gr_test"  # Dataset name variable

COMMON_ARGS=(
    --stage sft
    --preprocessing_num_workers 16
    --finetuning_type lora
    --quantization_method bnb
    --flash_attn auto
    --dataset_dir data
    --eval_dataset "$DATASET_NAME"
    --cutoff_len 1024
    --max_samples 100000
    --predict_with_generate True
    --report_to none
    --max_new_tokens 1024
    --top_p 0.7
    --temperature 0.2
    --trust_remote_code True
    --ddp_timeout 180000000
    --do_predict True
    --default_system "You are an expert radiologist committed to accurate and precise medical image analysis. Your role is to identify and localize radiological evidence of specific diseases in chest X-ray images. You will be provided with a chest X-ray image and a disease name. Your task is to accurately localize the relevant region(s) in the image."
)

# Model configurations
declare -A MODEL_PATHS=(
    ["InternVL3-8B"]="/home/june/Code/new_llamafactory/saves/huggingface_origin/InternVL3-8B-hf"
    ["Qwen2-VL-7B"]="/home/june/Code/new_llamafactory/saves/huggingface_origin/Qwen2-VL-7B-Instruct"
    ["InternVL3-2B"]="/home/june/Code/new_llamafactory/saves/huggingface_origin/InternVL3-2B-hf"
    ["Qwen2-VL-2B"]="/home/june/Code/new_llamafactory/saves/huggingface_origin/Qwen2-VL-2B-Instruct"
)

declare -A MODEL_TEMPLATES=(
    ["InternVL3-8B"]="intern_vl"
    ["Qwen2-VL-7B"]="qwen2_vl"
    ["InternVL3-2B"]="intern_vl"
    ["Qwen2-VL-2B"]="qwen2_vl"
)

declare -A MODEL_BATCH_SIZES=(
    ["InternVL3-8B"]=8
    ["Qwen2-VL-7B"]=8
    ["InternVL3-2B"]=16
    ["Qwen2-VL-2B"]=16
)

# List of models to evaluate
MODELS_TO_EVALUATE=("InternVL3-8B" "Qwen2-VL-7B" "InternVL3-2B" "Qwen2-VL-2B")

# Loop through each model for evaluation
for model_name in "${MODELS_TO_EVALUATE[@]}"; do
    model_path="${MODEL_PATHS[$model_name]}"
    template="${MODEL_TEMPLATES[$model_name]}"
    batch_size="${MODEL_BATCH_SIZES[$model_name]}"
    
    echo "=============================================="
    echo "Starting evaluation for model: $model_name"
    echo "Model path: $model_path"
    echo "Template: $template"
    echo "Batch size: $batch_size"
    echo "=============================================="
    
    # Execute training command
    llamafactory-cli train \
        "${COMMON_ARGS[@]}" \
        --model_name_or_path "$model_path" \
        --template "$template" \
        --per_device_eval_batch_size "$batch_size" \
        --output_dir "evaluate_outputs/results/$DATASET_NAME/$model_name"
    
    # Check execution result
    if [ $? -eq 0 ]; then
        echo "✅ $model_name evaluation completed successfully"
    else
        echo "❌ $model_name evaluation failed"
    fi
    
    echo ""
done

echo "🎉 All model evaluations completed!" 