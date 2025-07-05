#!/bin/bash
### 一定要使用VLLM_USE_V1=0, 用vllm v0版本https://github.com/vllm-project/vllm/pull/16618
## --enforce-eager 或设置 VLLM_USE_V1=0 则恢复正常输出。 vLLM v1 中对某些特殊结构（如 GLM4 的 residual 处理、KV-cache 策略、多模态输入）没有正确兼容，导致输出逻辑异常。

declare -a FILES=("vindr_sft_base")  # 修改为实际的目录名 vindr_sft_def,vindr_sft_base
EVAL_DATASET="vinder_adkg_test" # vinder_adkg_def_test,vinder_adkg_test
declare -a MODELS=("qwen2_vl-3b")  # 修改为实际的模型名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for model_name in "${MODELS[@]}"; do
    # 设置路径
    CHECKPOINT_DIR="saves/$model_name/${FILES[0]}"
    OUTPUT_DIR="./evaluate_outputs/new_results/${FILES[0]}_${model_name}"
    echo "Checkpoint directory: $CHECKPOINT_DIR"
    echo "Output directory: $OUTPUT_DIR"

    echo "Processing model: $model_name"
    mkdir -p "$OUTPUT_DIR"
    

    # 遍历每个 checkpoint
    for ckpt in $CHECKPOINT_DIR/checkpoint-*; do
        if [ -d "$ckpt" ]; then
            ckpt_name=$(basename "$ckpt")
            echo "Running inference on checkpoint: $ckpt_name"

            save_path="$OUTPUT_DIR/$ckpt_name"
            mkdir -p "$save_path"

            # 执行 Python 脚本进行推理 (使用贪婪搜索进行测试)
             env VLLM_USE_V1=0 python scripts/vllm_infer.py \
                --model_name_or_path "$ckpt" \
                --dataset $EVAL_DATASET \
                --template qwen2_vl \
                --save_name "$save_path/generated_predictions.jsonl" \
                --max_new_tokens 4096 \
                --temperature 0.0 \
                --top_p 1.0 \
                --top_k -1 \
                --repetition_penalty 1.0 \
                --batch_size 32 \
                --seed 42
        fi
    done
done
