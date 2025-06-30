#!/bin/bash

# 原始配置文件
CONFIG_FILE="examples/train_full/qwen2_5vl_full_sft.yaml"
# 用于保存修改后的副本
MODIFIED_CONFIG_FILE="examples/train_full/qwen2_vl_sft_modified_def.yaml"

# 运行实验的通用函数
run_experiment () {
  echo "🚀 Running experiment: $1"
  echo "-----------------------------------"
  echo "Model: $MODEL_NAME"
  echo "Output Dir: $OUTPUT_DIR"
  echo "Train Dataset: $TRAIN_DATASET"
  # echo "Eval Dataset: $EVAL_DATASET"
  echo "Cutoff Length: $CUTOFF_LENGTH"
  echo "Save Steps: $SAVE_STEPS"
  # echo "Eval Steps: $EVAL_STEPS"
  echo "Learning Rate: $LEARNING_RATE"
  echo "Num Train Epochs: $NUM_TRAIN_EPOCHS"
  echo "Logging Steps: $LOGGING_STEPS"
  echo "Per Device Train Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
  echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
  # echo "Per Device Eval Batch Size: $PER_DEVICE_EVAL_BATCH_SIZE"
  echo "-----------------------------------"

  # 复制原始文件到新文件
  cp $CONFIG_FILE $MODIFIED_CONFIG_FILE

  # 使用 sed 替换 yaml 文件中的相关参数
  # 确保替换的值是数字而不是字符串
  sed -i "s|model_name_or_path: .*|model_name_or_path: $MODEL_NAME|" $MODIFIED_CONFIG_FILE
  sed -i "s|output_dir: .*|output_dir: $OUTPUT_DIR|" $MODIFIED_CONFIG_FILE
  sed -i "s|dataset: .*|dataset: $TRAIN_DATASET|" $MODIFIED_CONFIG_FILE
  # sed -i "s|eval_dataset: .*|eval_dataset: $EVAL_DATASET|" $MODIFIED_CONFIG_FILE
  sed -i "s|cutoff_len: .*|cutoff_len: $CUTOFF_LENGTH|" $MODIFIED_CONFIG_FILE
  sed -i "s|save_steps: .*|save_steps: $SAVE_STEPS|" $MODIFIED_CONFIG_FILE
  # sed -i "s|eval_steps: .*|eval_steps: $EVAL_STEPS|" $MODIFIED_CONFIG_FILE
  sed -i "s|learning_rate: .*|learning_rate: $LEARNING_RATE|" $MODIFIED_CONFIG_FILE
  sed -i "s|num_train_epochs: .*|num_train_epochs: $NUM_TRAIN_EPOCHS|" $MODIFIED_CONFIG_FILE
  sed -i "s|logging_steps: .*|logging_steps: $LOGGING_STEPS|" $MODIFIED_CONFIG_FILE
  sed -i "s|per_device_train_batch_size: .*|per_device_train_batch_size: $PER_DEVICE_TRAIN_BATCH_SIZE|" $MODIFIED_CONFIG_FILE
  sed -i "s|gradient_accumulation_steps: .*|gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS|" $MODIFIED_CONFIG_FILE
  # sed -i "s|per_device_eval_batch_size: .*|per_device_eval_batch_size: $PER_DEVICE_EVAL_BATCH_SIZE|" $MODIFIED_CONFIG_FILE
  
  # 启动训练，使用修改后的副本配置文件
  llamafactory-cli train $MODIFIED_CONFIG_FILE

  # 可选：打印已生成的 config（可用于调试）
  cat $MODIFIED_CONFIG_FILE
}


# ### 🚀 Experiment 1
# MODEL_NAME="/home/june/Code/new_llamafactory/saves/huggingface_origin/Qwen2-VL-2B-Instruct/"
# OUTPUT_DIR="saves/qwen2_vl-3b/vindr_sft_base"
# TRAIN_DATASET="vinder_train_base"
# CUTOFF_LENGTH=1024
# SAVE_STEPS=63
# LEARNING_RATE=3.0e-5
# NUM_TRAIN_EPOCHS=10.0
# PER_DEVICE_TRAIN_BATCH_SIZE=8
# GRADIENT_ACCUMULATION_STEPS=8
# LOGGING_STEPS=5
# run_experiment "Experiment 1:Qwen2-VL-2B-Instruct SFT Base"

# ### 🚀 Experiment 2
MODEL_NAME="/home/june/Code/new_llamafactory/saves/huggingface_origin/Qwen2-VL-2B-Instruct/"
OUTPUT_DIR="saves/qwen2_vl-3b/vindr_sft_def"
TRAIN_DATASET="vinder_train_def"
CUTOFF_LENGTH=1024
SAVE_STEPS=63
LEARNING_RATE=3.0e-5
NUM_TRAIN_EPOCHS=10.0
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
LOGGING_STEPS=5
run_experiment "Experiment 1:Qwen2-VL-2B-Instruct SFT Base"

