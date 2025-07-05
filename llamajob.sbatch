#!/bin/bash
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --job-name=qwen2_sft_def

if ! enroot list | grep -q "llamafactory"; then
    echo "Container llamafactory not found. Creating container..."
    enroot create -n llamafactory /dss/dsshome1/01/go68zuc2/llamafactory.sqsh  
    if [ $? -ne 0 ]; then
        echo "Failed to create llamafactory container"
        exit 1s
    fi
    echo "Container llamafactory created successfully."
else
    echo "Container llamafactory already exists."
fi

# 启动容器
enroot start --root --mount /dss/dssmcmlfs01/pn39hu/pn39hu-dss-0000/june:/home/june/ llamafactory << EOF

source ~/.bashrc

conda activate llamafactory

nvidia-smi

cd /home/june/Code/LLaMA-Factory-Latest

bash experiment.sh

EOF
