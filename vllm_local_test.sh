python scripts/vllm_infer.py \
    --model_name_or_path /home/june/Code/new_llamafactory/saves/huggingface_origin/InternVL3-8B-hf\
    --dataset coco_test\
    --template intern_vl\
    --save_name "evaluate_outputs/zero_shot_intervlm/generated_predictions_llamafactory_vlm.jsonl"\
    --max_new_tokens 1024 \
    --temperature 0.0