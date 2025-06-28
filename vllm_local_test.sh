
# python scripts/vllm_infer.py \
#     --model_name_or_path Qwen/Qwen2-VL-7B-Instruct\
#     --dataset coco_test\
#     --template qwen2_vl\
#     --save_name "evaluate_outputs/test/generated_predictions_llamafactory_vlm.jsonl"\
#     --max_new_tokens 1024 \
#      --use_cache true \
#     --temperature 0.0


python scripts/vllm_infer.py \
    --model_name_or_path /home/june/Code/new_llamafactory/saves/huggingface_origin/InternVL3-8B\
    --dataset coco_test\
    ---trust-remote-code true \
    --template intern_vl\
    --save_name "evaluate_outputs/zero_shot_intervlm/generated_predictions_llamafactory_vlm.jsonl"\
    --max_new_tokens 1024 \
     --use_cache true \
    --temperature 0.0

# python scripts/my_vllm_infer.py \
#     --json-file  data/llamainput/mcml/coco_test_format.mcml.json \
#     --save-name evaluate_outputs/test/generated_predictions.jsonl

# python scripts/my_vllm_infer.py \
#     --json-file  data/llamainput/local/coco_test_format.mcml_local.json \
#     --save-name evaluate_outputs/test/generated_predictions.jsonl