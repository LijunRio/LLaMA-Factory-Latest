import json
import re
import supervision as sv
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
from rodeo import RoDeO
import os
import pandas as pd
import supervision as sv
import wandb
import re

# CLASSES = ['Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Cardiomegaly', 'Nodule or Mass', 'Lung Opacity', 'Other lesion', 'Pleural effusion', 'Interstitial lung disease(ILD)', 'Infiltration', 'Calcification', 'Consolidation', 'Atelectasis', 'Rib fracture', 'Mediastinal shift', 'Enlarged PA', 'Pneumothorax', 'Emphysema', 'Lung cavity', 'Lung cyst', 'Clavicle fracture', 'Edema', 'None', 'Not in the list']
# # CLASSES_LOWER = [cls.lower() for cls in CLASSES]


CLASSES =['person', 'chair', 'car', 'dining table', 'bottle', 'cup', 'bowl', 'handbag', 'truck', 'backpack', 'cell phone', 'bench', 'sports ball', 'knife', 'couch', 'sink', 'book', 'tie', 'potted plant', 'vase', 'umbrella', 'traffic light', 'bed', 'clock', 'bus', 'tv', 'dog', 'motorcycle', 'bird', 'horse', 'spoon', 'laptop', 'bicycle', 'cake', 'surfboard', 'tennis racket', 'baseball bat', 'remote', 'boat', 'skateboard', 'pizza', 'cow', 'baseball glove', 'giraffe', 'oven', 'keyboard', 'skis', 'banana', 'carrot', 'kite', 'wine glass', 'elephant', 'stop sign', 'refrigerator', 'broccoli', 'microwave', 'teddy bear', 'sheep', 'donut', 'zebra', 'orange', 'apple', 'fire hydrant', 'toothbrush', 'scissors', 'None', 'Not in the list']



def read_jsonl(output_file):
    data = []
    with open(output_file, 'r') as f:
        for line in f:
            data.append(json.loads(line)) 
    return data

# def extract_all_findings_and_bboxes(text):
#     results = []
#     # 每个对象提取为一个块：找到 { ... } 的结构
#     matches = re.finditer(
#         r'\{\s*"finding"\s*:\s*"([^"]+)"\s*,\s*"bounding_box"\s*:\s*\[([^\]]+)\]\s*\}', 
#         text
#     )

#     for match in matches:
#         finding = match.group(1)
#         bbox_str = match.group(2)
#         try:
#             bbox = [int(x.strip()) for x in bbox_str.split(',')]
#         except:
#             bbox = None  # 如果有格式错误就跳过/标记
#         results.append({
#             "finding": finding,
#             "bounding_box": bbox
#         })

#     return results if results else None

import re

def extract_all_findings_and_bboxes(text):
    # 清理输入文本，去除 <answer> 标签
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    
    # 匹配 finding 和 bounding_box 使用正则表达式
    pattern = r'"finding"\s*:\s*"([^"]+)"\s*,\s*"bounding_box"\s*:\s*\[\s*([^\]]+)\s*\]'
    
    # 提取所有匹配项
    matches = re.findall(pattern, cleaned_text)
    
    results = []
    for match in matches:
        finding = match[0]
        # 将 bounding_box 的字符串分割成数字列表
        try:
            bounding_box = [int(x.strip()) for x in match[1].split(',')]
        except ValueError:
            bounding_box = None  # 如果遇到无效格式则跳过
        results.append({
            "finding": finding,
            "bounding_box": bounding_box
        })
    
    return results if results else None


def convert_bboxes2pixel(data, width, height):
    # 验证 width 和 height 是否有效
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        print("Invalid width or height values")
        return None

    converted = []
    try:
        for item in data:
            finding = item.get('finding', '')
            
            # 检查 bounding_box 是否有效
            if isinstance(item.get('bounding_box'), (list, tuple)) and len(item['bounding_box']) == 4:
                xmin, ymin, xmax, ymax = item['bounding_box']
            else:
                print(f"Skipping item with invalid bounding box: {item}")
                continue  # 如果 bounding_box 无效，则跳过该项
            
            # 计算新的 bounding box
            new_bbox = [
                int(xmin / 1000 * width),   # xmin
                int(ymin / 1000 * height),  # ymin
                int(xmax / 1000 * width),   # xmax
                int(ymax / 1000 * height)   # ymax
            ]
            
            # 将处理结果添加到 converted 列表中
            converted.append({'finding': finding, 'bounding_box': new_bbox})
        
        return converted  # 把返回值放在循环外部
    except Exception as e:
        print(f"Error: {e}")
        return None



def convert_to_svformat(data):
    pred_boxes = []
    pred_labels = []
    
    for item in data:
        pred_boxes.append(item["bounding_box"])  
        pred_labels.append(item["finding"])     
    formatted_output = {
        "<OD>": {
            "bboxes": pred_boxes,
            "labels": pred_labels
        }
    }
    return formatted_output



from difflib import get_close_matches

# 宽松匹配函数
def correct_class_names(predicted_names, valid_classes, threshold=0.8):
    corrected_names = []
    for name in predicted_names:
        matches = get_close_matches(name, valid_classes, n=1, cutoff=threshold)
        corrected_names.append(matches[0] if matches else name)  # 若无匹配，保留原值
    return corrected_names



def convert_for_rodeo(detections):
    converted = []
    
    try:
        for d in detections:
            if 'finding' not in d or 'bounding_box' not in d:
                continue  # 跳过缺少必要字段的数据
            
            if not isinstance(d['bounding_box'], (list, tuple)) or len(d['bounding_box']) != 4:
                continue  # 跳过 bounding_box 长度不够的数据
            
            if d['finding'] not in CLASSES:
                continue  # 跳过不在 CLASSES 里的类别
            
            matches = get_close_matches(d['finding'], CLASSES, n=1, cutoff=0.5)
            if not len(matches)==0:
                class_id = CLASSES.index(matches[0])  
            else:
                class_id = CLASSES.index('Not in the list')
            bbox = d['bounding_box']
            converted.append([bbox[0], bbox[1], bbox[2], bbox[3], class_id])

        if not converted:
            converted.append([0, 0, 0, 0, CLASSES.index('None')])  # 返回默认值

    except Exception as e:
        print(f"Error in convert_for_rodeo: {e}")
        converted = [[0, 0, 0, 0, CLASSES.index('None')]]  # 发生异常也返回默认值

    return np.array(converted)







def process_prediction_file(file_path, original_path, visualize_every=20):
    data = read_jsonl(file_path)
    original_data = json.load(open(original_path, 'r'))

    bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS)
    color_annotator = sv.ColorAnnotator(color_lookup=sv.ColorLookup.CLASS)

    skipped_samples = []
    rodeo = RoDeO(class_names=CLASSES)
    all_predictions = []
    all_gt = []
    visual_image_list = []
    captions = []
    visilize_image = 0
    for i, record in tqdm(enumerate(data), total=len(data), desc=f"Processing {os.path.basename(file_path)}"):
        # import pdb; pdb.set_trace()
        prediction = extract_all_findings_and_bboxes(record['predict'])
        ground_truth = extract_all_findings_and_bboxes(record['label'])

        # 获取图像路径并读取图像尺寸
        image_path = original_data[i]['images'][0]
        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        resolution = (width, height)

        pixel_prediction = convert_bboxes2pixel(prediction, width, height)
        pixel_ground_truth = convert_bboxes2pixel(ground_truth, width, height)

        if prediction is None or ground_truth is None or len(pixel_prediction) == 0 or len(pixel_ground_truth) == 0:
            print(f"Skipped index {i}: No findings or bounding boxes.")
            skipped_samples.append(i)
            continue

        # RoDeO format conversion
        rodeo_pred = convert_for_rodeo(pixel_prediction)
        rodeo_gt = convert_for_rodeo(pixel_ground_truth)
        rodeo.add([rodeo_pred], [rodeo_gt])

        # Supervision format
        sv_pred = convert_to_svformat(pixel_prediction)
        sv_gt = convert_to_svformat(pixel_ground_truth)

        pred = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, sv_pred, resolution_wh=resolution)
        gt = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, sv_gt, resolution_wh=resolution)

        # Class ID mapping for predictions
        pred_class_ids = []
        for class_name in pred['class_name']:
            matches = get_close_matches(class_name, CLASSES, n=1, cutoff=0.5)
            class_id = CLASSES.index(matches[0]) if matches else CLASSES.index('Not in the list')
            pred_class_ids.append(class_id)
        pred.class_id = np.array(pred_class_ids)
        pred.confidence = np.ones(len(pred))

        # Class ID mapping for ground truths
        gt_class_ids = [CLASSES.index(class_name) for class_name in gt['class_name']]
        gt.class_id = np.array(gt_class_ids)

        all_predictions.append(pred)
        all_gt.append(gt)

        # if i % visualize_every == 0:
        if visilize_image <=20:
            visilize_image += 1
            image_with_gt = bounding_box_annotator.annotate(image.copy(), gt)
            image_with_gt = label_annotator.annotate(image_with_gt, gt)
            image_with_pred = bounding_box_annotator.annotate(image_with_gt.copy(), pred)
            image_with_pred = color_annotator.annotate(image_with_pred, pred)
            visual_image_list.append(Image.fromarray(image_with_pred.astype(np.uint8)))
            captions.append(f"Image_path:{os.path.basename(image_path)}\nGT: {pixel_ground_truth}\nPred: {pixel_prediction}")

    # Scoring
    score = rodeo.compute()
    print("RoDeO Scores:")
    for key, val in score.items():
        print(f'{key}: {val}')

    # Confusion Matrix
    confusion_matrix = sv.ConfusionMatrix.from_detections(predictions=all_predictions, targets=all_gt, classes=CLASSES)

    # mAP
    mean_ap = sv.MeanAveragePrecision.from_detections(predictions=all_predictions, targets=all_gt)
    print("mAP_50_95:", mean_ap.map50_95)
    print("mAP_50:", mean_ap.map50)
    print("mAP_75:", mean_ap.map75)

   # 🔽🔽🔽 写入 failure.jsonl 的逻辑（新加部分） 🔽🔽🔽
    if skipped_samples:
        skipped_records = [data[i] for i in skipped_samples]
        failure_path = os.path.join(os.path.dirname(file_path), 'failure.jsonl')
        with open(failure_path, 'w') as f:
            for record in skipped_records:
                json.dump(record, f)
                f.write('\n')
        print(f"Saved {len(skipped_samples)} skipped samples to {failure_path}")
    # 🔼🔼🔼 END 🔼🔼🔼

    return {
        "rodeo_score": score,
        "confusion_matrix": confusion_matrix,
        "mean_ap": mean_ap,
        "visuals": visual_image_list,
        "captions": captions,
        "skipped_samples": skipped_samples
    }


# 
if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "disabled"
    # folder = '/u/home/lj0/Code/New_LLaMA-Factory/evaluate_outputs/zero_shot_coco/Qwen_Qwen2.5-VL-3B-Instruct'  # <-- 设置你的 zero-shot 输出路径
    # folder = '/u/home/lj0/Code/New_LLaMA-Factory/evaluate_outputs/zero_shot_coco_2/'
    folder = '/u/home/lj0/Code/LLaMA-Factory-Latest/evaluate_outputs/InternVL3-8B-hf/lora/eval_2025-06-28-13-50-22'
    model_name = os.path.basename(folder)

    wandb.init(
        project="Qwen Experiment",
        entity="compai",
        name='zero-shot-coco-7b' + model_name
    )

    file_path = os.path.join(folder, 'generated_predictions.jsonl')
    # file_path = os.path.join(folder, 'generated_predictions_llamafactory_vlm.jsonl')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    results = process_prediction_file(
        file_path=file_path,
        original_path='/u/home/lj0/Code/LLaMA-Factory-Latest/data/0_my_dataset/coco/coco_test_format.mcml.json'
    )

    batch_data = [
        wandb.Image(img, caption=cap) 
        for img, cap in zip(results['visuals'], results['captions'])
    ]

    wandb.log({
        "step_id": "zero-shot",
        "RoDeO/localization": results['rodeo_score']['RoDeO/localization'],
        "RoDeO/shape_matching": results['rodeo_score']['RoDeO/shape_matching'],
        "RoDeO/classification": results['rodeo_score']['RoDeO/classification'], 
        "RoDeO/total": results['rodeo_score']['RoDeO/total'],
        "mAP_50_95": results['mean_ap'].map50_95,
        "mAP_50": results['mean_ap'].map50,
        "mAP_75": results['mean_ap'].map75,
        "Confusion Matrix": wandb.Image(results['confusion_matrix'].plot()),
        "skipped_samples": len(results['skipped_samples']),
        "Visual Results": batch_data
    })


    wandb.finish()

