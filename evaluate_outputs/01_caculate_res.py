"""
Multi-Model Evaluation Script for Medical Image Analysis
Evaluates multiple models' performance on medical image detection tasks
"""

import json
import re
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from difflib import get_close_matches
import supervision as sv
from rodeo import RoDeO
import wandb

# ==================== CONFIGURATION PARAMETERS ====================
# Paths Configuration
RESULTS_FOLDER = "/u/home/lj0/Code/LLaMA-Factory-Latest/evaluate_outputs/results/vinder_adkg_test"
ORIGINAL_DATA_PATH = "/u/home/lj0/Code/LLaMA-Factory-Latest/data/0_my_dataset/vindr/qwen2_vindr_input_test_len_2108.json"
# RESULTS_FOLDER = "/u/home/lj0/Code/LLaMA-Factory-Latest/evaluate_outputs/results/padchest_gr_test"
# ORIGINAL_DATA_PATH = "/u/home/lj0/Code/LLaMA-Factory-Latest/data/0_my_dataset/padchest_gt/padchest_input_data_qwen2_test_len_1285.json"
PADCHEST_GR=False
if PADCHEST_GR:
    IMAGE_PATH_REPLACEMENT = {
    "from": "/home/june/datasets/padchest_512/images_512/",
    "to": "/u/home/lj0/datasets/2gpu-workstations/padchest_gr/images_512/"
    }
    CLASSES = [
    'pleural thickening', 'atelectasis', 'pleural effusion', 'other entities',
    'cardiomegaly', 'aortic elongation', 'vertebral degenerative changes',
    'aortic atheromatosis', 'nodule', 'alveolar pattern', 'hiatal hernia',
    'scoliosis', 'hemidiaphragm elevation', 'hyperinflated lung',
    'interstitial pattern', 'fracture', 'vascular hilar enlargement',
    'nsg tube', 'endotracheal tube', 'hypoexpansion',
    'central venous catheter', 'electrical device', 'bronchiectasis', 'goiter','Not in the list'
]
else:
    IMAGE_PATH_REPLACEMENT = {
        "from": "/home/june/datasets/",
        "to": "/u/home/lj0/datasets/2gpu-workstations/"
    }
    # Classes Definition
    CLASSES = [
        'Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Cardiomegaly', 
        'Nodule/Mass', 'Lung Opacity', 'Other lesion', 'Pleural effusion', 'ILD', 
        'Infiltration', 'Calcification', 'Consolidation', 'Atelectasis', 'Rib fracture', 
        'Mediastinal shift', 'Enlarged PA', 'Pneumothorax', 'Emphysema', 'Lung cavity', 
        'Lung cyst', 'Clavicle fracture', 'Edema', 'None', 'Not in the list'
    ]

# Model Evaluation Configuration
PREDICTION_FILE_NAME = "generated_predictions.jsonl"
DEFAULT_BBOX = [0, 0, 0, 0]  # Default bbox for no_output cases
VISUALIZE_LIMIT = 20  # Number of images to visualize per model

# Output Configuration
OUTPUT_CSV_NAME = "evaluation_summary.csv"
OUTPUT_JSON_NAME = "evaluation_results.json"
FAILURE_FILE_NAME = "failure.jsonl"



# WandB Configuration
WANDB_PROJECT = "Multi-Model-Medical-Detection"
WANDB_ENTITY = "compai"
WANDB_DISABLED = False  # Set to False to enable wandb logging

# Evaluation Parameters
CLASS_MATCH_THRESHOLD = 0.5  # Threshold for fuzzy class name matching
BBOX_COORDINATE_SCALE = 1000  # Scale factor for bbox coordinates

# ==================== END CONFIGURATION ====================



def read_jsonl(file_path):
    """Read JSONL file and return list of JSON objects"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data


def extract_all_findings_and_bboxes(text):
    """Extract findings and bounding boxes from prediction text"""
    if not text or text.strip() == "":
        return [{"label": "None", "bbox_2d": DEFAULT_BBOX}]
    
    results = []
    
    # Method 1: Try parsing entire text as JSON array
    try:
        cleaned_text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
        cleaned_text = re.sub(r'```json\s*', '', cleaned_text)  # Remove json code block start
        cleaned_text = re.sub(r'```\s*$', '', cleaned_text)  # Remove code block end
        cleaned_text = cleaned_text.strip()
        
        json_data = json.loads(cleaned_text)
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict) and 'label' in item and 'bbox_2d' in item:
                    results.append({
                        "label": item['label'],
                        "bbox_2d": item['bbox_2d']
                    })
        if results:
            return results
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Method 2: Regex pattern for {"bbox_2d": [...], "label": "..."}
    object_pattern = r'\{\s*"bbox_2d"\s*:\s*\[\s*([^\]]+)\s*\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(object_pattern, text)
    for match in matches:
        bbox_str, label = match
        try:
            bbox_2d = [int(x.strip()) for x in bbox_str.split(',')]
            if len(bbox_2d) == 4:
                results.append({"label": label, "bbox_2d": bbox_2d})
        except ValueError:
            continue
    
    if results:
        return results
    
    # Method 3: Regex pattern for "label": "...", "bbox_2d": [...]
    pattern = r'"label"\s*:\s*"([^"]+)"\s*,\s*"bbox_2d"\s*:\s*\[\s*([^\]]+)\s*\]'
    matches = re.findall(pattern, text)
    for match in matches:
        label, bbox_str = match
        try:
            bbox_2d = [int(x.strip()) for x in bbox_str.split(',')]
            if len(bbox_2d) == 4:
                results.append({"label": label, "bbox_2d": bbox_2d})
        except ValueError:
            continue
    
    # If no valid extractions found, return default no_output
    return results if results else [{"label": "None", "bbox_2d": DEFAULT_BBOX}]


def convert_bboxes2pixel(data, width, height):
    """Convert normalized bounding boxes to pixel coordinates"""
    if not data or width <= 0 or height <= 0:
        return [{"label": "None", "bbox_2d": DEFAULT_BBOX}]
        
    converted = []
    try:
        for item in data:
            if not isinstance(item, dict) or 'label' not in item or 'bbox_2d' not in item:
                continue
                
            label = item['label']
            bbox = item['bbox_2d']
            
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
                
            xmin, ymin, xmax, ymax = bbox
            
            # Convert normalized coordinates to pixel coordinates
            new_bbox = [
                max(0, min(int(xmin / BBOX_COORDINATE_SCALE * width), width)),
                max(0, min(int(ymin / BBOX_COORDINATE_SCALE * height), height)),
                max(0, min(int(xmax / BBOX_COORDINATE_SCALE * width), width)),
                max(0, min(int(ymax / BBOX_COORDINATE_SCALE * height), height))
            ]
            
            # Ensure valid bbox (xmax > xmin, ymax > ymin)
            if new_bbox[2] > new_bbox[0] and new_bbox[3] > new_bbox[1]:
                converted.append({'label': label, 'bbox_2d': new_bbox})
                
    except Exception as e:
        print(f"Error in bbox conversion: {e}")
        
    return converted if converted else [{"label": "None", "bbox_2d": DEFAULT_BBOX}]


def correct_class_names(predicted_names, valid_classes, threshold=CLASS_MATCH_THRESHOLD):
    """Correct class names using fuzzy matching"""
    corrected_names = []
    for name in predicted_names:
        matches = get_close_matches(name, valid_classes, n=1, cutoff=threshold)
        corrected_names.append(matches[0] if matches else "Not in the list")
    return corrected_names


def convert_for_rodeo(detections):
    """Convert detections to RoDeO format"""
    converted = []
    
    try:
        for d in detections:
            if not isinstance(d, dict) or 'label' not in d or 'bbox_2d' not in d:
                continue
                
            if not isinstance(d['bbox_2d'], (list, tuple)) or len(d['bbox_2d']) != 4:
                continue
            
            # Skip default/invalid bboxes
            bbox = d['bbox_2d']
            if bbox == DEFAULT_BBOX or bbox == [0, 0, 0, 0]:
                continue
                
            # Skip if bbox is invalid (no area)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            
            # Skip "None" labels
            if d['label'] in ['None', 'none', '']:
                continue
            
            # Use fuzzy matching for class names
            matches = get_close_matches(d['label'], CLASSES, n=1, cutoff=CLASS_MATCH_THRESHOLD)
            if matches:
                class_id = CLASSES.index(matches[0])
            else:
                class_id = CLASSES.index('Not in the list')
                
            converted.append([bbox[0], bbox[1], bbox[2], bbox[3], class_id])

        # Return empty if no valid detections (let RoDeO handle empty predictions)
        
    except Exception as e:
        print(f"Error in convert_for_rodeo: {e}")

    # Return empty array if no valid detections  
    return np.array(converted) if converted else np.empty((0, 5))


def convert_to_svformat(data):
    """Convert data to supervision format - only include valid detections"""
    pred_boxes = []
    pred_labels = []
    
    for item in data:
        if isinstance(item, dict) and 'bbox_2d' in item and 'label' in item:
            bbox = item["bbox_2d"]
            label = item["label"]
            
            # Skip default/invalid bboxes
            if bbox == DEFAULT_BBOX or bbox == [0, 0, 0, 0]:
                continue
                
            # Skip if bbox is invalid (no area)
            if len(bbox) == 4 and (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                continue
                
            # Skip "None" labels
            if label in ['None', 'none', '']:
                continue
                
            pred_boxes.append(bbox)  
            pred_labels.append(label)
    
    # If no valid detections, return empty format (supervision handles this)
    return {
        "<OD>": {
            "bboxes": pred_boxes,
            "labels": pred_labels
        }
    }







def process_single_model(model_folder, original_data_path):
    """Process a single model's predictions and calculate metrics"""
    model_name = os.path.basename(model_folder)
    prediction_file = os.path.join(model_folder, PREDICTION_FILE_NAME)
    
    if not os.path.exists(prediction_file):
        print(f"Prediction file not found for {model_name}: {prediction_file}")
        return None
    
    print(f"Processing model: {model_name}")
    
    # Load data
    try:
        prediction_data = read_jsonl(prediction_file)
        with open(original_data_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"Error loading data for {model_name}: {e}")
        return None
    
    if len(prediction_data) != len(original_data):
        print(f"Warning: Data length mismatch for {model_name}. Pred: {len(prediction_data)}, Orig: {len(original_data)}")
    
    # Initialize metrics
    rodeo = RoDeO(class_names=CLASSES)
    all_predictions = []
    all_gt = []
    skipped_samples = []
    processed_samples = 0
    visual_images = []
    captions = []
    valid_pred_count = 0  # Count samples with valid predictions
    samples_with_bbox = []  # Store samples that have valid bboxes for visualization
    
    # Debug counters for RoDeO vs mAP analysis
    rodeo_pred_total = 0
    rodeo_gt_total = 0
    sv_pred_total = 0
    sv_gt_total = 0
    
    # Process each sample
    for i, record in tqdm(enumerate(prediction_data), total=len(prediction_data), desc=f"Processing {model_name}"):
        try:
            # Extract predictions and ground truth
            prediction = extract_all_findings_and_bboxes(record.get('predict', ''))
            ground_truth = extract_all_findings_and_bboxes(record.get('label', ''))
            
            # Check if prediction has valid bbox (not default [0,0,0,0])
            has_valid_pred = any(
                pred.get('bbox_2d', DEFAULT_BBOX) != DEFAULT_BBOX and 
                pred.get('bbox_2d', DEFAULT_BBOX) != [0, 0, 0, 0] and
                pred.get('label', 'None') not in ['None', 'none', ''] and
                len(pred.get('bbox_2d', [])) == 4 and
                pred.get('bbox_2d', [0,0,0,0])[2] > pred.get('bbox_2d', [0,0,0,0])[0] and
                pred.get('bbox_2d', [0,0,0,0])[3] > pred.get('bbox_2d', [0,0,0,0])[1]
                for pred in prediction
            )
            
            if has_valid_pred:
                valid_pred_count += 1
            
            # Get image information
            if i < len(original_data):
                image_path = original_data[i]['images'][0]
                # Replace path if needed
                if IMAGE_PATH_REPLACEMENT["from"] in image_path:
                    image_path = image_path.replace(IMAGE_PATH_REPLACEMENT["from"], IMAGE_PATH_REPLACEMENT["to"])
                
                # Get image dimensions
                try:
                    image = Image.open(image_path).convert("RGB")
                    width, height = image.size
                    image_array = np.array(image)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    skipped_samples.append(i)
                    continue
            else:
                print(f"No original data for index {i}")
                skipped_samples.append(i)
                continue
            
            # Convert to pixel coordinates
            pixel_prediction = convert_bboxes2pixel(prediction, width, height)
            pixel_ground_truth = convert_bboxes2pixel(ground_truth, width, height)
            
            # Convert for RoDeO evaluation
            rodeo_pred = convert_for_rodeo(pixel_prediction)
            rodeo_gt = convert_for_rodeo(pixel_ground_truth)
            
            # Debug info for RoDeO vs mAP discrepancy
            rodeo_pred_count = len(rodeo_pred) if len(rodeo_pred.shape) > 0 and rodeo_pred.shape[0] > 0 else 0
            rodeo_gt_count = len(rodeo_gt) if len(rodeo_gt.shape) > 0 and rodeo_gt.shape[0] > 0 else 0
            rodeo_pred_total += rodeo_pred_count
            rodeo_gt_total += rodeo_gt_count
            
            # Only add to RoDeO if we have valid predictions or ground truth
            if rodeo_pred_count > 0 or rodeo_gt_count > 0:
                rodeo.add([rodeo_pred], [rodeo_gt])
            
            # Convert for supervision evaluation (simplified)
            sv_pred = convert_to_svformat(pixel_prediction)
            sv_gt = convert_to_svformat(pixel_ground_truth)
            
            try:
                resolution = (width, height)
                
                # Handle empty predictions - create empty detection
                if not sv_pred["<OD>"]["bboxes"]:
                    pred = sv.Detections.empty()
                    pred.class_id = np.array([], dtype=int)
                    pred.confidence = np.array([], dtype=float)
                else:
                    pred = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, sv_pred, resolution_wh=resolution)
                    # Set class IDs for predictions
                    pred_class_ids = []
                    for class_name in pred['class_name']:
                        matches = get_close_matches(class_name, CLASSES, n=1, cutoff=CLASS_MATCH_THRESHOLD)
                        class_id = CLASSES.index(matches[0]) if matches else CLASSES.index('Not in the list')
                        pred_class_ids.append(class_id)
                    pred.class_id = np.array(pred_class_ids)
                    pred.confidence = np.ones(len(pred))
                
                # Handle ground truth - should always have data
                if not sv_gt["<OD>"]["bboxes"]:
                    gt = sv.Detections.empty()
                    gt.class_id = np.array([], dtype=int)
                else:
                    gt = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, sv_gt, resolution_wh=resolution)
                    # Set class IDs for ground truth
                    gt_class_ids = []
                    for class_name in gt['class_name']:
                        matches = get_close_matches(class_name, CLASSES, n=1, cutoff=CLASS_MATCH_THRESHOLD)
                        class_id = CLASSES.index(matches[0]) if matches else CLASSES.index('Not in the list')
                        gt_class_ids.append(class_id)
                    gt.class_id = np.array(gt_class_ids)
                
                all_predictions.append(pred)
                all_gt.append(gt)
                processed_samples += 1
                
                # Debug info: compare detections between RoDeO and supervision
                sv_pred_count = len(pred) if pred is not None else 0
                sv_gt_count = len(gt) if gt is not None else 0
                sv_pred_total += sv_pred_count
                sv_gt_total += sv_gt_count
                
                # print(f"Sample {i}: RoDeO Pred: {rodeo_pred_count}, RoDeO GT: {rodeo_gt_count}, SV Pred: {sv_pred_count}, SV GT: {sv_gt_count}, Has valid pred: {has_valid_pred}")
                
                # Store sample info for visualization (only if has valid prediction bbox)
                if has_valid_pred and len(samples_with_bbox) < VISUALIZE_LIMIT:
                    samples_with_bbox.append({
                        'image_array': image_array,
                        'image_path': image_path,
                        'pred': pred,
                        'gt': gt,
                        'pixel_prediction': pixel_prediction,
                        'pixel_ground_truth': pixel_ground_truth,
                        'index': i
                    })
                        
            except Exception as e:
                print(f"Error processing supervision format for sample {i}: {e}")
                skipped_samples.append(i)
                continue
                
        except Exception as e:
            print(f"Error processing sample {i} for {model_name}: {e}")
            skipped_samples.append(i)
            continue
    
    # Generate visualizations from stored samples
    try:
        bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS)
        color_annotator = sv.ColorAnnotator(color_lookup=sv.ColorLookup.CLASS)
        
        for sample in samples_with_bbox:
            try:
                image_with_gt = bounding_box_annotator.annotate(sample['image_array'].copy(), sample['gt'])
                image_with_gt = label_annotator.annotate(image_with_gt, sample['gt'])
                image_with_pred = bounding_box_annotator.annotate(image_with_gt.copy(), sample['pred'])
                image_with_pred = color_annotator.annotate(image_with_pred, sample['pred'])
                
                visual_images.append(Image.fromarray(image_with_pred.astype(np.uint8)))
                captions.append(f"Model: {model_name}\nImage: {os.path.basename(sample['image_path'])}\nIndex: {sample['index']}\nGT: {sample['pixel_ground_truth']}\nPred: {sample['pixel_prediction']}")
            except Exception as e:
                print(f"Error creating visualization for sample {sample['index']}: {e}")
    except Exception as e:
        print(f"Error in visualization setup: {e}")
    
    # Calculate metrics
    results = {
        "model_name": model_name,
        "processed_samples": processed_samples,
        "skipped_samples": len(skipped_samples),
        "total_samples": len(prediction_data),
        "valid_pred_count": valid_pred_count,  # New: count of samples with valid predictions
        "valid_pred_rate": valid_pred_count / len(prediction_data) if len(prediction_data) > 0 else 0,
        # Debug info for RoDeO vs mAP analysis
        "rodeo_pred_total": rodeo_pred_total,
        "rodeo_gt_total": rodeo_gt_total,
        "sv_pred_total": sv_pred_total,
        "sv_gt_total": sv_gt_total
    }
    
    try:
        # RoDeO scores
        rodeo_scores = rodeo.compute()
        results["rodeo_scores"] = rodeo_scores
        
        # mAP only (no confusion matrix)
        if all_predictions and all_gt:
            mean_ap = sv.MeanAveragePrecision.from_detections(
                predictions=all_predictions, 
                targets=all_gt
            )
            
            results["mean_ap"] = {
                "map50_95": mean_ap.map50_95,
                "map50": mean_ap.map50,
                "map75": mean_ap.map75
            }
        else:
            print(f"Warning: No valid predictions/ground truth for {model_name}")
            results["mean_ap"] = {"map50_95": 0.0, "map50": 0.0, "map75": 0.0}
            
        results["visuals"] = visual_images
        results["captions"] = captions
        
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        results["rodeo_scores"] = {}
        results["mean_ap"] = {"map50_95": 0.0, "map50": 0.0, "map75": 0.0}
        results["visuals"] = []
        results["captions"] = []
    
    # Save failed samples
    if skipped_samples:
        try:
            skipped_records = [prediction_data[i] for i in skipped_samples if i < len(prediction_data)]
            failure_path = os.path.join(model_folder, FAILURE_FILE_NAME)
            with open(failure_path, 'w', encoding='utf-8') as f:
                for record in skipped_records:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
            print(f"Saved {len(skipped_samples)} failed samples to {failure_path}")
        except Exception as e:
            print(f"Error saving failed samples for {model_name}: {e}")
    
    return results


def get_model_folders(results_folder):
    """Get all model folders from the results directory"""
    if not os.path.exists(results_folder):
        print(f"Results folder not found: {results_folder}")
        return []
    
    model_folders = []
    for item in os.listdir(results_folder):
        item_path = os.path.join(results_folder, item)
        if os.path.isdir(item_path):
            prediction_file = os.path.join(item_path, PREDICTION_FILE_NAME)
            if os.path.exists(prediction_file):
                model_folders.append(item_path)
            else:
                print(f"No prediction file found in {item_path}")
    
    return model_folders


def log_results_to_wandb(all_results):
    """Log results to wandb with proper visualization"""
    if WANDB_DISABLED:
        print("WandB logging is disabled")
        return
    
    if not all_results or all([r is None for r in all_results]):
        print("No valid results to log to WandB")
        return
    
    try:
        # Initialize wandb
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"multi_model_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            reinit=True
        )
        
        print(f"WandB run initialized: {run.url}")
        
        # Prepare summary data
        summary_data = []
        total_images_logged = 0
        
        for result in all_results:
            if result is None:
                continue
                
            model_name = result["model_name"]
            rodeo_scores = result.get("rodeo_scores", {})
            mean_ap = result.get("mean_ap", {})
            
            # Add to summary table
            row_data = {
                "model": model_name,
                "processed_samples": result["processed_samples"],
                "skipped_samples": result["skipped_samples"],
                "valid_pred_count": result.get("valid_pred_count", 0),
                "valid_pred_rate": float(result.get("valid_pred_rate", 0)),
                "success_rate": float(result["processed_samples"] / result["total_samples"]) if result["total_samples"] > 0 else 0.0,
                "rodeo_localization": float(rodeo_scores.get("RoDeO/localization", 0)),
                "rodeo_shape_matching": float(rodeo_scores.get("RoDeO/shape_matching", 0)),
                "rodeo_classification": float(rodeo_scores.get("RoDeO/classification", 0)),
                "rodeo_total": float(rodeo_scores.get("RoDeO/total", 0)),
                "map50_95": float(mean_ap.get("map50_95", 0)),
                "map50": float(mean_ap.get("map50", 0)),
                "map75": float(mean_ap.get("map75", 0))
            }
            summary_data.append(row_data)
            
            # Log individual model metrics
            model_log = {}
            for key, value in row_data.items():
                if key != "model":
                    model_log[f"{model_name}/{key}"] = value
            
            wandb.log(model_log)
            print(f"Logged metrics for {model_name}")
            
            # Log visualizations
            if result.get("visuals") and result.get("captions"):
                print(f"Processing {len(result['visuals'])} visualizations for {model_name}")
                
                wandb_images = []
                for i, (img, caption) in enumerate(zip(result["visuals"], result["captions"])):
                    try:
                        # Ensure image is PIL Image
                        if isinstance(img, np.ndarray):
                            img = Image.fromarray(img.astype(np.uint8))
                        elif not isinstance(img, Image.Image):
                            print(f"Skipping invalid image type: {type(img)}")
                            continue
                        
                        # Create wandb image
                        wandb_img = wandb.Image(img, caption=caption)
                        wandb_images.append(wandb_img)
                        
                        if len(wandb_images) % 5 == 0:
                            print(f"  Processed {len(wandb_images)} images for {model_name}")
                            
                    except Exception as e:
                        print(f"Error processing image {i} for {model_name}: {e}")
                        continue
                
                # Log images in batches to avoid memory issues
                if wandb_images:
                    batch_size = 10
                    for i in range(0, len(wandb_images), batch_size):
                        batch = wandb_images[i:i+batch_size]
                        wandb.log({f"{model_name}_visualizations_batch_{i//batch_size}": batch})
                        total_images_logged += len(batch)
                        print(f"Logged batch {i//batch_size} with {len(batch)} images for {model_name}")
                
                print(f"Successfully logged {len(wandb_images)} images for {model_name}")
        
        # Log summary table
        if summary_data:
            columns = list(summary_data[0].keys())
            data = [[row[col] for col in columns] for row in summary_data]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"model_comparison_table": table})
            print(f"Logged summary table with {len(summary_data)} models")
        
        # Log overall statistics
        total_samples = sum(r['total_samples'] for r in all_results if r is not None)
        total_valid_preds = sum(r.get('valid_pred_count', 0) for r in all_results if r is not None)
        
        overall_stats = {
            "overall/total_models": len([r for r in all_results if r is not None]),
            "overall/total_samples": total_samples,
            "overall/total_valid_predictions": total_valid_preds,
            "overall/valid_prediction_rate": total_valid_preds / total_samples if total_samples > 0 else 0,
            "overall/images_logged": total_images_logged
        }
        wandb.log(overall_stats)
        print("Logged overall statistics")
        
        # Create summary log
        wandb.summary.update({
            "total_models_evaluated": len([r for r in all_results if r is not None]),
            "total_samples_processed": total_samples,
            "total_visualizations_logged": total_images_logged,
            "evaluation_completed": True
        })
        
        wandb.finish()
        print(f"WandB logging completed successfully! Visit: {run.url}")
        
    except Exception as e:
        print(f"Error logging to WandB: {e}")
        import traceback
        traceback.print_exc()
        try:
            wandb.finish()
        except:
            pass


def save_results_to_csv(all_results, output_path):
    """Save results summary to CSV file"""
    try:
        summary_data = []
        for result in all_results:
            if result is None:
                continue
                
            model_name = result["model_name"]
            rodeo_scores = result.get("rodeo_scores", {})
            mean_ap = result.get("mean_ap", {})
            
            summary_data.append({
                "model": model_name,
                "processed_samples": result["processed_samples"],
                "skipped_samples": result["skipped_samples"],
                "total_samples": result["total_samples"],
                "valid_pred_count": result.get("valid_pred_count", 0),
                "valid_pred_rate": result.get("valid_pred_rate", 0),
                "success_rate": result["processed_samples"] / result["total_samples"] if result["total_samples"] > 0 else 0,
                "rodeo_localization": rodeo_scores.get("RoDeO/localization", 0),
                "rodeo_shape_matching": rodeo_scores.get("RoDeO/shape_matching", 0),
                "rodeo_classification": rodeo_scores.get("RoDeO/classification", 0),
                "rodeo_total": rodeo_scores.get("RoDeO/total", 0),
                "map50_95": mean_ap["map50_95"],
                "map50": mean_ap["map50"],
                "map75": mean_ap["map75"]
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


def save_results_to_json(all_results, output_path):
    """Save simplified results to JSON file"""
    try:
        json_results = {
            "evaluation_metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "results_folder": RESULTS_FOLDER,
                "original_data_path": ORIGINAL_DATA_PATH,
                "total_models_processed": len([r for r in all_results if r is not None]),
                "classes": CLASSES,
                "configuration": {
                    "bbox_coordinate_scale": BBOX_COORDINATE_SCALE,
                    "class_match_threshold": CLASS_MATCH_THRESHOLD,
                    "visualize_limit": VISUALIZE_LIMIT,
                    "default_bbox": DEFAULT_BBOX
                }
            },
            "models": {}
        }
        
        for result in all_results:
            if result is None:
                continue
                
            model_name = result["model_name"]
            rodeo_scores = result.get("rodeo_scores", {})
            mean_ap = result.get("mean_ap", {})
            
            # Simplified model data
            model_data = {
                "model_name": model_name,
                "processing_stats": {
                    "processed_samples": result["processed_samples"],
                    "skipped_samples": result["skipped_samples"],
                    "total_samples": result["total_samples"],
                    "valid_pred_count": result.get("valid_pred_count", 0),
                    "valid_pred_rate": result.get("valid_pred_rate", 0),
                    "success_rate": result["processed_samples"] / result["total_samples"] if result["total_samples"] > 0 else 0,
                    # Debug info for RoDeO vs mAP analysis
                    "rodeo_pred_total": result.get("rodeo_pred_total", 0),
                    "rodeo_gt_total": result.get("rodeo_gt_total", 0),
                    "sv_pred_total": result.get("sv_pred_total", 0),
                    "sv_gt_total": result.get("sv_gt_total", 0)
                },
                "rodeo_scores": dict(rodeo_scores) if rodeo_scores else {},
                "mean_average_precision": {
                    "map50_95": float(mean_ap["map50_95"]) if mean_ap["map50_95"] is not None else 0.0,
                    "map50": float(mean_ap["map50"]) if mean_ap["map50"] is not None else 0.0,
                    "map75": float(mean_ap["map75"]) if mean_ap["map75"] is not None else 0.0
                },
                "visualization_info": {
                    "num_visual_samples": len(result.get("visuals", [])),
                    "captions_available": len(result.get("captions", []))
                }
            }
            
            json_results["models"][model_name] = model_data
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to JSON: {output_path}")
        
    except Exception as e:
        print(f"Error saving results to JSON: {e}")


def main():
    """Main execution function"""
    print("Starting multi-model evaluation...")
    print(f"Results folder: {RESULTS_FOLDER}")
    print(f"Original data path: {ORIGINAL_DATA_PATH}")
    print(f"WandB disabled: {WANDB_DISABLED}")
    
    # Disable wandb if configured
    if WANDB_DISABLED:
        os.environ["WANDB_MODE"] = "disabled"
    
    # Get all model folders
    model_folders = get_model_folders(RESULTS_FOLDER)
    if not model_folders:
        print("No model folders found!")
        return
    
    print(f"Found {len(model_folders)} model folders:")
    for folder in model_folders:
        print(f"  - {os.path.basename(folder)}")
    
    # Process each model
    all_results = []
    for model_folder in model_folders:
        print(f"\n{'='*50}")
        result = process_single_model(model_folder, ORIGINAL_DATA_PATH)
        all_results.append(result)
        
        if result:
            print(f"Model: {result['model_name']}")
            print(f"Processed: {result['processed_samples']}/{result['total_samples']}")
            print(f"Valid predictions: {result.get('valid_pred_count', 0)} ({result.get('valid_pred_rate', 0):.2%})")
            print(f"Skipped: {result['skipped_samples']}")
            print(f"Visualizations: {len(result.get('visuals', []))}")
            
            # Debug info for RoDeO vs mAP analysis
            print(f"RoDeO Detection counts - Pred: {result.get('rodeo_pred_total', 0)}, GT: {result.get('rodeo_gt_total', 0)}")
            print(f"Supervision Detection counts - Pred: {result.get('sv_pred_total', 0)}, GT: {result.get('sv_gt_total', 0)}")
            
            if result.get('rodeo_scores'):
                print("RoDeO Scores:")
                for key, val in result['rodeo_scores'].items():
                    print(f"  {key}: {val:.4f}")
            if result.get('mean_ap'):
                print("mAP Scores:")
                for key, val in result['mean_ap'].items():
                    print(f"  {key}: {val:.4f}")
                    
            # Analysis of potential discrepancy
            if result.get('rodeo_scores') and result.get('mean_ap'):
                rodeo_main = result['rodeo_scores'].get('iou', 0)
                map50 = result['mean_ap'].get('map50', 0)
                if rodeo_main > 0.5 and map50 < 0.1:
                    print(f"⚠️  WARNING: High RoDeO ({rodeo_main:.3f}) but low mAP@50 ({map50:.3f}) - potential evaluation inconsistency!")
                    print(f"   This may indicate RoDeO is counting empty/invalid predictions as matches.")
                    print(f"   RoDeO detections: {result.get('rodeo_pred_total', 0)}, Supervision detections: {result.get('sv_pred_total', 0)}")
    
    # Filter out None results
    valid_results = [r for r in all_results if r is not None]
    
    if not valid_results:
        print("No valid results to process!")
        return
    
    print(f"\n{'='*50}")
    print(f"Successfully processed {len(valid_results)} models")
    
    # Save results in simplified formats
    output_dir = RESULTS_FOLDER  # Save results in the results folder itself
    
    # Save CSV summary
    output_csv = os.path.join(output_dir, OUTPUT_CSV_NAME)
    save_results_to_csv(valid_results, output_csv)
    
    # Save JSON results
    output_json = os.path.join(output_dir, OUTPUT_JSON_NAME)
    save_results_to_json(valid_results, output_json)
    
    # Log to wandb with visualizations (only if we have valid results)
    print("\nStarting WandB logging...")
    log_results_to_wandb(valid_results)
    
    print(f"\n{'='*50}")
    print("Evaluation completed!")
    print(f"Processed {len(valid_results)} models successfully")
    print("Results saved to:")
    print(f"  - CSV Summary: {output_csv}")
    print(f"  - JSON Results: {output_json}")
    print(f"  - WandB: {'Enabled' if not WANDB_DISABLED else 'Disabled'}")
    
    # Print summary statistics
    total_samples = sum(r['total_samples'] for r in valid_results)
    total_valid_preds = sum(r.get('valid_pred_count', 0) for r in valid_results)
    print(f"\nOverall Statistics:")
    print(f"  - Total samples across all models: {total_samples}")
    print(f"  - Total samples with valid predictions: {total_valid_preds}")
    print(f"  - Overall valid prediction rate: {total_valid_preds/total_samples:.2%}" if total_samples > 0 else "")


if __name__ == "__main__":
    main()

