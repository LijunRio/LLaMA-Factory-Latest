"""
Medical Image Grounding Evaluation Script
Based on 01_caculate_res.py structure but using MedicalGroundingEvaluator metrics
Handles batch evaluation of multiple models with file I/O and coordinate conversion
"""

import json
import re
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION PARAMETERS ====================
# # Paths Configuration
# RESULTS_FOLDER = "/u/home/lj0/Code/LLaMA-Factory-Latest/evaluate_outputs/results/vinder_adkg_test"
# ORIGINAL_DATA_PATH = "/u/home/lj0/Code/LLaMA-Factory-Latest/data/0_my_dataset/vindr/qwen2_vindr_input_test_len_2108.json"

# Alternative path for PadChest
RESULTS_FOLDER = "/u/home/lj0/Code/LLaMA-Factory-Latest/evaluate_outputs/results/padchest_gr_test"
ORIGINAL_DATA_PATH = "/u/home/lj0/Code/LLaMA-Factory-Latest/data/0_my_dataset/padchest_gt/padchest_input_data_qwen2_test_len_1285.json"

PADCHEST_GR=True
if PADCHEST_GR:
    IMAGE_PATH_REPLACEMENT = {
    "from": "/home/june/datasets/padchest_512/images_512/",
    "to": "/u/home/lj0/datasets/2gpu-workstations/padchest_gr/images_512/"
    }
else:
    IMAGE_PATH_REPLACEMENT = {
        "from": "/home/june/datasets/",
        "to": "/u/home/lj0/datasets/2gpu-workstations/"
    }

# Model Evaluation Configuration
PREDICTION_FILE_NAME = "generated_predictions.jsonl"
DEFAULT_BBOX = [0, 0, 0, 0]  # Default bbox for no_output cases
BBOX_COORDINATE_SCALE = 1000  # Scale factor for bbox coordinates

# Output Configuration
OUTPUT_CSV_NAME = "medical_grounding_evaluation.csv"
OUTPUT_JSON_NAME = "medical_grounding_results.json"
# ==================== END CONFIGURATION ====================


class MedicalGroundingEvaluator:
    """
    Comprehensive evaluator for medical image grounding tasks
    
    Calculates 10 key metrics:
    1. mAP30, mAP50, mAP75 - Average Precision at different IoU thresholds
    2. ACC30, ACC50 - Accuracy at different IoU thresholds  
    3. Recall30, Recall50 - Recall at different IoU thresholds
    4. Shape_Matching - Shape matching score
    5. Detection_Rate - Valid detection rate
    6. Miss_Rate - Miss detection rate
    """
    
    def __init__(self, iou_thresholds: List[float] = [0.3, 0.5, 0.75]):
        """
        Initialize evaluator
        
        Args:
            iou_thresholds: IoU thresholds for evaluation
        """
        self.iou_thresholds = iou_thresholds
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.ground_truths = []
        self.image_sizes = []
    
    def add_sample(self, pred_boxes: List[List[float]], gt_boxes: List[List[float]], 
                   image_size: Tuple[int, int] = None):
        """
        Add a sample for evaluation
        
        Args:
            pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
            gt_boxes: List of ground truth bounding boxes [x1, y1, x2, y2]  
            image_size: (width, height) of the image
        """
        self.predictions.append(pred_boxes if pred_boxes else [])
        self.ground_truths.append(gt_boxes if gt_boxes else [])
        self.image_sizes.append(image_size)
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value between 0 and 1
        """
        if not self.is_valid_box(box1) or not self.is_valid_box(box2):
            return 0.0
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def is_valid_box(self, box: List[float]) -> bool:
        """
        Check if a bounding box is valid
        
        Args:
            box: [x1, y1, x2, y2] format
            
        Returns:
            True if box is valid
        """
        if not box or len(box) != 4:
            return False
        
        x1, y1, x2, y2 = box
        
        # Check for default/invalid boxes
        if box == [0, 0, 0, 0]:
            return False
        
        # Check for valid coordinates
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Check for negative coordinates (optional)
        if any(coord < 0 for coord in box):
            return False
        
        return True
    
    def calculate_shape_matching(self, pred_box: List[float], gt_box: List[float]) -> float:
        """
        Calculate shape matching score between predicted and ground truth boxes
        
        Args:
            pred_box, gt_box: [x1, y1, x2, y2] format
            
        Returns:
            Shape matching score between 0 and 1
        """
        if not self.is_valid_box(pred_box) or not self.is_valid_box(gt_box):
            return 0.0
        
        # Calculate dimensions
        pred_w = pred_box[2] - pred_box[0]
        pred_h = pred_box[3] - pred_box[1]
        gt_w = gt_box[2] - gt_box[0]
        gt_h = gt_box[3] - gt_box[1]
        
        # Aspect ratio similarity
        pred_ratio = pred_w / pred_h if pred_h > 0 else 1
        gt_ratio = gt_w / gt_h if gt_h > 0 else 1
        ratio_sim = 1 - abs(pred_ratio - gt_ratio) / max(pred_ratio, gt_ratio)
        
        # Scale similarity
        pred_area = pred_w * pred_h
        gt_area = gt_w * gt_h
        scale_sim = min(pred_area, gt_area) / max(pred_area, gt_area) if max(pred_area, gt_area) > 0 else 0
        
        # Combined shape matching score
        return (ratio_sim + scale_sim) / 2
    
    def match_boxes(self, pred_boxes: List[List[float]], gt_boxes: List[List[float]], 
                   iou_threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match predicted boxes with ground truth boxes using Hungarian algorithm
        
        Args:
            pred_boxes: List of predicted boxes
            gt_boxes: List of ground truth boxes
            iou_threshold: IoU threshold for matching
            
        Returns:
            matches: List of (pred_idx, gt_idx) pairs
            unmatched_preds: List of unmatched prediction indices
            unmatched_gts: List of unmatched ground truth indices
        """
        if not pred_boxes or not gt_boxes:
            return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                if self.is_valid_box(pred_box):
                    iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)
        
        # Simple greedy matching (can be replaced with Hungarian algorithm)
        matches = []
        matched_preds = set()
        matched_gts = set()
        
        # Sort by IoU in descending order
        candidates = []
        for i in range(len(pred_boxes)):
            for j in range(len(gt_boxes)):
                if iou_matrix[i, j] >= iou_threshold:
                    candidates.append((iou_matrix[i, j], i, j))
        
        candidates.sort(reverse=True)
        
        for iou_val, pred_idx, gt_idx in candidates:
            if pred_idx not in matched_preds and gt_idx not in matched_gts:
                matches.append((pred_idx, gt_idx))
                matched_preds.add(pred_idx)
                matched_gts.add(gt_idx)
        
        unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
        unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gts]
        
        return matches, unmatched_preds, unmatched_gts
    
    def calculate_detection_rate(self) -> float:
        """
        Calculate detection rate (percentage of samples with valid predictions)
        
        Returns:
            Detection rate between 0 and 1
        """
        if not self.predictions:
            return 0.0
        
        valid_detections = 0
        for pred_boxes in self.predictions:
            if any(self.is_valid_box(box) for box in pred_boxes):
                valid_detections += 1
        
        return valid_detections / len(self.predictions)
    
    def calculate_accuracy(self, iou_threshold: float) -> float:
        """
        Calculate accuracy at given IoU threshold
        
        Args:
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Accuracy between 0 and 1
        """
        if not self.predictions or not self.ground_truths:
            return 0.0
        
        correct_samples = 0
        total_samples = len(self.predictions)
        
        for pred_boxes, gt_boxes in zip(self.predictions, self.ground_truths):
            if not gt_boxes:  # Skip samples without ground truth
                continue
                
            # Check if at least one prediction matches at least one GT
            sample_correct = False
            for pred_box in pred_boxes:
                if not self.is_valid_box(pred_box):
                    continue
                for gt_box in gt_boxes:
                    if self.calculate_iou(pred_box, gt_box) >= iou_threshold:
                        sample_correct = True
                        break
                if sample_correct:
                    break
            
            if sample_correct:
                correct_samples += 1
        
        return correct_samples / total_samples if total_samples > 0 else 0.0
    
    def calculate_recall(self, iou_threshold: float) -> float:
        """
        Calculate recall at given IoU threshold
        
        Args:
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Recall between 0 and 1
        """
        if not self.predictions or not self.ground_truths:
            return 0.0
        
        total_gt = 0
        total_tp = 0
        
        for pred_boxes, gt_boxes in zip(self.predictions, self.ground_truths):
            if not gt_boxes:
                continue
            
            matches, _, _ = self.match_boxes(pred_boxes, gt_boxes, iou_threshold)
            total_gt += len(gt_boxes)
            total_tp += len(matches)
        
        return total_tp / total_gt if total_gt > 0 else 0.0
    
    def calculate_miss_rate(self) -> float:
        """
        Calculate miss rate (percentage of GT boxes that were not detected)
        
        Returns:
            Miss rate between 0 and 1
        """
        return 1.0 - self.calculate_recall(0.3)  # Use 0.3 as base threshold
    
    def calculate_map(self, iou_threshold: float) -> float:
        """
        Calculate mAP at given IoU threshold
        
        Args:
            iou_threshold: IoU threshold
            
        Returns:
            mAP score between 0 and 1
        """
        if not self.predictions or not self.ground_truths:
            return 0.0
        
        # Collect all predictions with confidence scores
        all_predictions = []
        all_ground_truths = []
        
        for i, (pred_boxes, gt_boxes) in enumerate(zip(self.predictions, self.ground_truths)):
            # Add predictions with dummy confidence (all set to 1.0)
            for pred_box in pred_boxes:
                if self.is_valid_box(pred_box):
                    all_predictions.append({
                        'box': pred_box,
                        'confidence': 1.0,  # Dummy confidence
                        'image_id': i
                    })
            
            # Add ground truths
            for gt_box in gt_boxes:
                all_ground_truths.append({
                    'box': gt_box,
                    'image_id': i
                })
        
        if not all_predictions or not all_ground_truths:
            return 0.0
        
        # Sort predictions by confidence (descending)
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall at each threshold
        tp = 0
        fp = 0
        gt_matched = set()
        
        precisions = []
        recalls = []
        
        for pred in all_predictions:
            # Find best matching GT
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(all_ground_truths):
                if gt['image_id'] == pred['image_id']:
                    iou = self.calculate_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Check if match is valid
            if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
                tp += 1
                gt_matched.add(best_gt_idx)
            else:
                fp += 1
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(all_ground_truths) if len(all_ground_truths) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            # Find precision at recall >= t
            precision_at_t = 0
            for i, r in enumerate(recalls):
                if r >= t:
                    precision_at_t = max(precisions[i:])
                    break
            ap += precision_at_t
        
        return ap / 11  # Average over 11 points
    
    def calculate_shape_matching_average(self) -> float:
        """
        Calculate average shape matching score across all valid predictions
        
        Returns:
            Average shape matching score between 0 and 1
        """
        if not self.predictions or not self.ground_truths:
            return 0.0
        
        total_shape_score = 0
        total_matches = 0
        
        for pred_boxes, gt_boxes in zip(self.predictions, self.ground_truths):
            matches, _, _ = self.match_boxes(pred_boxes, gt_boxes, 0.3)  # Use 0.3 threshold
            
            for pred_idx, gt_idx in matches:
                shape_score = self.calculate_shape_matching(pred_boxes[pred_idx], gt_boxes[gt_idx])
                total_shape_score += shape_score
                total_matches += 1
        
        return total_shape_score / total_matches if total_matches > 0 else 0.0
    
    def evaluate(self) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Returns:
            Dictionary containing all 10 metrics
        """
        if not self.predictions or not self.ground_truths:
            return {metric: 0.0 for metric in [
                'mAP30', 'mAP50', 'mAP75', 'ACC30', 'ACC50', 
                'Recall30', 'Recall50', 'Shape_Matching', 
                'Detection_Rate', 'Miss_Rate'
            ]}
        
        results = {}
        
        # Calculate mAP at different thresholds
        results['mAP30'] = self.calculate_map(0.3)
        results['mAP50'] = self.calculate_map(0.5)
        results['mAP75'] = self.calculate_map(0.75)
        
        # Calculate accuracy at different thresholds
        results['ACC30'] = self.calculate_accuracy(0.3)
        results['ACC50'] = self.calculate_accuracy(0.5)
        
        # Calculate recall at different thresholds
        results['Recall30'] = self.calculate_recall(0.3)
        results['Recall50'] = self.calculate_recall(0.5)
        
        # Calculate special metrics
        results['Shape_Matching'] = self.calculate_shape_matching_average()
        results['Detection_Rate'] = self.calculate_detection_rate()
        results['Miss_Rate'] = self.calculate_miss_rate()
        
        return results
    
    def print_results(self, results: Dict[str, float] = None):
        """
        Print evaluation results in a formatted table
        
        Args:
            results: Results dictionary (if None, will calculate)
        """
        if results is None:
            results = self.evaluate()
        
        print("\n" + "="*60)
        print("MEDICAL GROUNDING EVALUATION RESULTS")
        print("="*60)
        
        # Group metrics
        map_metrics = ['mAP30', 'mAP50', 'mAP75']
        acc_metrics = ['ACC30', 'ACC50']
        recall_metrics = ['Recall30', 'Recall50']
        special_metrics = ['Shape_Matching', 'Detection_Rate', 'Miss_Rate']
        
        print("\n📊 Average Precision (mAP):")
        for metric in map_metrics:
            print(f"  {metric:15s}: {results[metric]:.4f}")
        
        print("\n🎯 Accuracy:")
        for metric in acc_metrics:
            print(f"  {metric:15s}: {results[metric]:.4f}")
        
        print("\n🔍 Recall:")
        for metric in recall_metrics:
            print(f"  {metric:15s}: {results[metric]:.4f}")
        
        print("\n⚡ Special Metrics:")
        for metric in special_metrics:
            print(f"  {metric:15s}: {results[metric]:.4f}")
        
        print("\n" + "="*60)


# Example usage and utility functions
def read_jsonl(file_path: str) -> List[Dict]:
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


def extract_all_findings_and_bboxes(text: str) -> List[Dict]:
    """
    Extract findings and bounding boxes from prediction text
    Based on the method from 01_caculate_res.py
    """
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


def convert_bboxes2pixel(data: List[Dict], width: int, height: int) -> List[List[float]]:
    """
    Convert normalized bounding boxes to pixel coordinates
    Based on the method from 01_caculate_res.py
    """
    if not data or width <= 0 or height <= 0:
        return []
        
    converted = []
    try:
        for item in data:
            if not isinstance(item, dict) or 'label' not in item or 'bbox_2d' not in item:
                continue
                
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
                converted.append(new_bbox)
                
    except Exception as e:
        print(f"Error in bbox conversion: {e}")
        
    return converted


def process_single_model(model_folder: str, original_data_path: str) -> Optional[Dict]:
    """
    Process a single model's predictions and calculate metrics
    Based on the structure from 01_caculate_res.py
    """
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
    
    # Initialize evaluator
    evaluator = MedicalGroundingEvaluator()
    
    # Process each sample
    for i, record in tqdm(enumerate(prediction_data), total=len(prediction_data), desc=f"Processing {model_name}"):
        try:
            # Extract predictions and ground truth
            prediction = extract_all_findings_and_bboxes(record.get('predict', ''))
            ground_truth = extract_all_findings_and_bboxes(record.get('label', ''))
            
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
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    # Use default size if image loading fails
                    width, height = 512, 512
            else:
                width, height = 512, 512
            
            # Convert to pixel coordinates
            pixel_prediction = convert_bboxes2pixel(prediction, width, height)
            pixel_ground_truth = convert_bboxes2pixel(ground_truth, width, height)
            
            # Add to evaluator
            evaluator.add_sample(pixel_prediction, pixel_ground_truth, (width, height))
                
        except Exception as e:
            print(f"Error processing sample {i} for {model_name}: {e}")
            continue
    
    # Calculate metrics
    try:
        results = evaluator.evaluate()
        results["model_name"] = model_name
        results["total_samples"] = len(prediction_data)
        
        return results
        
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        return None


def get_model_folders(results_folder: str) -> List[str]:
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


def save_results_to_csv(all_results: List[Dict], output_path: str):
    """Save results summary to CSV file"""
    try:
        summary_data = []
        for result in all_results:
            if result is None:
                continue
                
            summary_data.append({
                "model": result["model_name"],
                "total_samples": result["total_samples"],
                "mAP30": result["mAP30"],
                "mAP50": result["mAP50"],
                "mAP75": result["mAP75"],
                "ACC30": result["ACC30"],
                "ACC50": result["ACC50"],
                "Recall30": result["Recall30"],
                "Recall50": result["Recall50"],
                "Shape_Matching": result["Shape_Matching"],
                "Detection_Rate": result["Detection_Rate"],
                "Miss_Rate": result["Miss_Rate"]
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


def save_results_to_json(all_results: List[Dict], output_path: str):
    """Save detailed results to JSON file"""
    try:
        json_results = {
            "evaluation_metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "results_folder": RESULTS_FOLDER,
                "original_data_path": ORIGINAL_DATA_PATH,
                "total_models_processed": len([r for r in all_results if r is not None]),
                "bbox_coordinate_scale": BBOX_COORDINATE_SCALE
            },
            "models": {}
        }
        
        for result in all_results:
            if result is None:
                continue
                
            model_name = result["model_name"]
            json_results["models"][model_name] = result
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to JSON: {output_path}")
        
    except Exception as e:
        print(f"Error saving results to JSON: {e}")


def main():
    """Main execution function"""
    print("Starting Medical Grounding Evaluation...")
    print(f"Results folder: {RESULTS_FOLDER}")
    print(f"Original data path: {ORIGINAL_DATA_PATH}")
    
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
            print(f"Total samples: {result['total_samples']}")
            print(f"mAP30: {result['mAP30']:.4f}")
            print(f"mAP50: {result['mAP50']:.4f}")
            print(f"mAP75: {result['mAP75']:.4f}")
            print(f"ACC30: {result['ACC30']:.4f}")
            print(f"ACC50: {result['ACC50']:.4f}")
            print(f"Recall30: {result['Recall30']:.4f}")
            print(f"Recall50: {result['Recall50']:.4f}")
            print(f"Shape_Matching: {result['Shape_Matching']:.4f}")
            print(f"Detection_Rate: {result['Detection_Rate']:.4f}")
            print(f"Miss_Rate: {result['Miss_Rate']:.4f}")
    
    # Filter out None results
    valid_results = [r for r in all_results if r is not None]
    
    if not valid_results:
        print("No valid results to process!")
        return
    
    print(f"\n{'='*50}")
    print(f"Successfully processed {len(valid_results)} models")
    
    # Save results
    output_dir = RESULTS_FOLDER
    
    # Save CSV summary
    output_csv = os.path.join(output_dir, OUTPUT_CSV_NAME)
    save_results_to_csv(valid_results, output_csv)
    
    # Save JSON results
    output_json = os.path.join(output_dir, OUTPUT_JSON_NAME)
    save_results_to_json(valid_results, output_json)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to:")
    print(f"  - CSV Summary: {output_csv}")
    print(f"  - JSON Results: {output_json}")


if __name__ == "__main__":
    main()
    
    # Optional: Example usage with dummy data (commented out)
    # Uncomment the following lines if you want to test the evaluator with sample data
    """
    print("\n" + "="*50)
    print("TESTING WITH DUMMY DATA")
    print("="*50)
    
    evaluator = MedicalGroundingEvaluator()
    
    # Add some example samples
    evaluator.add_sample(
        pred_boxes=[[100, 100, 200, 200]],
        gt_boxes=[[90, 90, 210, 210]],
        image_size=(512, 512)
    )
    
    evaluator.add_sample(
        pred_boxes=[[50, 50, 150, 150], [300, 300, 400, 400]],
        gt_boxes=[[45, 45, 155, 155], [295, 295, 405, 405], [100, 400, 200, 450]],
        image_size=(512, 512)
    )
    
    # No prediction example
    evaluator.add_sample(
        pred_boxes=[],
        gt_boxes=[[200, 200, 300, 300]],
        image_size=(512, 512)
    )
    
    # Calculate and display results
    results = evaluator.evaluate()
    evaluator.print_results(results)
    """
