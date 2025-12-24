"""
Evaluation Metrics for Face Mask Detection
Implements mAP, precision, recall, and other metrics.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Tuple


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]
        
    Returns:
        IoU value
    """
    # Calculate intersection
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    
    inter_width = max(0, xmax - xmin)
    inter_height = max(0, ymax - ymin)
    inter_area = inter_width * inter_height
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-7)
    
    return iou


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) from precision-recall curve.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        
    Returns:
        Average Precision
    """
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def calculate_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 3
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) at given IoU threshold.
    
    Args:
        predictions: List of prediction dicts with 'bbox', 'class', 'confidence'
        ground_truths: List of ground truth dicts with 'bbox', 'class'
        iou_threshold: IoU threshold for considering a detection as correct
        num_classes: Number of classes
        
    Returns:
        Dictionary with mAP and per-class AP
    """
    class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    aps = []
    
    for class_id in range(num_classes):
        # Filter predictions and ground truths for this class
        class_preds = [p for p in predictions if p['class'] == class_id]
        class_gts = [g for g in ground_truths if g['class'] == class_id]
        
        if len(class_gts) == 0:
            continue
        
        # Sort predictions by confidence
        class_preds = sorted(class_preds, key=lambda x: x.get('confidence', 1.0), reverse=True)
        
        # Calculate precision and recall at each threshold
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        matched_gts = set()
        
        for i, pred in enumerate(class_preds):
            max_iou = 0
            max_gt_idx = -1
            
            for j, gt in enumerate(class_gts):
                if j in matched_gts:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = j
            
            if max_iou >= iou_threshold:
                tp[i] = 1
                matched_gts.add(max_gt_idx)
            else:
                fp[i] = 1
        
        # Calculate cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
        
        # Calculate AP
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
        
        print(f"  {class_names[class_id]}: AP = {ap:.4f}")
    
    # Calculate mAP
    mAP = np.mean(aps) if aps else 0.0
    
    return {
        'mAP': mAP,
        'per_class_ap': aps
    }


def calculate_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3
) -> Dict[str, np.ndarray]:
    """
    Calculate precision and recall per class.
    
    Args:
        y_true: Ground truth class labels
        y_pred: Predicted class labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with precision and recall per class
    """
    class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_id in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))
        
        # Calculate metrics
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f"  {class_names[class_id]}:")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")
    
    return {
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f1_score': np.array(f1_scores)
    }


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> np.ndarray:
    """
    Generate confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Confusion matrix
    """
    if class_names is None:
        class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                ", "  ".join(f"{name[:10]:>10}" for name in class_names))
    print("Actual")
    for i, name in enumerate(class_names):
        print(f"{name[:10]:>10}", "  ".join(f"{cm[i, j]:>10}" for j in range(len(class_names))))
    
    return cm


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        iou_threshold: IoU threshold for mAP calculation
        
    Returns:
        Dictionary with all evaluation metrics
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    all_predictions = []
    all_ground_truths = []
    y_true_classes = []
    y_pred_classes = []
    
    for images, labels in test_dataset:
        predictions = model(images, training=False)
        
        # Extract predictions and ground truths
        for i in range(len(images)):
            pred_bbox = predictions['bbox'][i].numpy()
            pred_class = np.argmax(predictions['class'][i].numpy())
            pred_confidence = np.max(predictions['class'][i].numpy())
            
            true_bbox = labels['bbox'][i].numpy()
            true_class = np.argmax(labels['class'][i].numpy())
            
            all_predictions.append({
                'bbox': pred_bbox,
                'class': pred_class,
                'confidence': pred_confidence
            })
            
            all_ground_truths.append({
                'bbox': true_bbox,
                'class': true_class
            })
            
            y_true_classes.append(true_class)
            y_pred_classes.append(pred_class)
    
    y_true_classes = np.array(y_true_classes)
    y_pred_classes = np.array(y_pred_classes)
    
    # Calculate mAP
    print("\nCalculating mAP...")
    map_results = calculate_map(all_predictions, all_ground_truths, iou_threshold)
    print(f"\nmAP@IoU={iou_threshold}: {map_results['mAP']:.4f}")
    
    # Calculate precision, recall, F1
    print("\nPer-class Metrics:")
    pr_results = calculate_precision_recall(y_true_classes, y_pred_classes)
    
    # Generate confusion matrix
    cm = generate_confusion_matrix(y_true_classes, y_pred_classes)
    
    # Overall accuracy
    accuracy = np.mean(y_true_classes == y_pred_classes)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    return {
        'mAP': map_results['mAP'],
        'per_class_ap': map_results['per_class_ap'],
        'precision': pr_results['precision'],
        'recall': pr_results['recall'],
        'f1_score': pr_results['f1_score'],
        'confusion_matrix': cm,
        'accuracy': accuracy
    }


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("=" * 60)
    print("Available functions:")
    print("  - calculate_iou")
    print("  - calculate_ap")
    print("  - calculate_map")
    print("  - calculate_precision_recall")
    print("  - generate_confusion_matrix")
    print("  - evaluate_model")
    print("\nMetrics module loaded successfully!")
