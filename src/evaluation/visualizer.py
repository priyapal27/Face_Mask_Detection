"""
Visualization Utilities for Face Mask Detection
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional


# Class names and colors
CLASS_NAMES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
CLASS_COLORS = {
    0: 'green',      # with_mask
    1: 'red',        # without_mask
    2: 'orange'      # mask_weared_incorrect
}


def denormalize_bbox(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert normalized bbox to pixel coordinates.
    
    Args:
        bbox: Normalized bbox [xmin, ymin, xmax, ymax] in [0, 1]
        img_width: Image width
        img_height: Image height
        
    Returns:
        Bbox in pixel coordinates
    """
    xmin, ymin, xmax, ymax = bbox
    return np.array([
        xmin * img_width,
        ymin * img_height,
        xmax * img_width,
        ymax * img_height
    ])


def visualize_prediction(
    image: np.ndarray,
    pred_bbox: np.ndarray,
    pred_class: int,
    pred_confidence: float,
    true_bbox: Optional[np.ndarray] = None,
    true_class: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Visualize a single prediction with bounding box.
    
    Args:
        image: Input image (normalized or uint8)
        pred_bbox: Predicted bbox (normalized)
        pred_class: Predicted class ID
        pred_confidence: Prediction confidence
        true_bbox: Ground truth bbox (optional)
        true_class: Ground truth class (optional)
        save_path: Path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display image
    if image.max() <= 1.0:
        ax.imshow(image)
    else:
        ax.imshow(image.astype(np.uint8))
    
    img_height, img_width = image.shape[:2]
    
    # Draw predicted bbox
    pred_bbox_pixel = denormalize_bbox(pred_bbox, img_width, img_height)
    x, y, x2, y2 = pred_bbox_pixel
    width, height = x2 - x, y2 - y
    
    rect = patches.Rectangle(
        (x, y), width, height,
        linewidth=3,
        edgecolor=CLASS_COLORS[pred_class],
        facecolor='none',
        linestyle='-'
    )
    ax.add_patch(rect)
    
    # Add label
    label = f"{CLASS_NAMES[pred_class]}: {pred_confidence:.2f}"
    ax.text(
        x, y - 10,
        label,
        color='white',
        fontsize=12,
        bbox=dict(facecolor=CLASS_COLORS[pred_class], alpha=0.8, edgecolor='none')
    )
    
    # Draw ground truth bbox if provided
    if true_bbox is not None and true_class is not None:
        true_bbox_pixel = denormalize_bbox(true_bbox, img_width, img_height)
        x, y, x2, y2 = true_bbox_pixel
        width, height = x2 - x, y2 - y
        
        rect_gt = patches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor=CLASS_COLORS[true_class],
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect_gt)
        
        # Add ground truth label
        gt_label = f"GT: {CLASS_NAMES[true_class]}"
        ax.text(
            x, y2 + 20,
            gt_label,
            color='white',
            fontsize=10,
            bbox=dict(facecolor=CLASS_COLORS[true_class], alpha=0.6, edgecolor='none')
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch_predictions(
    images: np.ndarray,
    pred_bboxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_confidences: np.ndarray,
    true_bboxes: Optional[np.ndarray] = None,
    true_classes: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
    num_samples: int = 10
):
    """
    Visualize predictions for a batch of images.
    
    Args:
        images: Batch of images
        pred_bboxes: Predicted bboxes
        pred_classes: Predicted classes
        pred_confidences: Prediction confidences
        true_bboxes: Ground truth bboxes (optional)
        true_classes: Ground truth classes (optional)
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        save_path = None
        if save_dir:
            save_path = Path(save_dir) / f"prediction_{i+1}.png"
        
        true_bbox = true_bboxes[i] if true_bboxes is not None else None
        true_class = true_classes[i] if true_classes is not None else None
        
        visualize_prediction(
            images[i],
            pred_bboxes[i],
            pred_classes[i],
            pred_confidences[i],
            true_bbox,
            true_class,
            save_path
        )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Bbox loss
    axes[0, 1].plot(history['train_bbox_loss'], label='Train Bbox Loss', linewidth=2)
    axes[0, 1].plot(history['val_bbox_loss'], label='Val Bbox Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Bounding Box Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Class loss
    axes[1, 0].plot(history['train_class_loss'], label='Train Class Loss', linewidth=2)
    axes[1, 0].plot(history['val_class_loss'], label='Val Class Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 1].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    axes[1, 1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Classification Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Visualization Module")
    print("=" * 60)
    print("Available functions:")
    print("  - visualize_prediction")
    print("  - visualize_batch_predictions")
    print("  - plot_confusion_matrix")
    print("  - plot_training_history")
    print("\nVisualization module loaded successfully!")
