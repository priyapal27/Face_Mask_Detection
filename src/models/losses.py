"""
Custom Loss Functions for Face Mask Detection
Multi-task loss combining bounding box regression and classification.
"""

import tensorflow as tf
from tensorflow import keras


class CombinedLoss(keras.losses.Loss):
    """
    Combined loss for multi-task learning: bbox regression + classification.
    """
    
    def __init__(
        self,
        bbox_weight: float = 1.0,
        class_weight: float = 1.0,
        name: str = 'combined_loss'
    ):
        """
        Initialize combined loss.
        
        Args:
            bbox_weight: Weight for bounding box loss (lambda_bbox)
            class_weight: Weight for classification loss (lambda_class)
            name: Loss name
        """
        super().__init__(name=name)
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        
        # Individual loss functions
        self.bbox_loss_fn = keras.losses.MeanSquaredError(name='bbox_mse')
        self.class_loss_fn = keras.losses.CategoricalCrossentropy(name='class_cce')
    
    def call(self, y_true, y_pred):
        """
        Compute combined loss.
        
        Args:
            y_true: Dictionary with 'bbox' and 'class' ground truth
            y_pred: Dictionary with 'bbox' and 'class' predictions
            
        Returns:
            Combined loss value
        """
        # Bounding box loss (MSE)
        bbox_loss = self.bbox_loss_fn(y_true['bbox'], y_pred['bbox'])
        
        # Classification loss (Categorical Cross-Entropy)
        class_loss = self.class_loss_fn(y_true['class'], y_pred['class'])
        
        # Combined loss
        total_loss = (self.bbox_weight * bbox_loss) + (self.class_weight * class_loss)
        
        return total_loss
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'bbox_weight': self.bbox_weight,
            'class_weight': self.class_weight
        })
        return config


def bbox_iou_loss(y_true, y_pred):
    """
    IoU-based loss for bounding box regression.
    
    Args:
        y_true: Ground truth bboxes [batch, 4] (xmin, ymin, xmax, ymax)
        y_pred: Predicted bboxes [batch, 4] (xmin, ymin, xmax, ymax)
        
    Returns:
        IoU loss (1 - IoU)
    """
    # Extract coordinates
    true_xmin, true_ymin, true_xmax, true_ymax = tf.split(y_true, 4, axis=-1)
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = tf.split(y_pred, 4, axis=-1)
    
    # Calculate intersection
    inter_xmin = tf.maximum(true_xmin, pred_xmin)
    inter_ymin = tf.maximum(true_ymin, pred_ymin)
    inter_xmax = tf.minimum(true_xmax, pred_xmax)
    inter_ymax = tf.minimum(true_ymax, pred_ymax)
    
    inter_width = tf.maximum(0.0, inter_xmax - inter_xmin)
    inter_height = tf.maximum(0.0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    
    # Calculate union
    true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    union_area = true_area + pred_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-7)
    
    # IoU loss
    loss = 1.0 - iou
    
    return tf.reduce_mean(loss)


def smooth_l1_loss(y_true, y_pred, beta: float = 1.0):
    """
    Smooth L1 loss for bounding box regression.
    Less sensitive to outliers than MSE.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        beta: Threshold for switching between L1 and L2
        
    Returns:
        Smooth L1 loss
    """
    diff = tf.abs(y_true - y_pred)
    
    # Use L2 for small differences, L1 for large differences
    loss = tf.where(
        diff < beta,
        0.5 * tf.square(diff) / beta,
        diff - 0.5 * beta
    )
    
    return tf.reduce_mean(loss)


def focal_loss(y_true, y_pred, alpha: float = 0.25, gamma: float = 2.0):
    """
    Focal loss for handling class imbalance.
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_pred: Predicted probabilities
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss
    """
    # Clip predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Calculate cross-entropy
    ce = -y_true * tf.math.log(y_pred)
    
    # Calculate focal loss
    focal_weight = alpha * tf.pow(1.0 - y_pred, gamma)
    loss = focal_weight * ce
    
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


def create_loss_functions(loss_type: str = 'combined'):
    """
    Create loss functions based on type.
    
    Args:
        loss_type: Type of loss ('combined', 'iou', 'smooth_l1', 'focal')
        
    Returns:
        Dictionary of loss functions for each output
    """
    if loss_type == 'combined':
        return {
            'bbox': keras.losses.MeanSquaredError(),
            'class': keras.losses.CategoricalCrossentropy()
        }
    elif loss_type == 'iou':
        return {
            'bbox': bbox_iou_loss,
            'class': keras.losses.CategoricalCrossentropy()
        }
    elif loss_type == 'smooth_l1':
        return {
            'bbox': smooth_l1_loss,
            'class': keras.losses.CategoricalCrossentropy()
        }
    elif loss_type == 'focal':
        return {
            'bbox': keras.losses.MeanSquaredError(),
            'class': focal_loss
        }
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    y_true_bbox = tf.constant([[0.2, 0.3, 0.6, 0.7]], dtype=tf.float32)
    y_pred_bbox = tf.constant([[0.25, 0.35, 0.65, 0.75]], dtype=tf.float32)
    
    y_true_class = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
    y_pred_class = tf.constant([[0.8, 0.15, 0.05]], dtype=tf.float32)
    
    # Test MSE loss
    mse_loss = keras.losses.MeanSquaredError()
    print(f"MSE Loss (bbox): {mse_loss(y_true_bbox, y_pred_bbox).numpy():.4f}")
    
    # Test IoU loss
    print(f"IoU Loss (bbox): {bbox_iou_loss(y_true_bbox, y_pred_bbox).numpy():.4f}")
    
    # Test Smooth L1 loss
    print(f"Smooth L1 Loss (bbox): {smooth_l1_loss(y_true_bbox, y_pred_bbox).numpy():.4f}")
    
    # Test CCE loss
    cce_loss = keras.losses.CategoricalCrossentropy()
    print(f"CCE Loss (class): {cce_loss(y_true_class, y_pred_class).numpy():.4f}")
    
    # Test Focal loss
    print(f"Focal Loss (class): {focal_loss(y_true_class, y_pred_class).numpy():.4f}")
    
    # Test Combined loss
    combined = CombinedLoss(bbox_weight=1.0, class_weight=1.0)
    y_true = {'bbox': y_true_bbox, 'class': y_true_class}
    y_pred = {'bbox': y_pred_bbox, 'class': y_pred_class}
    print(f"Combined Loss: {combined(y_true, y_pred).numpy():.4f}")
    
    print("\n✅ All loss functions working correctly!")
