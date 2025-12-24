"""
Image Preprocessing Pipeline for Face Mask Detection
"""

import tensorflow as tf
import numpy as np
from typing import Tuple


def resize_image_and_bbox(
    image: tf.Tensor,
    bbox: tf.Tensor,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Resize image to target size. Bounding boxes are already normalized,
    so they don't need adjustment.
    
    Args:
        image: Input image tensor
        bbox: Normalized bounding box [xmin, ymin, xmax, ymax] in [0, 1]
        target_size: Target size (height, width)
        
    Returns:
        Resized image and bbox
    """
    image = tf.image.resize(image, target_size)
    return image, bbox


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image tensor
        
    Returns:
        Normalized image
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


def preprocess_image(
    image_path: str,
    bbox: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image_path: Path to image file
        bbox: Bounding box coordinates (normalized)
        target_size: Target image size
        
    Returns:
        Preprocessed image and bbox
    """
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    
    # Normalize
    image = normalize_image(image)
    
    # Resize
    image, bbox = resize_image_and_bbox(image, bbox, target_size)
    
    return image, bbox


def clip_bbox(bbox: tf.Tensor) -> tf.Tensor:
    """
    Clip bounding box coordinates to [0, 1] range.
    
    Args:
        bbox: Bounding box [xmin, ymin, xmax, ymax]
        
    Returns:
        Clipped bounding box
    """
    return tf.clip_by_value(bbox, 0.0, 1.0)


def validate_bbox(bbox: tf.Tensor) -> bool:
    """
    Validate that bounding box is valid (xmax > xmin, ymax > ymin).
    
    Args:
        bbox: Bounding box [xmin, ymin, xmax, ymax]
        
    Returns:
        True if valid, False otherwise
    """
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    return (xmax > xmin) and (ymax > ymin)


def preprocess_for_training(image: tf.Tensor, bbox: tf.Tensor, class_id: int):
    """
    Preprocessing function for training dataset.
    
    Args:
        image: Input image
        bbox: Bounding box
        class_id: Class ID
        
    Returns:
        Preprocessed image and labels
    """
    # Ensure image is in correct format
    image = tf.ensure_shape(image, [224, 224, 3])
    bbox = tf.ensure_shape(bbox, [4])
    
    # Clip bbox to valid range
    bbox = clip_bbox(bbox)
    
    # One-hot encode class
    class_one_hot = tf.one_hot(class_id, depth=3)
    
    return image, {'bbox': bbox, 'class': class_one_hot}


def preprocess_for_inference(image: tf.Tensor, target_size: Tuple[int, int] = (224, 224)):
    """
    Preprocessing function for inference.
    
    Args:
        image: Input image (can be any size)
        target_size: Target size for model input
        
    Returns:
        Preprocessed image ready for inference
    """
    # Normalize if needed
    if image.dtype == tf.uint8:
        image = tf.cast(image, tf.float32) / 255.0
    
    # Resize first
    image = tf.image.resize(image, target_size)
    
    # Handle image channels AFTER resize to ensure consistency
    # This handles RGB, RGBA, and Grayscale
    channels = tf.shape(image)[-1]
    
    if channels == 4:
        image = image[:, :, :3]
    elif channels == 1:
        image = tf.image.grayscale_to_rgb(image)
    
    # Final check for 3 channels
    image = image[:, :, :3]
    
    # Ensure correct shape for TensorFlow graph
    image = tf.ensure_shape(image, [target_size[0], target_size[1], 3])
    
    return image


if __name__ == "__main__":
    # Test preprocessing
    print("Preprocessing module loaded successfully!")
    print("Available functions:")
    print("  - resize_image_and_bbox")
    print("  - normalize_image")
    print("  - preprocess_image")
    print("  - preprocess_for_training")
    print("  - preprocess_for_inference")
