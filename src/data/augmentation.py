"""
Data Augmentation for Face Mask Detection
Implements various augmentation techniques while preserving bounding box accuracy.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple


def random_flip_horizontal(image: tf.Tensor, bbox: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Randomly flip image horizontally and adjust bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box [xmin, ymin, xmax, ymax] in normalized coordinates
        
    Returns:
        Augmented image and adjusted bbox
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        
        # Adjust bbox coordinates for horizontal flip
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        bbox = tf.stack([1.0 - xmax, ymin, 1.0 - xmin, ymax])
    
    return image, bbox


def random_brightness(image: tf.Tensor, max_delta: float = 0.2) -> tf.Tensor:
    """
    Randomly adjust image brightness.
    
    Args:
        image: Input image
        max_delta: Maximum brightness adjustment
        
    Returns:
        Augmented image
    """
    image = tf.image.random_brightness(image, max_delta)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def random_contrast(image: tf.Tensor, lower: float = 0.8, upper: float = 1.2) -> tf.Tensor:
    """
    Randomly adjust image contrast.
    
    Args:
        image: Input image
        lower: Lower bound for contrast factor
        upper: Upper bound for contrast factor
        
    Returns:
        Augmented image
    """
    image = tf.image.random_contrast(image, lower, upper)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def random_saturation(image: tf.Tensor, lower: float = 0.8, upper: float = 1.2) -> tf.Tensor:
    """
    Randomly adjust image saturation.
    
    Args:
        image: Input image
        lower: Lower bound for saturation factor
        upper: Upper bound for saturation factor
        
    Returns:
        Augmented image
    """
    image = tf.image.random_saturation(image, lower, upper)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def random_hue(image: tf.Tensor, max_delta: float = 0.05) -> tf.Tensor:
    """
    Randomly adjust image hue.
    
    Args:
        image: Input image
        max_delta: Maximum hue adjustment
        
    Returns:
        Augmented image
    """
    image = tf.image.random_hue(image, max_delta)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def color_jitter(image: tf.Tensor) -> tf.Tensor:
    """
    Apply random color jittering (brightness, contrast, saturation, hue).
    
    Args:
        image: Input image
        
    Returns:
        Augmented image
    """
    image = random_brightness(image, max_delta=0.2)
    image = random_contrast(image, lower=0.8, upper=1.2)
    image = random_saturation(image, lower=0.8, upper=1.2)
    image = random_hue(image, max_delta=0.05)
    return image


def augment_training_data(image: tf.Tensor, bbox: tf.Tensor, class_id: tf.Tensor):
    """
    Apply augmentation pipeline for training data.
    
    Args:
        image: Input image
        bbox: Bounding box
        class_id: Class ID
        
    Returns:
        Augmented image, bbox, and class_id
    """
    # Random horizontal flip (affects bbox)
    image, bbox = random_flip_horizontal(image, bbox)
    
    # Color augmentations (don't affect bbox)
    image = color_jitter(image)
    
    # Ensure values are in valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    bbox = tf.clip_by_value(bbox, 0.0, 1.0)
    
    return image, bbox, class_id


def create_augmented_dataset(dataset: tf.data.Dataset, augment: bool = True) -> tf.data.Dataset:
    """
    Create an augmented dataset from the original dataset.
    
    Args:
        dataset: Original TensorFlow dataset
        augment: Whether to apply augmentation
        
    Returns:
        Augmented dataset
    """
    if not augment:
        return dataset
    
    def augment_fn(image, labels):
        bbox = labels['bbox']
        class_id = labels['class']
        
        # Apply augmentation
        image, bbox, class_id = augment_training_data(image, bbox, class_id)
        
        return image, {'bbox': bbox, 'class': class_id}
    
    return dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)


if __name__ == "__main__":
    # Test augmentation
    print("Augmentation module loaded successfully!")
    print("Available augmentation functions:")
    print("  - random_flip_horizontal")
    print("  - random_brightness")
    print("  - random_contrast")
    print("  - random_saturation")
    print("  - random_hue")
    print("  - color_jitter")
    print("  - augment_training_data")
    print("  - create_augmented_dataset")
