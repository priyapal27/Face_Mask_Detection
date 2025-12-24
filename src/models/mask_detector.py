"""
Face Mask Detector Model Architecture
Transfer learning-based model using MobileNetV2 backbone.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Dict


def create_mask_detector(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 3,
    backbone: str = 'mobilenet_v2',
    trainable_backbone: bool = False
) -> Model:
    """
    Create a face mask detection model with bounding box regression and classification.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of classes (3: with_mask, without_mask, mask_weared_incorrect)
        backbone: Backbone architecture ('mobilenet_v2', 'resnet50', 'efficientnet_b0')
        trainable_backbone: Whether to fine-tune the backbone
        
    Returns:
        Keras Model with two outputs: bbox and class predictions
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name='image_input')
    
    # Load pre-trained backbone
    if backbone == 'mobilenet_v2':
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif backbone == 'resnet50':
        base_model = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif backbone == 'efficientnet_b0':
        base_model = keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Freeze/unfreeze backbone
    base_model.trainable = trainable_backbone
    
    # Extract features
    x = base_model(inputs, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Shared dense layers
    x = layers.Dense(512, activation='relu', name='shared_dense_1')(x)
    x = layers.Dropout(0.3, name='shared_dropout_1')(x)
    x = layers.Dense(256, activation='relu', name='shared_dense_2')(x)
    x = layers.Dropout(0.3, name='shared_dropout_2')(x)
    
    # Bounding box regression head
    bbox_head = layers.Dense(128, activation='relu', name='bbox_dense_1')(x)
    bbox_head = layers.Dropout(0.2, name='bbox_dropout')(bbox_head)
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(bbox_head)
    
    # Classification head
    class_head = layers.Dense(128, activation='relu', name='class_dense_1')(x)
    class_head = layers.Dropout(0.2, name='class_dropout')(class_head)
    class_output = layers.Dense(num_classes, activation='softmax', name='class')(class_head)
    
    # Create model
    model = Model(
        inputs=inputs,
        outputs={'bbox': bbox_output, 'class': class_output},
        name='face_mask_detector'
    )
    
    return model


def create_lightweight_detector(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Model:
    """
    Create a lightweight custom CNN for faster inference.
    
    Args:
        input_shape: Input image shape
        
    Returns:
        Lightweight Keras Model
    """
    inputs = keras.Input(shape=input_shape, name='image_input')
    
    # Convolutional blocks
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output heads
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(x)
    class_output = layers.Dense(3, activation='softmax', name='class')(x)
    
    model = Model(
        inputs=inputs,
        outputs={'bbox': bbox_output, 'class': class_output},
        name='lightweight_mask_detector'
    )
    
    return model


def get_model_summary(model: Model) -> None:
    """
    Print model summary and architecture details.
    
    Args:
        model: Keras model
    """
    print("=" * 80)
    print(f"Model: {model.name}")
    print("=" * 80)
    model.summary()
    print("\n" + "=" * 80)
    print("Model Outputs:")
    for output_name, output_tensor in model.output.items():
        print(f"  {output_name}: {output_tensor.shape}")
    print("=" * 80)


if __name__ == "__main__":
    # Test model creation
    print("Creating Face Mask Detector with MobileNetV2 backbone...")
    model = create_mask_detector(backbone='mobilenet_v2')
    get_model_summary(model)
    
    print("\n\nCreating Lightweight Detector...")
    lightweight_model = create_lightweight_detector()
    get_model_summary(lightweight_model)
    
    # Test forward pass
    print("\n\nTesting forward pass...")
    dummy_input = tf.random.normal((1, 224, 224, 3))
    outputs = model(dummy_input)
    print(f"Bbox output shape: {outputs['bbox'].shape}")
    print(f"Class output shape: {outputs['class'].shape}")
    print("\n✅ Model architecture created successfully!")
