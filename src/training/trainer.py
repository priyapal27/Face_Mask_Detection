"""
Training Pipeline for Face Mask Detection
Implements custom training loop with checkpointing and monitoring.
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.mask_detector import create_mask_detector
from src.models.losses import create_loss_functions
from src.data.dataset_loader import VOCAnnotationParser, create_dataset
from src.data.augmentation import create_augmented_dataset
from src.data.preprocessing import preprocess_for_training


class MaskDetectorTrainer:
    """Trainer class for Face Mask Detection model."""
    
    def __init__(
        self,
        model: keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        learning_rate: float = 0.001,
        bbox_weight: float = 1.0,
        class_weight: float = 1.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            learning_rate: Initial learning rate
            bbox_weight: Weight for bbox loss
            class_weight: Weight for classification loss
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Loss functions
        self.bbox_loss_fn = keras.losses.MeanSquaredError()
        self.class_loss_fn = keras.losses.CategoricalCrossentropy()
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        
        # Metrics
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        self.train_bbox_loss_metric = keras.metrics.Mean(name='train_bbox_loss')
        self.train_class_loss_metric = keras.metrics.Mean(name='train_class_loss')
        self.train_accuracy_metric = keras.metrics.CategoricalAccuracy(name='train_accuracy')
        
        self.val_loss_metric = keras.metrics.Mean(name='val_loss')
        self.val_bbox_loss_metric = keras.metrics.Mean(name='val_bbox_loss')
        self.val_class_loss_metric = keras.metrics.Mean(name='val_class_loss')
        self.val_accuracy_metric = keras.metrics.CategoricalAccuracy(name='val_accuracy')
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_bbox_loss': [],
            'train_class_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_bbox_loss': [],
            'val_class_loss': [],
            'val_accuracy': []
        }
        
        # Best validation loss for checkpointing
        self.best_val_loss = float('inf')
    
    def train_step(self, images, labels):
        """Single training step."""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(images, training=True)
            
            # Calculate losses
            bbox_loss = self.bbox_loss_fn(labels['bbox'], predictions['bbox'])
            class_loss = self.class_loss_fn(labels['class'], predictions['class'])
            total_loss = (self.bbox_weight * bbox_loss) + (self.class_weight * class_loss)
        
        # Backward pass
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss_metric.update_state(total_loss)
        self.train_bbox_loss_metric.update_state(bbox_loss)
        self.train_class_loss_metric.update_state(class_loss)
        self.train_accuracy_metric.update_state(labels['class'], predictions['class'])
        
        return total_loss
    
    def val_step(self, images, labels):
        """Single validation step."""
        # Forward pass
        predictions = self.model(images, training=False)
        
        # Calculate losses
        bbox_loss = self.bbox_loss_fn(labels['bbox'], predictions['bbox'])
        class_loss = self.class_loss_fn(labels['class'], predictions['class'])
        total_loss = (self.bbox_weight * bbox_loss) + (self.class_weight * class_loss)
        
        # Update metrics
        self.val_loss_metric.update_state(total_loss)
        self.val_bbox_loss_metric.update_state(bbox_loss)
        self.val_class_loss_metric.update_state(class_loss)
        self.val_accuracy_metric.update_state(labels['class'], predictions['class'])
        
        return total_loss
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        print(f"\nEpoch {epoch + 1}")
        print("-" * 60)
        
        # Reset metrics
        self.train_loss_metric.reset_state()
        self.train_bbox_loss_metric.reset_state()
        self.train_class_loss_metric.reset_state()
        self.train_accuracy_metric.reset_state()
        
        # Training loop
        for step, (images, labels) in enumerate(self.train_dataset):
            loss = self.train_step(images, labels)
            
            if step % 10 == 0:
                print(f"  Step {step}: loss = {loss.numpy():.4f}", end='\r')
        
        # Print epoch results
        train_loss = self.train_loss_metric.result().numpy()
        train_bbox_loss = self.train_bbox_loss_metric.result().numpy()
        train_class_loss = self.train_class_loss_metric.result().numpy()
        train_acc = self.train_accuracy_metric.result().numpy()
        
        print(f"\n  Train Loss: {train_loss:.4f} (bbox: {train_bbox_loss:.4f}, class: {train_class_loss:.4f})")
        print(f"  Train Accuracy: {train_acc:.4f}")
        
        return train_loss, train_bbox_loss, train_class_loss, train_acc
    
    def validate_epoch(self):
        """Validate for one epoch."""
        # Reset metrics
        self.val_loss_metric.reset_state()
        self.val_bbox_loss_metric.reset_state()
        self.val_class_loss_metric.reset_state()
        self.val_accuracy_metric.reset_state()
        
        # Validation loop
        for images, labels in self.val_dataset:
            self.val_step(images, labels)
        
        # Get results
        val_loss = self.val_loss_metric.result().numpy()
        val_bbox_loss = self.val_bbox_loss_metric.result().numpy()
        val_class_loss = self.val_class_loss_metric.result().numpy()
        val_acc = self.val_accuracy_metric.result().numpy()
        
        print(f"  Val Loss: {val_loss:.4f} (bbox: {val_bbox_loss:.4f}, class: {val_class_loss:.4f})")
        print(f"  Val Accuracy: {val_acc:.4f}")
        
        return val_loss, val_bbox_loss, val_class_loss, val_acc
    
    def train(
        self,
        epochs: int = 50,
        checkpoint_dir: str = 'models/checkpoints',
        early_stopping_patience: int = 10
    ):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
        """
        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_bbox_loss, train_class_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_bbox_loss, val_class_loss, val_acc = self.validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_bbox_loss'].append(train_bbox_loss)
            self.history['train_class_loss'].append(train_class_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_bbox_loss'].append(val_bbox_loss)
            self.history['val_class_loss'].append(val_class_loss)
            self.history['val_accuracy'].append(val_acc)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.h5')
                self.model.save(checkpoint_path)
                print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                break
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history


def split_dataset(annotations, train_ratio=0.7, val_ratio=0.15):
    """Split dataset into train, validation, and test sets."""
    np.random.shuffle(annotations)
    
    n = len(annotations)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_annotations = annotations[:train_end]
    val_annotations = annotations[train_end:val_end]
    test_annotations = annotations[val_end:]
    
    return train_annotations, val_annotations, test_annotations


if __name__ == "__main__":
    print("Face Mask Detection - Training Pipeline")
    print("=" * 60)
    
    # This is a template - actual training requires dataset
    print("\nTo train the model:")
    print("1. Download dataset: python download_dataset.py")
    print("2. Run training script with proper dataset paths")
    print("\n✅ Training module loaded successfully!")
