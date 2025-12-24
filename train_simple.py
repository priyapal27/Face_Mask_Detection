"""
Simple Training Script for Face Mask Detection (Windows Compatible - No Augmentation)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset_loader import VOCAnnotationParser, create_dataset
from src.models.mask_detector import create_mask_detector
from src.training.trainer import MaskDetectorTrainer, split_dataset
from src.evaluation.visualizer import plot_training_history

def main():
    print("=" * 80)
    print("Face Mask Detection - Training")
    print("=" * 80)
    
    # Configuration
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224
    
    # Load dataset
    print("\nLoading dataset...")
    parser = VOCAnnotationParser(
        images_dir="data/raw/images",
        annotations_dir="data/raw/annotations"
    )
    annotations = parser.load_dataset()
    print(f"Total samples: {len(annotations)}")
    
    # Split dataset
    print("\nSplitting dataset...")
    train_annotations, val_annotations, test_annotations = split_dataset(
        annotations,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"Train: {len(train_annotations)} samples")
    print(f"Validation: {len(val_annotations)} samples")
    print(f"Test: {len(test_annotations)} samples")
    
    # Save test annotations
    test_annotations_path = Path('data/processed/test_annotations.npy')
    test_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(test_annotations_path, test_annotations)
    print(f"Saved test annotations to {test_annotations_path}")
    
    # Create datasets (WITHOUT augmentation for now)
    print("\nCreating TensorFlow datasets...")
    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    
    train_dataset = create_dataset(
        train_annotations,
        Path("data/raw/images"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=target_size
    )
    
    val_dataset = create_dataset(
        val_annotations,
        Path("data/raw/images"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        target_size=target_size
    )
    
    print("Datasets created successfully (no augmentation applied)")
    
    # Create model
    print("\nCreating model with MobileNetV2 backbone...")
    model = create_mask_detector(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=3,
        backbone='mobilenet_v2',
        trainable_backbone=False
    )
    
    print(f"Model created with {len(model.trainable_variables)} trainable variables")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = MaskDetectorTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=LEARNING_RATE,
        bbox_weight=1.0,
        class_weight=1.0
    )
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    history = trainer.train(
        epochs=EPOCHS,
        checkpoint_dir='models/checkpoints',
        early_stopping_patience=10
    )
    
    # Save training history plot
    print("\nSaving training history plot...")
    history_plot_path = Path('models/checkpoints/training_history.png')
    plot_training_history(history, save_path=str(history_plot_path))
    
    # Save final model
    final_model_path = Path('models/checkpoints/final_model.h5')
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print("\nNext steps:")
    print("1. Evaluate model: python evaluate.py --model models/checkpoints/best_model.h5 --visualize")
    print("2. Run web app: streamlit run src/deployment/streamlit_app.py")
    print("3. Real-time detection: python src/deployment/opencv_detector.py --source webcam")

if __name__ == "__main__":
    main()
