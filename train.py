"""
Main Training Script for Face Mask Detection
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset_loader import VOCAnnotationParser, create_dataset
from src.data.preprocessing import preprocess_for_training
from src.data.augmentation import create_augmented_dataset
from src.models.mask_detector import create_mask_detector, get_model_summary
from src.models.losses import create_loss_functions
from src.training.trainer import MaskDetectorTrainer, split_dataset
from src.evaluation.visualizer import plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Face Mask Detection Model')
    
    # Data arguments
    parser.add_argument('--images-dir', type=str, default='data/raw/images',
                        help='Directory containing images')
    parser.add_argument('--annotations-dir', type=str, default='data/raw/annotations',
                        help='Directory containing XML annotations')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'resnet50', 'efficientnet_b0'],
                        help='Backbone architecture')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--bbox-weight', type=float, default=1.0,
                        help='Weight for bbox loss')
    parser.add_argument('--class-weight', type=float, default=1.0,
                        help='Weight for classification loss')
    
    # Data split arguments
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-history', action='store_true',
                        help='Save training history plot')
    
    # Other arguments
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("=" * 80)
    print("Face Mask Detection - Training")
    print("=" * 80)
    
    # Check if dataset exists
    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    
    if not images_dir.exists() or not annotations_dir.exists():
        print("\n❌ Dataset not found!")
        print(f"   Images directory: {images_dir}")
        print(f"   Annotations directory: {annotations_dir}")
        print("\nPlease run: python download_dataset.py")
        sys.exit(1)
    
    # Load dataset
    print("\n📦 Loading dataset...")
    parser = VOCAnnotationParser(str(images_dir), str(annotations_dir))
    annotations = parser.load_dataset()
    
    if len(annotations) == 0:
        print("❌ No annotations found!")
        sys.exit(1)
    
    print(f"   Total samples: {len(annotations)}")
    
    # Split dataset
    print("\n✂️  Splitting dataset...")
    train_annotations, val_annotations, test_annotations = split_dataset(
        annotations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    print(f"   Train: {len(train_annotations)} samples")
    print(f"   Validation: {len(val_annotations)} samples")
    print(f"   Test: {len(test_annotations)} samples")
    
    # Save test annotations for later evaluation
    test_annotations_path = Path('data/processed/test_annotations.npy')
    test_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(test_annotations_path, test_annotations)
    print(f"   Saved test annotations to {test_annotations_path}")
    
    # Create datasets
    print("\n🔄 Creating TensorFlow datasets...")
    target_size = (args.image_size, args.image_size)
    
    train_dataset = create_dataset(
        train_annotations,
        images_dir,
        batch_size=args.batch_size,
        shuffle=True,
        target_size=target_size
    )
    
    val_dataset = create_dataset(
        val_annotations,
        images_dir,
        batch_size=args.batch_size,
        shuffle=False,
        target_size=target_size
    )
    
    # Apply augmentation to training set
    if args.augment:
        print("   Applying data augmentation to training set...")
        train_dataset = create_augmented_dataset(train_dataset, augment=True)
    
    print("   ✅ Datasets created")
    
    # Create model
    print(f"\n🏗️  Creating model with {args.backbone} backbone...")
    model = create_mask_detector(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=3,
        backbone=args.backbone,
        trainable_backbone=False
    )
    
    get_model_summary(model)
    
    # Create trainer
    print("\n🎯 Initializing trainer...")
    trainer = MaskDetectorTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=args.learning_rate,
        bbox_weight=args.bbox_weight,
        class_weight=args.class_weight
    )
    
    # Train model
    print("\n🚀 Starting training...")
    history = trainer.train(
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Save training history
    if args.save_history:
        history_plot_path = Path(args.checkpoint_dir) / 'training_history.png'
        plot_training_history(history, save_path=str(history_plot_path))
    
    # Save final model
    final_model_path = Path(args.checkpoint_dir) / 'final_model.h5'
    model.save(final_model_path)
    print(f"\n💾 Saved final model to {final_model_path}")
    
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Evaluate model: python evaluate.py --model {args.checkpoint_dir}/best_model.h5")
    print(f"2. Run web app: streamlit run src/deployment/streamlit_app.py")
    print(f"3. Real-time detection: python src/deployment/opencv_detector.py --source webcam")


if __name__ == "__main__":
    main()
