"""
Main Evaluation Script for Face Mask Detection
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset_loader import VOCAnnotationParser, create_dataset
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualizer import (
    visualize_batch_predictions,
    plot_confusion_matrix,
    CLASS_NAMES
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Face Mask Detection Model')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    
    # Data arguments
    parser.add_argument('--images-dir', type=str, default='data/raw/images',
                        help='Directory containing images')
    parser.add_argument('--annotations-dir', type=str, default='data/raw/annotations',
                        help='Directory containing XML annotations')
    parser.add_argument('--test-annotations', type=str, default='data/processed/test_annotations.npy',
                        help='Path to saved test annotations')
    
    # Evaluation arguments
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for mAP calculation')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--num-visualizations', type=int, default=10,
                        help='Number of predictions to visualize')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for visualizations')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 80)
    print("Face Mask Detection - Evaluation")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    try:
        model = tf.keras.models.load_model(args.model)
        print("   OK - Model loaded successfully")
    except Exception as e:
        print(f"   ERROR - Error loading model: {e}")
        sys.exit(1)
    
    # Load test annotations
    test_annotations_path = Path(args.test_annotations)
    
    if test_annotations_path.exists():
        print(f"\nLoading test annotations from {test_annotations_path}...")
        test_annotations = np.load(test_annotations_path, allow_pickle=True)
        print(f"   Found {len(test_annotations)} test samples")
    else:
        print(f"\n⚠️  Test annotations not found at {test_annotations_path}")
        print("   Loading full dataset and using last 15%...")
        
        images_dir = Path(args.images_dir)
        annotations_dir = Path(args.annotations_dir)
        
        parser = VOCAnnotationParser(str(images_dir), str(annotations_dir))
        all_annotations = parser.load_dataset()
        
        # Use last 15% as test set
        test_start = int(len(all_annotations) * 0.85)
        test_annotations = all_annotations[test_start:]
        print(f"   Using {len(test_annotations)} samples for testing")
    
    print("\nCreating test dataset...")
    images_dir = Path(args.images_dir)
    target_size = (args.image_size, args.image_size)
    
    test_dataset = create_dataset(
        test_annotations,
        images_dir,
        batch_size=args.batch_size,
        shuffle=False,
        target_size=target_size
    )
    
    print("   OK - Dataset created")
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(
        model,
        test_dataset,
        iou_threshold=args.iou_threshold
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"\nmAP@IoU={args.iou_threshold}: {results['mAP']:.4f}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    
    print("\nPer-class Metrics:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"\n  {class_name}:")
        if i < len(results['precision']):
            print(f"    Precision: {results['precision'][i]:.4f}")
            print(f"    Recall: {results['recall'][i]:.4f}")
            print(f"    F1-Score: {results['f1_score'][i]:.4f}")
            if i < len(results['per_class_ap']):
                print(f"    AP: {results['per_class_ap'][i]:.4f}")
    
    # Generate visualizations
    if args.visualize:
        print(f"\nGenerating visualizations...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get some test samples
        images_list = []
        pred_bboxes_list = []
        pred_classes_list = []
        pred_confidences_list = []
        true_bboxes_list = []
        true_classes_list = []
        
        count = 0
        for images, labels in test_dataset:
            predictions = model(images, training=False)
            
            for i in range(len(images)):
                if count >= args.num_visualizations:
                    break
                
                images_list.append(images[i].numpy())
                pred_bboxes_list.append(predictions['bbox'][i].numpy())
                pred_classes_list.append(np.argmax(predictions['class'][i].numpy()))
                pred_confidences_list.append(np.max(predictions['class'][i].numpy()))
                true_bboxes_list.append(labels['bbox'][i].numpy())
                true_classes_list.append(np.argmax(labels['class'][i].numpy()))
                
                count += 1
            
            if count >= args.num_visualizations:
                break
        
        # Visualize predictions
        visualize_batch_predictions(
            np.array(images_list),
            np.array(pred_bboxes_list),
            np.array(pred_classes_list),
            np.array(pred_confidences_list),
            np.array(true_bboxes_list),
            np.array(true_classes_list),
            save_dir=str(output_dir / 'predictions'),
            num_samples=args.num_visualizations
        )
        
        # Plot confusion matrix
        cm_path = output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(results['confusion_matrix'], save_path=str(cm_path))
        
        print(f"   OK - Visualizations saved to {output_dir}")
    
    # Save results
    results_path = Path(args.output_dir) / 'evaluation_results.txt'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("Face Mask Detection - Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"IoU Threshold: {args.iou_threshold}\n\n")
        f.write(f"mAP@IoU={args.iou_threshold}: {results['mAP']:.4f}\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n\n")
        f.write("Per-class Metrics:\n")
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(f"\n  {class_name}:\n")
            if i < len(results['precision']):
                f.write(f"    Precision: {results['precision'][i]:.4f}\n")
                f.write(f"    Recall: {results['recall'][i]:.4f}\n")
                f.write(f"    F1-Score: {results['f1_score'][i]:.4f}\n")
                if i < len(results['per_class_ap']):
                    f.write(f"    AP: {results['per_class_ap'][i]:.4f}\n")
    
    print(f"\n💾 Results saved to {results_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
