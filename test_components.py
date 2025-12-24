"""
Simple test script to verify dataset loading and model creation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

print("=" * 60)
print("Testing Face Mask Detection Components")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from src.data.dataset_loader import VOCAnnotationParser
    from src.models.mask_detector import create_mask_detector
    print("   OK - All modules imported successfully")
except Exception as e:
    print(f"   ERROR - Import failed: {e}")
    sys.exit(1)

# Test 2: Load annotations
print("\n2. Testing annotation loading...")
try:
    parser = VOCAnnotationParser(
        images_dir="data/raw/images",
        annotations_dir="data/raw/annotations"
    )
    annotations = parser.load_dataset()
    print(f"   OK - Loaded {len(annotations)} annotations")
except Exception as e:
    print(f"   ERROR - Annotation loading failed: {e}")
    sys.exit(1)

# Test 3: Split dataset
print("\n3. Testing dataset split...")
try:
    np.random.shuffle(annotations)
    n = len(annotations)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_annotations = annotations[:train_end]
    val_annotations = annotations[train_end:val_end]
    test_annotations = annotations[val_end:]
    
    print(f"   OK - Train: {len(train_annotations)}, Val: {len(val_annotations)}, Test: {len(test_annotations)}")
except Exception as e:
    print(f"   ERROR - Dataset split failed: {e}")
    sys.exit(1)

# Test 4: Create model
print("\n4. Testing model creation...")
try:
    import tensorflow as tf
    model = create_mask_detector(
        input_shape=(224, 224, 3),
        num_classes=3,
        backbone='mobilenet_v2',
        trainable_backbone=False
    )
    print(f"   OK - Model created with {len(model.trainable_variables)} trainable variables")
except Exception as e:
    print(f"   ERROR - Model creation failed: {e}")
    sys.exit(1)

# Test 5: Create small dataset
print("\n5. Testing TensorFlow dataset creation...")
try:
    from src.data.dataset_loader import create_dataset
    
    # Use only first 10 samples for testing
    small_train = train_annotations[:10]
    
    train_dataset = create_dataset(
        small_train,
        Path("data/raw/images"),
        batch_size=2,
        shuffle=False,
        target_size=(224, 224)
    )
    
    # Try to get one batch
    for images, labels in train_dataset.take(1):
        print(f"   OK - Dataset created. Batch shape: images={images.shape}, bbox={labels['bbox'].shape}, class={labels['class'].shape}")
        break
except Exception as e:
    print(f"   ERROR - Dataset creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test forward pass
print("\n6. Testing model forward pass...")
try:
    predictions = model(images, training=False)
    print(f"   OK - Forward pass successful. Bbox shape: {predictions['bbox'].shape}, Class shape: {predictions['class'].shape}")
except Exception as e:
    print(f"   ERROR - Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed successfully!")
print("=" * 60)
print("\nYou can now run full training with:")
print("  python train.py --epochs 50 --batch-size 16")
