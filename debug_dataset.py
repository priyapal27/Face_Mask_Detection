"""
Debug script to check dataset shapes
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.data.dataset_loader import VOCAnnotationParser, create_dataset
from src.training.trainer import split_dataset

print("Loading dataset...")
parser = VOCAnnotationParser(
    images_dir="data/raw/images",
    annotations_dir="data/raw/annotations"
)
annotations = parser.load_dataset()

print(f"\nSplitting dataset...")
train_annotations, val_annotations, test_annotations = split_dataset(
    annotations,
    train_ratio=0.7,
    val_ratio=0.15
)

print(f"Creating dataset...")
train_dataset = create_dataset(
    train_annotations,
    Path("data/raw/images"),
    batch_size=16,
    shuffle=False,
    target_size=(224, 224)
)

print(f"\nChecking first batch...")
for images, labels in train_dataset.take(1):
    print(f"Images shape: {images.shape}")
    print(f"Bbox shape: {labels['bbox'].shape}")
    print(f"Class shape: {labels['class'].shape}")
    print(f"\nBbox dtype: {labels['bbox'].dtype}")
    print(f"Class dtype: {labels['class'].dtype}")
    print(f"\nFirst bbox: {labels['bbox'][0]}")
    print(f"First class: {labels['class'][0]}")
    
print("\nDataset check complete!")
