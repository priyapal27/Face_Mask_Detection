"""
Dataset Loader for Face Mask Detection
Parses Pascal VOC XML annotations and creates TensorFlow datasets.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import tensorflow as tf


# Class mapping
CLASS_NAMES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


class VOCAnnotationParser:
    """Parser for Pascal VOC XML annotations."""
    
    def __init__(self, images_dir: str, annotations_dir: str):
        """
        Initialize the parser.
        
        Args:
            images_dir: Directory containing images
            annotations_dir: Directory containing XML annotations
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        
    def parse_xml(self, xml_path: Path) -> Optional[Dict]:
        """
        Parse a single XML annotation file.
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            Dictionary with image info and annotations, or None if parsing fails
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image filename
            filename = root.find('filename').text
            
            # Get image size
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # Parse all objects in the image
            objects = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                # Skip if class not in our mapping
                if name not in CLASS_TO_IDX:
                    continue
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                objects.append({
                    'class': name,
                    'class_id': CLASS_TO_IDX[name],
                    'bbox': [xmin, ymin, xmax, ymax]
                })
            
            # Skip images with no valid objects
            if not objects:
                return None
            
            return {
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects
            }
            
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None
    
    def load_dataset(self) -> List[Dict]:
        """
        Load all annotations from the dataset.
        
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        xml_files = list(self.annotations_dir.glob('*.xml'))
        
        print(f"Found {len(xml_files)} annotation files")
        
        for xml_file in xml_files:
            annotation = self.parse_xml(xml_file)
            if annotation:
                annotations.append(annotation)
        
        print(f"Successfully parsed {len(annotations)} annotations")
        return annotations


def normalize_bbox(bbox: List[int], img_width: int, img_height: int) -> List[float]:
    """
    Normalize bounding box coordinates to [0, 1] range.
    
    Args:
        bbox: [xmin, ymin, xmax, ymax] in pixel coordinates
        img_width: Image width
        img_height: Image height
        
    Returns:
        Normalized bbox [xmin, ymin, xmax, ymax] in [0, 1]
    """
    xmin, ymin, xmax, ymax = bbox
    return [
        xmin / img_width,
        ymin / img_height,
        xmax / img_width,
        ymax / img_height
    ]


def create_tf_example(annotation: Dict, images_dir: Path, target_size: Tuple[int, int] = (224, 224)):
    """
    Create a TensorFlow example from annotation.
    
    Args:
        annotation: Annotation dictionary
        images_dir: Directory containing images
        target_size: Target image size (height, width)
        
    Returns:
        Tuple of (image, bbox, class_id) or None if image not found
    """
    # Load image
    img_path = images_dir / annotation['filename']
    if not img_path.exists():
        return None
    
    # Read and decode image
    img = tf.io.read_file(str(img_path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
    
    # Resize image
    img = tf.image.resize(img, target_size)
    
    # For simplicity, use the first object (can be extended for multi-object detection)
    obj = annotation['objects'][0]
    
    # Normalize bounding box
    bbox = normalize_bbox(obj['bbox'], annotation['width'], annotation['height'])
    
    # Class ID
    class_id = obj['class_id']
    
    return img, np.array(bbox, dtype=np.float32), class_id


def create_dataset(
    annotations: List[Dict],
    images_dir: Path,
    batch_size: int = 16,
    shuffle: bool = True,
    target_size: Tuple[int, int] = (224, 224)
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from annotations.
    
    Args:
        annotations: List of annotation dictionaries
        images_dir: Directory containing images
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        target_size: Target image size
        
    Returns:
        TensorFlow dataset
    """
    images = []
    bboxes = []
    class_ids = []
    
    for annotation in annotations:
        result = create_tf_example(annotation, images_dir, target_size)
        if result:
            img, bbox, class_id = result
            images.append(img.numpy())
            bboxes.append(bbox)
            class_ids.append(class_id)
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    bboxes = np.array(bboxes, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int32)
    
    # One-hot encode class labels
    num_classes = 3
    class_one_hot = np.eye(num_classes)[class_ids]
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, {'bbox': bboxes, 'class': class_one_hot}))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    # Test the dataset loader
    parser = VOCAnnotationParser(
        images_dir="data/raw/images",
        annotations_dir="data/raw/annotations"
    )
    
    annotations = parser.load_dataset()
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(annotations)}")
    
    # Count classes
    class_counts = {name: 0 for name in CLASS_NAMES}
    for ann in annotations:
        for obj in ann['objects']:
            class_counts[obj['class']] += 1
    
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
