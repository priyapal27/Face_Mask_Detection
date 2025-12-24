"""
Quick script to verify dataset structure
"""

from pathlib import Path

def verify_dataset():
    """Verify that the dataset is properly organized."""
    print("=" * 60)
    print("Dataset Structure Verification")
    print("=" * 60)
    
    # Check directories
    data_raw = Path("data/raw")
    images_dir = data_raw / "images"
    annotations_dir = data_raw / "annotations"
    
    print(f"\n📂 Checking directories...")
    
    if not data_raw.exists():
        print(f"❌ data/raw/ directory not found!")
        print(f"   Please create it first.")
        return False
    else:
        print(f"✅ data/raw/ exists")
    
    if not images_dir.exists():
        print(f"❌ data/raw/images/ directory not found!")
        print(f"   Please extract the dataset and place images here.")
        return False
    else:
        print(f"✅ data/raw/images/ exists")
    
    if not annotations_dir.exists():
        print(f"❌ data/raw/annotations/ directory not found!")
        print(f"   Please extract the dataset and place annotations here.")
        return False
    else:
        print(f"✅ data/raw/annotations/ exists")
    
    # Count files
    print(f"\n📊 Counting files...")
    
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    xml_files = list(annotations_dir.glob("*.xml"))
    
    print(f"   Images found: {len(image_files)}")
    print(f"   Annotations found: {len(xml_files)}")
    
    if len(image_files) == 0:
        print(f"\n❌ No image files found in data/raw/images/")
        print(f"   Expected: .png or .jpg files")
        return False
    
    if len(xml_files) == 0:
        print(f"\n❌ No XML files found in data/raw/annotations/")
        print(f"   Expected: .xml files")
        return False
    
    # Check if counts match
    if len(image_files) != len(xml_files):
        print(f"\n⚠️  Warning: Number of images ({len(image_files)}) doesn't match annotations ({len(xml_files)})")
    
    # Show sample files
    print(f"\n📄 Sample files:")
    print(f"   First image: {image_files[0].name if image_files else 'None'}")
    print(f"   First annotation: {xml_files[0].name if xml_files else 'None'}")
    
    # Expected count
    expected_count = 853
    if len(image_files) == expected_count and len(xml_files) == expected_count:
        print(f"\n✅ Perfect! Dataset is complete ({expected_count} images + {expected_count} annotations)")
    elif len(image_files) > 0 and len(xml_files) > 0:
        print(f"\n✅ Dataset found! You have {len(image_files)} images and {len(xml_files)} annotations")
        print(f"   (Expected: {expected_count} of each)")
    else:
        print(f"\n❌ Dataset incomplete or missing")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Dataset verification complete!")
    print("=" * 60)
    print("\nYou can now proceed with training:")
    print("  python train.py")
    
    return True


if __name__ == "__main__":
    verify_dataset()
