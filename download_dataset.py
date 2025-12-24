"""
Dataset Download Script for Face Mask Detection
Downloads the dataset from Kaggle and organizes it into the data/raw directory.
"""

import os
import sys
import zipfile
from pathlib import Path


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print("❌ Kaggle API credentials not found!")
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. This will download kaggle.json")
        print(f"4. Move kaggle.json to: {kaggle_json.parent}")
        print("\nOn Windows, run:")
        print(f"   mkdir {kaggle_json.parent}")
        print(f"   move kaggle.json {kaggle_json}")
        return False
    return True


def download_dataset():
    """Download the Face Mask Detection dataset from Kaggle."""
    try:
        import kaggle
        
        print("📦 Downloading Face Mask Detection dataset from Kaggle...")
        
        # Dataset identifier
        dataset = "andrewmvd/face-mask-detection"
        
        # Download path
        download_path = Path("data/raw")
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset,
            path=str(download_path),
            unzip=True,
            quiet=False
        )
        
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")
        print("Extract to: data/raw/")
        return False


def verify_dataset():
    """Verify the dataset structure and count files."""
    data_path = Path("data/raw")
    
    # Check for images and annotations
    images_path = data_path / "images"
    annotations_path = data_path / "annotations"
    
    if not images_path.exists() or not annotations_path.exists():
        print("⚠️  Dataset structure not as expected.")
        print("Looking for alternative structure...")
        
        # Check if files are directly in data/raw
        image_files = list(data_path.glob("*.png")) + list(data_path.glob("*.jpg"))
        xml_files = list(data_path.glob("*.xml"))
        
        if image_files and xml_files:
            print(f"✅ Found {len(image_files)} images and {len(xml_files)} annotations")
            return True
    else:
        image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
        xml_files = list(annotations_path.glob("*.xml"))
        print(f"✅ Found {len(image_files)} images and {len(xml_files)} annotations")
        return True
    
    print("❌ Dataset verification failed!")
    return False


def main():
    """Main function to download and verify dataset."""
    print("=" * 60)
    print("Face Mask Detection Dataset Download")
    print("=" * 60)
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        print("\n⚠️  Please set up Kaggle credentials and run again.")
        sys.exit(1)
    
    # Download dataset
    if download_dataset():
        print("\n" + "=" * 60)
        print("Verifying Dataset")
        print("=" * 60)
        
        if verify_dataset():
            print("\n✅ Dataset is ready for use!")
            print("\nNext steps:")
            print("1. Run data exploration: jupyter notebook notebooks/01_data_exploration.ipynb")
            print("2. Or start training: python src/training/trainer.py")
        else:
            print("\n⚠️  Please check the dataset structure.")
    else:
        print("\n⚠️  Dataset download failed. Please download manually.")


if __name__ == "__main__":
    main()
