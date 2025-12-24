# 😷 Face Mask Detection System

A comprehensive deep learning system for detecting and classifying face masks in images using CNN-based object detection with Transfer Learning.

## 📋 Overview

This project implements a face mask detection system that can:
- **Detect** faces in images
- **Classify** mask usage into 3 categories:
  - 🟢 **With Mask** - Properly worn face mask
  - 🔴 **Without Mask** - No face mask detected  
  - 🟠 **Mask Worn Incorrectly** - Mask present but improperly worn

## 🏗️ Architecture

**Model**: Transfer Learning with MobileNetV2 backbone
- **Input**: 224×224×3 RGB images
- **Backbone**: MobileNetV2 (pre-trained on ImageNet)
- **Dual-Head Design**:
  - Bounding Box Regression Head (4 outputs)
  - Classification Head (3 classes, softmax)
- **Loss Function**: Combined MSE (bbox) + Categorical Cross-Entropy (class)

## 📁 Project Structure

```
c:\Priya_DL_Lab_Assessment_Exam\
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Processed data
│   └── splits/                 # Train/val/test splits
├── models/
│   ├── checkpoints/            # Training checkpoints
│   ├── saved_model/            # Final SavedModel
│   └── tflite/                 # Optimized TFLite models
├── src/
│   ├── data/                   # Data processing
│   │   ├── dataset_loader.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── models/                 # Model architecture
│   │   ├── mask_detector.py
│   │   └── losses.py
│   ├── training/               # Training pipeline
│   │   └── trainer.py
│   ├── evaluation/             # Evaluation & metrics
│   │   ├── metrics.py
│   │   └── visualizer.py
│   └── deployment/             # Deployment tools
│       ├── streamlit_app.py
│       ├── opencv_detector.py
│       └── tflite_converter.py
├── notebooks/                  # Jupyter notebooks
├── download_dataset.py         # Dataset download script
├── train.py                    # Main training script
├── evaluate.py                 # Main evaluation script
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd c:\Priya_DL_Lab_Assessment_Exam

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Set up Kaggle API credentials first
# Download kaggle.json from https://www.kaggle.com/account
# Place it in: C:\Users\Priya\.kaggle\kaggle.json

# Download dataset
python download_dataset.py
```

### 3. Train Model

```bash
# Train with default settings
python train.py

# Train with custom parameters
python train.py --epochs 50 --batch-size 16 --learning-rate 0.001
```

### 4. Evaluate Model

```bash
# Evaluate on test set
python evaluate.py --model models/checkpoints/best_model.h5
```

### 5. Deploy

#### Option A: Streamlit Web App
```bash
streamlit run src/deployment/streamlit_app.py
```

#### Option B: Real-time Webcam Detection
```bash
python src/deployment/opencv_detector.py --source webcam
```

#### Option C: Process Video File
```bash
python src/deployment/opencv_detector.py --source input.mp4 --output output.mp4
```

## 📊 Dataset

- **Source**: [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- **Format**: Pascal VOC XML annotations
- **Total Images**: 853
- **Classes**: 3 (with_mask, without_mask, mask_weared_incorrect)
- **Split**: 70% train, 15% validation, 15% test

## 🎯 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| mAP@IoU=0.5 | > 0.70 | Mean Average Precision at 50% IoU |
| Precision | > 0.65 | Per-class precision |
| Recall | > 0.65 | Per-class recall |
| FPS (CPU) | > 15 | Real-time inference speed |
| Model Size | < 10 MB | TFLite quantized model |

## 🔧 Training Configuration

```python
# Default hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)
BBOX_WEIGHT = 1.0
CLASS_WEIGHT = 1.0
```

**Data Augmentation**:
- Random horizontal flip
- Random brightness (±20%)
- Random contrast (0.8-1.2)
- Random saturation (0.8-1.2)
- Random hue (±5%)

**Training Features**:
- Early stopping (patience=10)
- Model checkpointing (save best)
- Learning rate reduction on plateau
- TensorBoard logging

## 📈 Evaluation Metrics

The system provides comprehensive evaluation:

1. **mAP (mean Average Precision)** at IoU=0.5
2. **Precision, Recall, F1-Score** per class
3. **Confusion Matrix**
4. **Bounding Box Visualization**
5. **Training History Plots**

## 🌐 Deployment Options

### 1. Streamlit Web Application
- Upload images for detection
- Adjustable confidence threshold
- Real-time visualization
- Class probability display

### 2. OpenCV Real-time Detection
- Webcam support
- Video file processing
- FPS counter
- Color-coded bounding boxes

### 3. TFLite Optimization
- Dynamic range quantization
- INT8 quantization
- Float16 quantization
- Model size reduction (>50%)
- Inference speed benchmarking

## 🔬 Model Optimization

Convert trained model to TFLite:

```bash
# Dynamic quantization (recommended)
python src/deployment/tflite_converter.py --model models/checkpoints/best_model.h5 --quantization dynamic

# INT8 quantization (smallest size)
python src/deployment/tflite_converter.py --model models/checkpoints/best_model.h5 --quantization int8 --benchmark

# Compare models
python src/deployment/tflite_converter.py --model models/checkpoints/best_model.h5 --quantization dynamic --compare
```

## 📝 Usage Examples

### Python API

```python
import tensorflow as tf
from src.data.preprocessing import preprocess_for_inference
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('models/checkpoints/best_model.h5')

# Load and preprocess image
image = Image.open('test_image.jpg')
image_array = np.array(image)
processed = preprocess_for_inference(tf.constant(image_array))
batch = tf.expand_dims(processed, 0)

# Predict
predictions = model(batch, training=False)
bbox = predictions['bbox'][0].numpy()
class_probs = predictions['class'][0].numpy()
class_id = np.argmax(class_probs)

print(f"Class: {['with_mask', 'without_mask', 'mask_weared_incorrect'][class_id]}")
print(f"Confidence: {class_probs[class_id]:.2%}")
```

## 🛠️ Development

### Running Tests
```bash
# Test data loader
python src/data/dataset_loader.py

# Test model architecture
python src/models/mask_detector.py

# Test loss functions
python src/models/losses.py
```

### Jupyter Notebooks
Explore the notebooks for interactive analysis:
- `01_data_exploration.ipynb` - Dataset analysis
- `02_model_training.ipynb` - Training experiments
- `03_evaluation.ipynb` - Model evaluation

## 📦 Requirements

- Python >= 3.8
- TensorFlow >= 2.13.0
- OpenCV >= 4.8.0
- Streamlit >= 1.28.0
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn >= 1.3.0

See `requirements.txt` for complete list.

## 🤝 Contributing

This is an assessment project. For educational purposes only.

## 📄 License

Educational project - Face Mask Detection System

## 🙏 Acknowledgments

- Dataset: [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- Pre-trained backbone: MobileNetV2 (ImageNet weights)
- Framework: TensorFlow/Keras

## 📞 Support

For issues or questions, please refer to the documentation in the `notebooks/` directory.

---

**Built with ❤️ for Deep Learning Lab Assessment**
