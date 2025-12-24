"""
Streamlit Web Application for Face Mask Detection
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import preprocess_for_inference
from src.evaluation.visualizer import visualize_prediction, CLASS_NAMES, CLASS_COLORS


# Page configuration
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide"
)


@st.cache_resource
def load_model(model_path: str):
    """Load the trained model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict_image(model, image: np.ndarray, confidence_threshold: float = 0.5):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained model
        image: Input image
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Prediction dictionary
    """
    # Preprocess image
    processed_image = preprocess_for_inference(tf.constant(image))
    processed_image = tf.expand_dims(processed_image, 0)
    
    # Make prediction
    predictions = model(processed_image, training=False)
    
    # Extract results
    bbox = predictions['bbox'][0].numpy()
    class_probs = predictions['class'][0].numpy()
    class_id = np.argmax(class_probs)
    confidence = class_probs[class_id]
    
    return {
        'bbox': bbox,
        'class_id': class_id,
        'class_name': CLASS_NAMES[class_id],
        'confidence': confidence,
        'all_probs': class_probs
    }


def main():
    """Main Streamlit app."""
    
    # Title and description
    st.title("😷 Face Mask Detection System")
    st.markdown("""
    Upload an image to detect face masks and classify them as:
    - 🟢 **With Mask** (properly worn)
    - 🔴 **Without Mask**
    - 🟠 **Mask Worn Incorrectly**
    """)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        "models/checkpoints/best_model.keras"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Load model
    model = load_model(model_path)
    
    if model is None:
        st.warning("⚠️ Please train the model first or provide a valid model path.")
        st.info("To train the model, run: `python src/training/trainer.py`")
        return
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Main content
    tab1, tab2 = st.tabs(["📷 Image Upload", "📊 About"])
    
    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing a face"
        )
        
        if uploaded_file is not None:
            # Load image and ensure RGB
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                try:
                    # Double check shape before prediction
                    st.sidebar.write(f"Processed Image Shape: {image_np.shape}")
                    prediction = predict_image(model, image_np, confidence_threshold)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.exception(e)
                    return
            
            # Display results
            with col2:
                st.subheader("Detection Results")
                
                # Show prediction
                if prediction['confidence'] >= confidence_threshold:
                    # Determine emoji based on class
                    emoji_map = {
                        'with_mask': '🟢',
                        'without_mask': '🔴',
                        'mask_weared_incorrect': '🟠'
                    }
                    emoji = emoji_map.get(prediction['class_name'], '⚪')
                    
                    st.markdown(f"### {emoji} {prediction['class_name'].replace('_', ' ').title()}")
                    st.metric("Confidence", f"{prediction['confidence']:.2%}")
                    
                    # Show all class probabilities
                    st.subheader("Class Probabilities")
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob = prediction['all_probs'][i]
                        st.progress(float(prob), text=f"{class_name.replace('_', ' ').title()}: {prob:.2%}")
                    
                    # Visualize bbox
                    st.subheader("Bounding Box Visualization")
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    ax.imshow(image_np)
                    
                    # Draw bbox
                    bbox = prediction['bbox']
                    img_height, img_width = image_np.shape[:2]
                    x1, y1, x2, y2 = bbox * np.array([img_width, img_height, img_width, img_height])
                    
                    import matplotlib.patches as patches
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=3,
                        edgecolor=CLASS_COLORS[prediction['class_id']],
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.axis('off')
                    st.pyplot(fig)
                    
                else:
                    st.warning(f"⚠️ Confidence ({prediction['confidence']:.2%}) below threshold ({confidence_threshold:.0%})")
                    st.info("Try adjusting the confidence threshold in the sidebar.")
        
        else:
            st.info("👆 Upload an image to get started!")
    
    with tab2:
        st.subheader("About This Project")
        st.markdown("""
        ### Face Mask Detection System
        
        This system uses a deep learning model based on **MobileNetV2** to detect and classify face masks in images.
        
        **Model Architecture:**
        - Backbone: MobileNetV2 (pre-trained on ImageNet)
        - Dual-head design:
          - Bounding box regression (4 outputs)
          - Classification (3 classes)
        
        **Classes:**
        1. **With Mask**: Face mask properly worn
        2. **Without Mask**: No face mask detected
        3. **Mask Worn Incorrectly**: Face mask present but not properly worn
        
        **Performance Metrics:**
        - Target mAP@IoU=0.5: > 0.70
        - Target Precision/Recall: > 0.65 per class
        
        **Dataset:**
        - Source: Kaggle Face Mask Detection Dataset
        - Format: Pascal VOC XML annotations
        - Total images: 853
        """)


if __name__ == "__main__":
    main()
