
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from src.models.mask_detector import create_mask_detector

model_path = 'models/checkpoints/best_model.h5'
print(f"Attempting to rebuild architecture and load weights from {model_path}")

try:
    # Build architecture
    model = create_mask_detector(
        input_shape=(224, 224, 3),
        num_classes=3,
        backbone='mobilenet_v2',
        trainable_backbone=False
    )
    
    # Try to load weights
    model.load_weights(model_path)
    print("Success loading weights!")
    
    # Save as .keras
    new_path = 'models/checkpoints/best_model.keras'
    model.save(new_path)
    print(f"Saved conversion to {new_path}")
    
except Exception as e:
    print("Failed to load weights:")
    import traceback
    traceback.print_exc()
