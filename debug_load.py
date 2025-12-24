
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import traceback

model_path = 'models/checkpoints/best_model.h5'
print(f"Attempting to load model from {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    print("Success!")
    model.summary()
except Exception as e:
    print("Failed to load model:")
    traceback.print_exc()
