"""
Real-time Face Mask Detection using OpenCV
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import argparse
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import preprocess_for_inference
from src.evaluation.visualizer import CLASS_NAMES, CLASS_COLORS


# Color mapping for OpenCV (BGR format)
OPENCV_COLORS = {
    0: (0, 255, 0),      # Green - with_mask
    1: (0, 0, 255),      # Red - without_mask
    2: (0, 165, 255)     # Orange - mask_weared_incorrect
}


class MaskDetector:
    """Real-time mask detector using OpenCV."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained model
            confidence_threshold: Minimum confidence for detection
        """
        self.model = tf.keras.models.load_model(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"Model loaded from {model_path}")
    
    def preprocess_frame(self, frame: np.ndarray) -> tf.Tensor:
        """Preprocess video frame for model input."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize and resize
        frame_tensor = tf.constant(frame_rgb, dtype=tf.float32)
        frame_processed = preprocess_for_inference(frame_tensor)
        frame_batch = tf.expand_dims(frame_processed, 0)
        
        return frame_batch
    
    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect mask in frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Detection results
        """
        # Preprocess
        frame_batch = self.preprocess_frame(frame)
        
        # Predict
        predictions = self.model(frame_batch, training=False)
        
        # Extract results
        bbox = predictions['bbox'][0].numpy()
        class_probs = predictions['class'][0].numpy()
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]
        
        return {
            'bbox': bbox,
            'class_id': int(class_id),
            'class_name': CLASS_NAMES[class_id],
            'confidence': float(confidence)
        }
    
    def draw_detection(self, frame: np.ndarray, detection: dict) -> np.ndarray:
        """
        Draw detection on frame.
        
        Args:
            frame: Input frame
            detection: Detection results
            
        Returns:
            Frame with detection drawn
        """
        if detection['confidence'] < self.confidence_threshold:
            return frame
        
        h, w = frame.shape[:2]
        bbox = detection['bbox']
        
        # Convert normalized bbox to pixel coordinates
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)
        
        # Get color
        color = OPENCV_COLORS[detection['class_id']]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label
        label = f"{detection['class_name'].replace('_', ' ')}: {detection['confidence']:.2f}"
        
        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return frame
    
    def run_webcam(self, camera_id: int = 0):
        """
        Run real-time detection on webcam.
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("🎥 Starting webcam detection...")
        print("Press 'q' to quit")
        
        # FPS calculation
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Detect
            detection = self.detect(frame)
            
            # Draw detection
            frame = self.draw_detection(frame, detection)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 10:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_counter = 0
            
            # Draw FPS
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display
            cv2.imshow('Face Mask Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Detection stopped")
    
    def process_video(self, input_path: str, output_path: str):
        """
        Process a video file.
        
        Args:
            input_path: Input video path
            output_path: Output video path
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"📹 Processing video: {input_path}")
        print(f"   Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            detection = self.detect(frame)
            
            # Draw detection
            frame = self.draw_detection(frame, detection)
            
            # Write frame
            out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"   Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)", end='\r')
        
        cap.release()
        out.release()
        
        print(f"\n✅ Video saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time Face Mask Detection')
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/best_model.keras',
        help='Path to trained model'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='webcam',
        help='Source: "webcam" or path to video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.mp4',
        help='Output video path (for video file processing)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Camera device ID'
    )
    
    args = parser.parse_args()
    
    # Create detector
    detector = MaskDetector(args.model, args.confidence)
    
    # Run detection
    if args.source == 'webcam':
        detector.run_webcam(args.camera_id)
    else:
        detector.process_video(args.source, args.output)


if __name__ == "__main__":
    main()
