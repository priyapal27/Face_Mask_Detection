"""
TensorFlow Lite Converter for Model Optimization
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import time


def convert_to_tflite(
    model_path: str,
    output_path: str,
    quantization: str = 'dynamic'
) -> None:
    """
    Convert TensorFlow model to TFLite format.
    
    Args:
        model_path: Path to saved model (.h5 or SavedModel)
        output_path: Output path for TFLite model
        quantization: Quantization type ('none', 'dynamic', 'int8', 'float16')
    """
    print(f"Converting model: {model_path}")
    print(f"Quantization: {quantization}")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization
    if quantization == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  Applying dynamic range quantization...")
    elif quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        print("  Applying int8 quantization...")
    elif quantization == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("  Applying float16 quantization...")
    else:
        print("  No quantization applied")
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    original_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
    tflite_size = len(tflite_model) / (1024 * 1024)  # MB
    compression_ratio = (1 - tflite_size / original_size) * 100
    
    print(f"\n✅ Conversion complete!")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  TFLite size: {tflite_size:.2f} MB")
    print(f"  Compression: {compression_ratio:.1f}%")
    print(f"  Saved to: {output_path}")


def benchmark_tflite_model(tflite_path: str, num_runs: int = 100) -> dict:
    """
    Benchmark TFLite model inference speed.
    
    Args:
        tflite_path: Path to TFLite model
        num_runs: Number of inference runs
        
    Returns:
        Benchmark results
    """
    print(f"\nBenchmarking TFLite model: {tflite_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    print(f"  Input shape: {input_shape}")
    
    # Create dummy input
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / avg_time
    
    print(f"\n📊 Benchmark Results ({num_runs} runs):")
    print(f"  Average inference time: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"  Min: {min_time:.2f} ms, Max: {max_time:.2f} ms")
    print(f"  Estimated FPS: {fps:.1f}")
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps
    }


def compare_models(
    original_model_path: str,
    tflite_model_path: str,
    test_images: np.ndarray = None
) -> None:
    """
    Compare original and TFLite model outputs.
    
    Args:
        original_model_path: Path to original model
        tflite_model_path: Path to TFLite model
        test_images: Test images for comparison
    """
    print("\n🔍 Comparing model outputs...")
    
    # Load original model
    original_model = tf.keras.models.load_model(original_model_path)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create test image if not provided
    if test_images is None:
        test_images = np.random.random((10, 224, 224, 3)).astype(np.float32)
    
    bbox_diffs = []
    class_diffs = []
    
    for img in test_images:
        img_batch = np.expand_dims(img, 0)
        
        # Original model prediction
        orig_pred = original_model(img_batch, training=False)
        orig_bbox = orig_pred['bbox'].numpy()
        orig_class = orig_pred['class'].numpy()
        
        # TFLite model prediction
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        
        # Get outputs (order may vary)
        tflite_outputs = {}
        for output_detail in output_details:
            output_data = interpreter.get_tensor(output_detail['index'])
            if output_data.shape[-1] == 4:
                tflite_outputs['bbox'] = output_data
            elif output_data.shape[-1] == 3:
                tflite_outputs['class'] = output_data
        
        # Calculate differences
        bbox_diff = np.mean(np.abs(orig_bbox - tflite_outputs['bbox']))
        class_diff = np.mean(np.abs(orig_class - tflite_outputs['class']))
        
        bbox_diffs.append(bbox_diff)
        class_diffs.append(class_diff)
    
    print(f"\n📈 Output Comparison:")
    print(f"  Bbox MAE: {np.mean(bbox_diffs):.6f} (±{np.std(bbox_diffs):.6f})")
    print(f"  Class MAE: {np.mean(class_diffs):.6f} (±{np.std(class_diffs):.6f})")
    
    if np.mean(bbox_diffs) < 0.01 and np.mean(class_diffs) < 0.01:
        print("  ✅ Models produce very similar outputs!")
    else:
        print("  ⚠️  Significant differences detected. Consider using less aggressive quantization.")


def main():
    """Main conversion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert model to TFLite')
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/best_model.h5',
        help='Path to original model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/tflite',
        help='Output directory for TFLite models'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['none', 'dynamic', 'int8', 'float16'],
        default='dynamic',
        help='Quantization type'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark after conversion'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare original and TFLite outputs'
    )
    
    args = parser.parse_args()
    
    # Convert model
    output_path = f"{args.output_dir}/model_{args.quantization}.tflite"
    convert_to_tflite(args.model, output_path, args.quantization)
    
    # Benchmark
    if args.benchmark:
        benchmark_tflite_model(output_path)
    
    # Compare
    if args.compare:
        compare_models(args.model, output_path)


if __name__ == "__main__":
    main()
