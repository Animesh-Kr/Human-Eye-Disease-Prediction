import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
import os

# 1. Load the FP32 Edge Model
model_path = "human_eye_fp32.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def preprocess_image(image_path):
    # 1. Read image (OpenCV loads as BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Error: Could not find or open {image_path}")
        sys.exit(1)
        
    # 2. Convert BGR to RGB (Crucial for your model!)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Resize to match EfficientNetV2L input
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # 4. ImageNet Normalization (Matches your Albumentations setup)
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std
    
    # 5. Add batch dimension -> Shape: (1, 224, 224, 3)
    return np.expand_dims(img_normalized, axis=0)

def predict_eye_disease(image_path):
    img_data = preprocess_image(image_path)

    # Run Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_data})
    
    # Post-process
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    prediction = classes[np.argmax(outputs[0])]
    confidence = np.max(outputs[0])
    
    return prediction, confidence, img_data

# --- COMMAND LINE INTERFACE ---
if __name__ == "__main__":
    # Check if user provided an image file in the terminal
    if len(sys.argv) < 2:
        print("⚠️ Usage: python run_inference.py <image_file_name>")
        print("Example: python run_inference.py scan_001.jpg")
        sys.exit(1)
        
    image_file = sys.argv[1]
    
    # 1. Verify it works
    print(f"\nAnalyzing: {image_file}...")
    result, conf, img_data = predict_eye_disease(image_file)
    print(f"✅ Prediction: {result} ({conf*100:.2f}%)\n")

    # 2. Warmup
    print("Running warmup...")
    input_name = session.get_inputs()[0].name
    for _ in range(10):
        session.run(None, {input_name: img_data})

    # 3. Official Benchmark
    print("Running official benchmark (50 passes)...")
    start_time = time.perf_counter()

    for _ in range(50):
        session.run(None, {input_name: img_data})

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time_ms = (total_time / 50) * 1000

    print("-" * 40)
    print(f"📊 Edge Latency Results:")
    print(f"Total Time (50 images): {total_time:.2f} seconds")
    print(f"Average Time per Image: {avg_time_ms:.2f} ms")
    print("-" * 40)