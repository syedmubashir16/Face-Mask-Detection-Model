from ultralytics import YOLO
import sys

# Load model
try:
    model = YOLO("artifacts/best.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Image path
img_path = r"C:\Users\Texon\Downloads\ffc432ff205efdc5e89811d6ec9899ae.jpg"

print(f"Running inference on {img_path}...")

# Run inference with low threshold to see everything
results = model.predict(img_path, conf=0.1)

for result in results:
    boxes = result.boxes
    print(f"Total Detections: {len(boxes)}")
    if len(boxes) == 0:
        print("No detections found even at conf=0.1")
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        print(f"Detected: {cls_name} | Confidence: {conf:.4f}")
