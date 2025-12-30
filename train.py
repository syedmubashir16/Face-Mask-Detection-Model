import argparse
import os
import shutil
# Fix for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def train(dataset_config, epochs, imgsz, batch, device):
    """
    Trains the YOLOv8n model and exports artifacts.
    """
    print(f"Starting training with config: {dataset_config}")
    
    # ensure artifacts dir exists
    os.makedirs("artifacts", exist_ok=True)
    
    # Load model
    model = YOLO("yolov8s.pt")  # load larger model (Small) for better accuracy

    # Train
    results = model.train(
        data=dataset_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs",
        name="train",
        exist_ok=True  # overwrite existing run
    )

    # Validate
    # model.val() # optional, executed by train usually

    # Copy best model to artifacts
    best_model_path = os.path.join("runs", "train", "weights", "best.pt")
    target_path = os.path.join("artifacts", "best.pt")
    
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, target_path)
        print(f"Best model saved to {target_path}")
    else:
        print("Warning: best.pt not found in runs/")

    # Export to ONNX
    print("Exporting to ONNX...")
    model.export(format="onnx", imgsz=imgsz, half=False) # half=True often requires GPU/specific onnx runtime support, keeping safe with False or default
    
    # Move ONNX if it wasn't exported directly to artifacts (usually exports to same dir as weights or current dir)
    # Ultralytics export usually saves in the same dir as the loaded model or inside runs.
    # Since we re-loaded or continue from 'model', let's check where it put it. 
    # Actually, easiest is to use the exported file path which return from model.export()
    # But for simplicity, let's assume it might be in runs/train/weights/best.onnx or just move it manually if we find it.
    
    # Re-check standard export location
    exported_onnx = os.path.join("runs", "train", "weights", "best.onnx")
    target_onnx = os.path.join("artifacts", "best.onnx")
    if os.path.exists(exported_onnx):
        shutil.copy(exported_onnx, target_onnx)
        print(f"ONNX model saved to {target_onnx}")
    
    # Print file sizes
    for f in [target_path, target_onnx]:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"Artifact {f}: {size_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n for Face Mask Detection")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, mps, auto)")
    
    args = parser.parse_args()
    
    train(args.data, args.epochs, args.imgsz, args.batch, args.device)
