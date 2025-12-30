import argparse
import os
from ultralytics import YOLO

def run_inference(source, weights, conf):
    """
    Runs inference on an image and saves the result.
    """
    if not os.path.exists(weights):
        print(f"Error: Weights file not found at {weights}")
        return

    model = YOLO(weights)
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    print(f"Running inference on {source}...")
    results = model.predict(source, conf=conf, save=True, project="outputs", name="predict", exist_ok=True)
    
    print(f"Inference complete. Results saved to outputs/predict/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face Mask Detection Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="artifacts/best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    run_inference(args.source, args.weights, args.conf)
