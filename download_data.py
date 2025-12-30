import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import torch
from roboflow import Roboflow

# Fix for PyTorch 2.6+ weights_only=True default
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except (ImportError, AttributeError):
    pass # Fallback for older versions

def download_dataset():
    rf = Roboflow(api_key="quVTJAj1uWTBCmNZpJyS")
    project = rf.workspace("new-workspace-2cnfr").project("mask-ecop7")
    version = project.version(2)
    dataset = version.download("yolov8")
    return dataset.location

if __name__ == "__main__":
    print("Downloading dataset...")
    location = download_dataset()
    print(f"Dataset downloaded to: {location}")
