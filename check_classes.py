from ultralytics import YOLO

model = YOLO("artifacts/best.pt")
print("Class Names:", model.names)
