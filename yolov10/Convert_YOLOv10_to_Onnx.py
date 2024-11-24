from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train2/weights/best.pt')  # load a custom trained model

# Export the model to ONNX with specific settings
model.export(format="onnx", imgsz=(1024, 1024), opset=17, simplify=True)
