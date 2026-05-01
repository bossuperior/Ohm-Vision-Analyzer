from pathlib import Path
from ultralytics import YOLO

current_path= Path(__file__).resolve()
project_root = current_path.parents[2]

weights_path = project_root / "models" / "Yolo_v8n_pose_weights.pt"

print(f" Attempting to load model from: {weights_path}")

# Check if it actually exists before YOLO crashes
if not weights_path.exists():
    raise FileNotFoundError(f" Could not find the weights file at {weights_path}. Did you misspell it?")

# Load the model
model = YOLO(str(weights_path))

print("Exporting model to ONNX format...")
exported_path = model.export(format="onnx", half=True)

print(f"Export complete! Your ONNX model is located at: {exported_path}")