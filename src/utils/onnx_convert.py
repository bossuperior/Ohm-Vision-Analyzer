from pathlib import Path
from ultralytics import YOLO

current_path= Path(__file__).resolve()

# 2. Go up the folder tree to find the root 'Ohm-Vision-Analyzer' folder
# .parent goes up one level: src/utils -> src -> Ohm-Vision-Analyzer
project_root = current_path.parents[2]

# 3. Build the absolute path to the weights file
weights_path = project_root / "models" / "Yolo_v8n_pose_weights.pt"

print(f" Attempting to load model from: {weights_path}")

# 4. Check if it actually exists before YOLO crashes
if not weights_path.exists():
    raise FileNotFoundError(f" Could not find the weights file at {weights_path}. Did you misspell it?")

# 5. Load the model
model = YOLO(str(weights_path))

print("Exporting model to ONNX format...")
exported_path = model.export(format="onnx", half=True)

print(f"Export complete! Your ONNX model is located at: {exported_path}")