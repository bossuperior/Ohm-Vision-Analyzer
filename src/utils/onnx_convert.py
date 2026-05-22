from pathlib import Path
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

current_path = Path(__file__).resolve()
project_root = current_path.parents[2]

weights_path = project_root / "models" / "Yolo_v8n_pose_weights_2.pt"

if not weights_path.exists():
    raise FileNotFoundError(f"Could not find weights at {weights_path}")

model = YOLO(str(weights_path))

print("Exporting to FP32 ONNX...")
onnx_path = Path(model.export(format="onnx", half=False))
print(f"FP32 ONNX: {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")

print("\nQuantizing to INT8...")
int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
quantize_dynamic(
    model_input=str(onnx_path),
    model_output=str(int8_path),
    weight_type=QuantType.QUInt8
)
print(f"INT8 ONNX: {int8_path}  ({int8_path.stat().st_size / 1e6:.1f} MB)")
print(f"Size reduction: {(1 - int8_path.stat().st_size / onnx_path.stat().st_size) * 100:.1f}%")
