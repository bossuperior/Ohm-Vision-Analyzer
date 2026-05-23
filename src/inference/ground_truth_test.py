import os
import cv2
from torchgen import model
from inference import model_engine
from pyexpat import model

test_dir = "/data/labels/test"
img_paths = sorted(test_dir.glob('*.jpg')) + sorted(test_dir.glob('*.png'))

print(f"Found {len(img_paths)} images")

for i, img_path in enumerate(img_paths):
    result = model_engine.infer(str(img_path), conf=0.5)
    results = model(str(img_path), conf=0.5)
    annotated = results[0].plot()

    h, w = annotated.shape[:2]
    if w > 960:
        annotated = cv2.resize(annotated, (960, int(h * 960 / w)))

    print(f"[{i+1}/{len(img_paths)}] {img_path.name}")
    cv2.imshow("Annotated Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
path = "/data/test"
ground_truth = {
    os.path.join(path, "test_img_001.jpg"): {"type": "One Resistor", "total_ohm": 1000},
    os.path.join(path, "test_img_002.jpg"): {"type": "Parallel", "total_ohm": 500},
    os.path.join(path, "test_img_003.jpg"): {"type": "Wheat Circuit", "total_ohm": 333},
}

correct_circuit_type = 0
correct_ohm_value = 0
total_images = len(ground_truth)

def my_circuit_pipeline(image_path):
    # predict_type, predict_ohm = your_main_function(image_path)
    return "Parallel", 500  # ตัวอย่างผลจำลอง

print("--- Start Testing ---")
for img_name, target in ground_truth.items():
    img_path = os.path.join("dataset/test", img_name)
    
    # รันผ่านระบบ Pipeline
    pred_type, pred_ohm = my_circuit_pipeline(img_path)
    
    # ตรวจคำตอบประเภทวงจร
    if pred_type == target["type"]:
        correct_circuit_type += 1
    
    # ตรวจคำตอบค่าโอห์ม (ยอมให้มี Error Tolerance ได้เล็กน้อย เช่น ±5%)
    error_margin = abs(pred_ohm - target["total_ohm"]) / target["total_ohm"]
    if error_margin <= 0.05: 
        correct_ohm_value += 1

type_accuracy = (correct_circuit_type / total_images) * 100
ohm_accuracy = (correct_ohm_value / total_images) * 100

print("\n--- Test Results ---")
print(f"จำนวนภาพที่ทดสอบทั้งหมด: {total_images} ภาพ")
print(f"ความแม่นยำการจำแนกประเภทวงจร (Circuit Type Accuracy): {type_accuracy:.2f}%")
print(f"ความแม่นยำการคำนวณค่าทางไฟฟ้า (Ohmic Value Accuracy): {ohm_accuracy:.2f}%")