import cv2
import os

SAVE_DIR = "../data/raw/dataset_capture"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
existing_files = os.listdir(SAVE_DIR)
count = len([f for f in existing_files if f.endswith('.jpg')])

print(f"📁 Save Location: {SAVE_DIR}")
print(f"🔢 Starting at image number: {count + 1}")
print("Press 'Spacebar' to capture | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        count += 1
        filename = os.path.join(SAVE_DIR, f"img_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Captured: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()