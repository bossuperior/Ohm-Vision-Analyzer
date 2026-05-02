import cv2

DICTS = {
    "DICT_4X4_50":    cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":   cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50":    cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":   cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50":    cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":   cv2.aruco.DICT_6X6_100,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

cap = cv2.VideoCapture(1)

print("กด Q เพื่อออก")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    found_any = False

    for name, dict_id in DICTS.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            id_list = ids.flatten().tolist()
            print(f"  [{name}] -> IDs: {id_list}")
            found_any = True
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            cv2.putText(display, f"{name}: {id_list}",
                        (10, 30 + list(DICTS).index(name) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if not found_any:
        cv2.putText(display, "ไม่พบ tag ใน dictionary ใดเลย",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("ArUco Dictionary Scanner", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
