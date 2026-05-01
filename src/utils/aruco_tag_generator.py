import cv2
import cv2.aruco as aruco
import os

output_dir = r"./data/aruco_markers"
os.makedirs(output_dir, exist_ok=True)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker_size = 400 

for marker_id in range(4):
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    filename = os.path.join(output_dir, f"marker_4x4_id{marker_id}.png")
    cv2.imwrite(filename, marker_image)
    print(f" Created: {filename}")

print("\n Created 4 ArUco markers with IDs 0 to 3 in the 'aruco_markers' directory.")