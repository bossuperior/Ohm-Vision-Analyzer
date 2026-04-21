import cv2
import numpy as np
import math

class ResistorCrop:
    #Data structure to hold cropped resistor image and related info
    def __init__(self, res_id, image_crop, text_position, keypoints):
        self.id = res_id
        self.image_crop = image_crop        # Resistor body image cropped and rotated to be horizontal
        self.text_position = text_position  # Position for drawing text on the main image
        self.keypoints = keypoints          # Original keypoints for circuit analysis

class BodyDetector:
    def __init__(self, resistor_class_id=1):
        self.resistor_class_id = resistor_class_id

    def extract_resistors(self, image, detection_results):
        #Get warped board image and the detection results from the model engine, then extract and align resistor bodies
        extracted_list = []
        
        # if no objects detected at all, return empty list immediately
        if len(detection_results.class_ids) == 0:
            return extracted_list

        # Find indices of detected resistors based on class IDs
        resistor_indices = np.where(detection_results.class_ids == self.resistor_class_id)[0]

        for idx, res_idx in enumerate(resistor_indices):
            box = detection_results.boxes[res_idx]
            kpts = detection_results.keypoints[res_idx]
            
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Position for drawing text on the main image
            text_pos = (x1, max(10, y1 - 10))

            # Check if we have at least 2 keypoints to calculate the angle for rotation
            if len(kpts) >= 2:
                kp1_x, kp1_y = kpts[0][:2]
                kp2_x, kp2_y = kpts[1][:2]

                # Calculate the angle of the resistor body based on the first two keypoints
                dx = kp2_x - kp1_x
                dy = kp2_y - kp1_y
                angle = math.degrees(math.atan2(dy, dx))

                # Find the center point of the bounding box to use as the rotation center
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Crop a larger region around the resistor to ensure we capture the whole body even if it's rotated
                pad = max(x2 - x1, y2 - y1) // 2 + 10
                H, W = image.shape[:2]
                
                px1 = max(0, cx - pad)
                py1 = max(0, cy - pad)
                px2 = min(W, cx + pad)
                py2 = min(H, cy + pad)
                
                roi = image[py1:py2, px1:px2]
                if roi.size == 0:
                    continue

                # New center of the ROI for rotation
                roi_cx = cx - px1
                roi_cy = cy - py1

                # Rotate the ROI to align the resistor horizontally
                matrix = cv2.getRotationMatrix2D((roi_cx, roi_cy), angle, 1.0)
                rotated_roi = cv2.warpAffine(roi, matrix, (roi.shape[1], roi.shape[0]))

                # Crop only the resistor body from the rotated ROI using the original bounding box dimensions
                body_length = int(math.hypot(dx, dy)) + 15
                # Add some padding to the thickness to ensure we capture the whole body even if the keypoints are not perfectly on the edges
                body_thickness = min(x2 - x1, y2 - y1) + 10

                rx1 = max(0, roi_cx - body_length // 2)
                ry1 = max(0, roi_cy - body_thickness // 2)
                rx2 = min(rotated_roi.shape[1], roi_cx + body_length // 2)
                ry2 = min(rotated_roi.shape[0], roi_cy + body_thickness // 2)

                final_crop = rotated_roi[ry1:ry2, rx1:rx2]

                if final_crop.size > 0:
                    extracted_list.append(ResistorCrop(idx, final_crop, text_pos, kpts))

            else:
                # Fallback: If we don't have enough keypoints to determine the angle, just crop the bounding box area without rotation
                final_crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
                if final_crop.size > 0:
                    extracted_list.append(ResistorCrop(idx, final_crop, text_pos, kpts))

        return extracted_list