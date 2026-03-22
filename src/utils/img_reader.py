import cv2
import numpy as np

def get_resistor_img(img, box):
    """ตัดภาพและหมุนให้เป็นแนวนอน"""
    rect = cv2.boxPoints(box)
    rect = np.intp(rect)
    s = rect.sum(axis=1)
    tl = rect[np.argmin(s)]
    br = rect[np.argmax(s)]
    diff = np.diff(rect, axis=1)
    tr = rect[np.argmin(diff)]
    bl = rect[np.argmax(diff)]
    wA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    wB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(wA), int(wB))
    hA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    hB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(hA), int(hB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    pts = np.array([tl, tr, br, bl], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped
