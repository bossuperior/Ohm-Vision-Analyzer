import cv2
import os
import numpy as np

img_dir = r'data/raw/dataset_capture'
files = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))
print('Total images: {}'.format(len(files)))

# Only flag genuinely blurry (camera out of focus), not "low texture" from breadboard
BLUR_THRESH = 35

# Wire/resistor colors: high saturation (not breadboard beige/white)
# Check if saturated/colorful objects are being cut at the edge
EDGE_THRESH = 20    # pixel band to inspect near edge
SAT_THRESH  = 70    # minimum saturation to be considered a colored object

issues_blur    = []
issues_clip    = []
issues_really_bad = []

for fname in files:
    path = os.path.join(img_dir, fname)
    img = cv2.imread(path)
    if img is None:
        continue

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Blur: only flag if genuinely soft (out-of-focus shot)
    if blur_score < BLUR_THRESH:
        issues_blur.append((fname, blur_score))
        if blur_score < 25:
            issues_really_bad.append((fname, 'VERY BLURRY', blur_score))

    # Object-at-edge: look for colored (non-beige) pixels in edge strips
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]  # saturation channel

    # colorful object mask (wire/resistor body, not the breadboard base)
    obj_mask = sat > SAT_THRESH

    # Exclude corners (ArUco tag area: ~50px corners) to avoid false positives
    obj_no_corners = obj_mask.copy()
    corner = 50
    obj_no_corners[:corner, :corner]   = 0
    obj_no_corners[:corner, w-corner:] = 0
    obj_no_corners[h-corner:, :corner] = 0
    obj_no_corners[h-corner:, w-corner:] = 0

    top_strip    = obj_no_corners[:EDGE_THRESH, corner:w-corner]
    bottom_strip = obj_no_corners[h-EDGE_THRESH:, corner:w-corner]
    left_strip   = obj_no_corners[corner:h-corner, :EDGE_THRESH]
    right_strip  = obj_no_corners[corner:h-corner, w-EDGE_THRESH:]

    MIN_PIXELS = 80  # minimum colored pixels to flag as clipped object
    clipped_sides = []
    if top_strip.sum() > MIN_PIXELS:    clipped_sides.append('top')
    if bottom_strip.sum() > MIN_PIXELS: clipped_sides.append('bottom')
    if left_strip.sum() > MIN_PIXELS:   clipped_sides.append('left')
    if right_strip.sum() > MIN_PIXELS:  clipped_sides.append('right')

    if clipped_sides:
        issues_clip.append((fname, clipped_sides, blur_score))

print()
print('=== BLURRY (out-of-focus, var<{}) ==='.format(BLUR_THRESH))
if issues_blur:
    for fname, score in issues_blur:
        print('  {} blur={:.0f}{}'.format(fname, score, '  *** VERY BAD' if score < 25 else ''))
else:
    print('  None')

print()
print('=== COLORED OBJECT AT FRAME EDGE (likely clipped) ===')
if issues_clip:
    for fname, sides, blur in issues_clip:
        print('  {}  clipped: {}  [blur={:.0f}]'.format(fname, ', '.join(sides), blur))
else:
    print('  None')

print()
print('=== SUMMARY ===')
print('  Total images:    {}'.format(len(files)))
print('  Blurry (<{}):    {}'.format(BLUR_THRESH, len(issues_blur)))
print('  Object clipped:  {}'.format(len(issues_clip)))
clean = len(files) - len(set([x[0] for x in issues_blur] + [x[0] for x in issues_clip]))
print('  Looks clean:     {}'.format(clean))
