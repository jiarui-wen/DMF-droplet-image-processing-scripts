import os
import cv2

folder_path = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\moving4.29\set2\1 Hz 0-N001"

image_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
image_files.sort()

for filename in image_files:
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)
    if img is not None:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(img_path, rotated)

