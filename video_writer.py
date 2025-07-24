import cv2
import glob
import os

# === USER PARAMETERS ===
folder_path = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\1 kHz 0 new\output 1kHz new volume"
output_video = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\output_1khz_new_volume_7_23.mp4"
fps = 200                          # Playback frames per second
scale_factor = 1.0                     # Resize factor, 1.0 = original size

# === Find and sort all TIFF images ===
# image_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))

image_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
image_files.sort()

if not image_files:
    raise ValueError("❌ No .tif files found in the specified folder!")

# === Read the first image to get size ===


image_path = os.path.join(folder_path, image_files[0])
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
if frame is None:
    raise ValueError("❌ Could not read the first image!")

# Optionally resize
if scale_factor != 1.0:
    frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)

height, width = frame.shape[:2]

# === Define video writer ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # Use 'XVID' for AVI
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === Write frames to video ===
for filename in image_files:
    image_path = os.path.join(folder_path, filename)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ Could not read {filename}, skipping...")
        continue

    # Optionally resize
    if scale_factor != 1.0:
        img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)

    # # Optionally overlay frame index
    # cv2.putText(img, f"Frame {idx}", (10,30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    video_writer.write(img)

print("✅ Video writing completed.")
video_writer.release()
