import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# ====== 使用者參數設定 ======
folder_path = r"C:\Users\wjrwe\OneDrive - University of Toronto\NTU2025\image processing practice\1 kHz 0"
output_folder = os.path.join(folder_path, "output_images")
os.makedirs(output_folder, exist_ok=True)

# # 設定新的原點位置
# origin_x = 140 # 新的原點 X 座標 (像素)
# origin_y = 535  # 新的原點 Y 座標 (像素) 

origin_x = 260 # 新的原點 X 座標 (像素)
origin_y = 400  # 新的原點 Y 座標 (像素) 
pixel_per_mm = 630  # 每毫米多少像素

# 儲存質心結果
centroid_list = []

# 取得所有 .tif 圖片
image_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
image_files.sort()

t = 0
for filename in image_files:
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"❌ 無法讀取圖片: {filename}")
        continue

    # 裁切區域：Y:148~694, X:634~914
    # cropped_image = image[148:694, 634:914]
    cropped_image = image[170:620, 250:1000]
    
    # 取得裁切後圖片的中心點
    (h, w) = cropped_image.shape[:2]
    center = (w // 2, h // 2)

    # 計算逆時針旋轉 1 度的旋轉矩陣c
    rotation_matrix = cv2.getRotationMatrix2D(center, 0, 1)  # 1 代表旋轉 1 度，1 為縮放比例

    # 使用旋轉矩陣來旋轉圖片
    rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (w, h))
    # rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    blurred_image = cv2.GaussianBlur(rotated_image, (5, 5), 10)

    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(np.absolute(sobel_edges))

    _, binary_image = cv2.threshold(sobel_edges, 7, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        if len(cnt) >= 5:
            (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(cnt)
            if major_axis >= 50 and minor_axis >= 50:
                valid_contours.append(cnt)

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            relative_cX = cX - origin_x
            relative_cY = cY - origin_y

            # 座標軸轉換
            # new_relative_X = -relative_cY
            # new_relative_Y = -relative_cX
            new_relative_X = relative_cX
            new_relative_Y = relative_cY
            new_relative_X_mm = new_relative_X / pixel_per_mm
            new_relative_Y_mm = new_relative_Y / pixel_per_mm

            # 儲存到列表
            centroid_list.append([filename, t, new_relative_X_mm, new_relative_Y_mm])

            # 畫圖與儲存
            image_with_contour = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(image_with_contour, (cX, cY), 5, (0, 0, 255), -1)
            cv2.circle(image_with_contour, (origin_x, origin_y), 5, (255, 0, 0), -1)

            output_path = os.path.join(output_folder, f"processed_{filename}")
            cv2.imwrite(output_path, image_with_contour)
        else:
            print(f"⚠️ 面積為零: {filename}")
    else:
        print(f"⚠️ 無有效輪廓: {filename}")

    t += 0.0005


# 輸出 Excel
df = pd.DataFrame(centroid_list, columns=["Filename", "Time", "X_mm", "Y_mm"])
excel_output = os.path.join(folder_path, "centroid_coordinates.xlsx")
df.to_excel(excel_output, index=False)
print(f"✅ 已輸出質心結果至 Excel：{excel_output}")