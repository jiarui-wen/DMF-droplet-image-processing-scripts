import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import interpolate


# ====== 使用者參數設定 ======
folder_path = r"C:\Users\wjrwe\OneDrive - University of Toronto\NTU2025\image processing practice\1 Hz 0-N001"
output_folder = os.path.join(folder_path, "output_images")
os.makedirs(output_folder, exist_ok=True)

# # 設定新的原點位置
# origin_x = 140 # 新的原點 X 座標 (像素)
# origin_y = 535  # 新的原點 Y 座標 (像素) 

# origin_x = 720 # 新的原點 X 座標 (像素)
# origin_y = 670  # 新的原點 Y 座標 (像素) 

origin_x = 180 # 新的原點 X 座標 (像素)
origin_y = 490  # 新的原點 Y 座標 (像素) 
pixel_per_mm = 630  # 每毫米多少像素

# 儲存質心結果
centroid_list = []

# 取得所有 .tif 圖片
image_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
image_files.sort()

t = 0
# change = False
change = True
for filename in image_files[2902:]:
    # if filename == "1 Hz 0-N001_00002902.tif":
    #     change = True

    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"❌ 無法讀取圖片: {filename}")
        continue

    # 裁切區域：Y:148~694, X:634~914
    # cropped_image = image[148:694, 634:914]
    cropped_image = image[180:690, 540:900]
    
    # 取得裁切後圖片的中心點
    (h, w) = cropped_image.shape[:2]
    center = (w // 2, h // 2)

    # 計算逆時針旋轉 1 度的旋轉矩陣c
    rotation_matrix = cv2.getRotationMatrix2D(center, 0, 1)  # 1 代表旋轉 1 度，1 為縮放比例

    # 使用旋轉矩陣來旋轉圖片
    rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (w, h))
    # rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    if change:
        blurred_image = cv2.GaussianBlur(rotated_image, (7, 7), 15)

        _, binary_image = cv2.threshold(
            blurred_image,
            0,                        # threshold value is ignored
            255,                      # max value to use
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )


        kernel = np.ones((5,5), np.uint8)
        # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    else:
        blurred_image = cv2.GaussianBlur(rotated_image, (7, 7), 15)

        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = np.uint8(np.absolute(sobel_edges))

        _, binary_image = cv2.threshold(sobel_edges, 15, 255, cv2.THRESH_BINARY)

        # binary_image = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

        kernel = np.ones((5,5), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        if len(cnt) >= 5:
            (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(cnt)
            if major_axis >= 50 and minor_axis >= 50 and major_axis <= 200 and minor_axis <=200:
                valid_contours.append(cnt)

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        image_with_contour = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR)


        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = np.array([cX, cY])
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

            # # find leading and trailing points
            # (ellipse_x, ellipse_y), (MA, ma), angle_deg = cv2.fitEllipse(largest_contour)
            # angle_rad = np.deg2rad(angle_deg)
            # major_axis_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            # pts = largest_contour[:,0,:]
            # projections = (pts - centroid) @ major_axis_vector
            # leading_idx = np.argmax(projections)
            # trailing_idx = np.argmin(projections)
            # leading_point = pts[leading_idx]
            # trailing_point = pts[trailing_idx]



            # Suppose you already have:
            pts = largest_contour[:,0,:]  # shape (N,2)

            # Close the contour (important if it wraps around)
            pts = np.vstack([pts, pts[0]])

            # Compute cumulative distance along the contour
            d = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
            s = np.concatenate([[0], np.cumsum(d)])

            # Build interpolators for x(s) and y(s)
            fx = interpolate.interp1d(s, pts[:,0], kind='linear')
            fy = interpolate.interp1d(s, pts[:,1], kind='linear')

            # Sample evenly along the contour
            num_samples = 1000   # how many points you want
            s_uniform = np.linspace(0, s[-1], num_samples)
            x_uniform = fx(s_uniform)
            y_uniform = fy(s_uniform)

            # Stack back into an array of points
            dense_pts = np.vstack([x_uniform, y_uniform]).T


            

            # 畫圖與儲存
            
            cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(image_with_contour, (cX, cY), 5, (0, 0, 255), -1)
            cv2.circle(image_with_contour, (origin_x, origin_y), 5, (255, 0, 0), -1)

            # # Example: draw sampled points
            # for p in dense_pts.astype(int):
            #     cv2.circle(image_with_contour, tuple(p), 1, (255,0,0), -1)

            x_center = centroid[0]
            x_min = x_center - 1
            x_max = x_center + 1

            mask = (dense_pts[:,0] >= x_min) & (dense_pts[:,0] <= x_max)
            filtered_pts = dense_pts[mask]

            if len(filtered_pts) == 0:
                print("No points found in specified X range!")
            else:
                min_y_idx = np.argmin(filtered_pts[:,1])
                max_y_idx = np.argmax(filtered_pts[:,1])

                highest_point = filtered_pts[min_y_idx]
                lowest_point = filtered_pts[max_y_idx]

                highest_point_int = tuple(np.round(highest_point).astype(int))
                lowest_point_int = tuple(np.round(lowest_point).astype(int))

                cv2.circle(image_with_contour, highest_point_int, 5, (0,255,0), -1)
                cv2.circle(image_with_contour, lowest_point_int, 5, (0,0,255), -1)


            # # draw ellipse, major axis, leading and trailing points
            # cv2.ellipse(image_with_contour, ((ellipse_x, ellipse_y), (MA, ma), angle_deg), (0, 255, 255), 1)
            # dx = (MA / 2) * np.cos(angle_rad)
            # dy = (MA / 2) * np.sin(angle_rad)
            # pt1 = (int(ellipse_x - dx), int(ellipse_y - dy))
            # pt2 = (int(ellipse_x + dx), int(ellipse_y + dy))
            # cv2.line(image_with_contour, pt1, pt2, (255, 0, 0), 2)

            # cv2.circle(image_with_contour, tuple(leading_point), 5, (0,255,255), -1)
            # cv2.circle(image_with_contour, tuple(trailing_point), 5, (0,255,255), -1)


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