import cv2
import numpy as np
import os
import pandas as pd
from scipy import interpolate
import copy

# ====== 使用者參數設定 ======
folder_path = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\splitting1kHz7.30"
output_folder = os.path.join(folder_path, "output splitting1kHz7.30")
os.makedirs(output_folder, exist_ok=True)

# 取得所有 .tif 圖片
image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
image_files.sort()

# # 設定新的原點位置 (1095，545)
origin_x = 125 # 新的原點 X 座標 (像素) 545，945
origin_y = 855  # 新的原點 Y 座標 (像素) 
pixel_per_mm = 153/2  # 每毫米多少像素
droplet_height_mm = 0.06

fps = 500
timestep = 1 / fps  # 每幀的時間間隔 (秒)

def find_2_largest_contours_from_binary(binary_img, filename):
    '''先找封闭曲线，如果一个封闭曲线套了小的封闭曲线，只保留最外层的（cv.RETR_EXTERNAL）；然后找面积最大的两个封闭曲线，也就是液滴轮廓'''
    cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"Found {len(cnts)} total contours in {filename}")
    
    valid_contours = []
    for cnt in cnts:
        if len(cnt) >= 5:
            (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(cnt)
            if major_axis >= 50 and minor_axis >= 50:
                valid_contours.append(cnt)

    # print(f"Valid contours: {len(valid_contours)}")
    
    if len(valid_contours) >= 2:
        # Sort contours by area in descending order and return the 2 largest
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        return sorted_contours[0], sorted_contours[1]
    elif len(valid_contours) == 1:
        # If only one valid contour, return it twice
        return valid_contours[0], valid_contours[0]
    else:
        print(f"❌ No valid contours: {filename}")
        return None, None


# 儲存質心結果
data_list = []
centroid_list = []

for index, filename in enumerate(image_files[2302:]): # electrodes are activated at frame 2802, 
                                                      # timestep = 1 / fps = 1 / 500 = 0.002s  
                                                      # 2802 - (1s/timestep) = 2802 - 500 = 2302
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"❌ 無法讀取圖片: {filename}")
        continue

    # 取得裁切後圖片的中心點
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 計算逆時針旋轉 1 度的旋轉矩陣c
    rotation_matrix = cv2.getRotationMatrix2D(center, 0, 1)  # 1 代表旋轉 1 度，1 為縮放比例
    # 使用旋轉矩陣來旋轉圖片
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # 裁切區域：Y:170:620, X:250:1000 (1030, 325)（1150，545)
    cropped_image = rotated_image[90:960, 410:670]

    # 模糊：用bilateral而不用gaussian，这样可以避免液滴边缘被模糊
    # blurred_image = cv2.GaussianBlur(cropped_image, (7, 7), 20)
    blurred_image = cv2.bilateralFilter(cropped_image, d=9, sigmaColor=75, sigmaSpace=15)

    # 检测边缘
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(np.absolute(sobel_edges))
    
    # 挑选主要边缘，生成黑白图像
    _, binary_image = cv2.threshold(sobel_edges, 10, 255, cv2.THRESH_BINARY)

    mask = np.zeros_like(cropped_image)
    largest_contour, second_largest_contour = find_2_largest_contours_from_binary(binary_image, filename)
    if largest_contour is None:
        print(f"❌ No valid contours 999: {filename}")
        continue
    if cv2.contourArea(largest_contour) > 60000:
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    else:
        cv2.drawContours(mask, [largest_contour,second_largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    kernel_size = 3
    while not (cv2.contourArea(largest_contour) > 60000 or (cv2.contourArea(largest_contour) > 37000 and cv2.contourArea(second_largest_contour) > 22000)):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        largest_contour, second_largest_contour = find_2_largest_contours_from_binary(mask, filename)
        if largest_contour is None:
            break
        kernel_size += 2
        # print(cv2.contourArea(largest_contour), cv2.contourArea(second_largest_contour))

    if largest_contour is None:
        print(f"❌ No valid contours after morph: {filename}")
        continue

    filled_droplet = np.zeros_like(cropped_image)
    if cv2.contourArea(largest_contour) > 60000:
        cv2.drawContours(filled_droplet, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    else:
        cv2.drawContours(filled_droplet, [largest_contour,second_largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    kernel = np.ones((9, 9), np.uint8)
    filled_droplet = cv2.morphologyEx(filled_droplet, cv2.MORPH_OPEN, kernel)
    largest_contour, second_largest_contour = find_2_largest_contours_from_binary(filled_droplet, filename)

    (h, w) = cropped_image.shape[:2]

    subtract_mask1 = np.zeros_like(cropped_image)
    subtract_mask1 = 255 - subtract_mask1
    cv2.rectangle(subtract_mask1, (0, 376), (w, h), 0, -1)
    overlap1 = cv2.bitwise_and(filled_droplet, subtract_mask1)
    overlap1_px = cv2.countNonZero(overlap1)
    overlap1_mm2 = overlap1_px / (pixel_per_mm ** 2)
    droplet_volume1_mm3 = overlap1_mm2 * droplet_height_mm
    droplet_volume1_nL = droplet_volume1_mm3 * 1000

    subtract_mask2 = np.zeros_like(cropped_image)
    subtract_mask2 = 255 - subtract_mask2
    cv2.rectangle(subtract_mask2, (0,0), (w, 674), 0, -1)
    overlap2 = cv2.bitwise_and(filled_droplet, subtract_mask2)
    overlap2_px = cv2.countNonZero(overlap2)
    overlap2_mm2 = overlap2_px / (pixel_per_mm ** 2)
    droplet_volume2_mm3 = overlap2_mm2 * droplet_height_mm
    droplet_volume2_nL = droplet_volume2_mm3 * 1000

    # cv2.imshow("overlap1", overlap1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("overlap2", overlap2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    overlap1_cnt,_ = cv2.findContours(overlap1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlap2_cnt,_= cv2.findContours(overlap2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    showimg1 = copy.deepcopy(cropped_image)
    showimg2 = copy.deepcopy(cropped_image)
    cv2.drawContours(showimg1, overlap1_cnt, -1, (255, 0, 0), 1)
    cv2.drawContours(showimg2, overlap2_cnt, -1, (255, 0, 0), 1)

    # cv2.imshow("overlap1_cnt", showimg1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("overlap2_cnt", showimg2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    if largest_contour is second_largest_contour:
        M1 = cv2.moments(largest_contour)
        M2 = M1
    else:
        M1 = cv2.moments(largest_contour)
        M2 = cv2.moments(second_largest_contour)

    if M1["m00"] != 0 and M2["m00"] != 0:
        cX1 = int(M1["m10"] / M1["m00"])
        cY1 = int(M1["m01"] / M1["m00"])
        cX2 = int(M2["m10"] / M2["m00"])
        cY2 = int(M2["m01"] / M2["m00"])

        centroid_list.append([cX1, cY1, cX2, cY2])
        relative_cX1 = cX1 - origin_x
        relative_cY1 = cY1 - origin_y
        relative_cX2 = cX2 - origin_x
        relative_cY2 = cY2 - origin_y

        # 座標軸轉換
        new_relative_X1 = -relative_cY1
        new_relative_Y1 = relative_cX1
        new_relative_X2 = -relative_cY2
        new_relative_Y2 = relative_cX2
        new_relative_X1_mm = new_relative_X1 / pixel_per_mm
        new_relative_Y1_mm = new_relative_Y1 / pixel_per_mm
        new_relative_X2_mm = new_relative_X2 / pixel_per_mm
        new_relative_Y2_mm = new_relative_Y2 / pixel_per_mm

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
        
        # # Example: draw sampled points
        # for p in dense_pts.astype(int):
        #     cv2.circle(image_with_contour, tuple(p), 1, (255,0,0), -1)

        x_min = origin_x - 1
        x_max = origin_x + 1

        mask = (dense_pts[:,0] >= x_min) & (dense_pts[:,0] <= x_max)
        filtered_pts = dense_pts[mask]

        if len(filtered_pts) == 0:
            print(f"No points found in specified X range: {filename}")
        else:
            min_y_idx = np.argmin(filtered_pts[:,1])
            max_y_idx = np.argmax(filtered_pts[:,1])

            trailing_point = filtered_pts[min_y_idx]
            leading_point = filtered_pts[max_y_idx]

            trailing_point_int = tuple(np.round(trailing_point).astype(int))
            leading_point_int = tuple(np.round(leading_point).astype(int))

        
        rel_leading_x = leading_point[0] - origin_x
        rel_leading_y = leading_point[1] - origin_y
        rel_trailing_x = trailing_point[0] - origin_x
        rel_trailing_y = trailing_point[1] - origin_y
        
        new_rel_leading_x = -rel_leading_y
        new_rel_leading_y = rel_leading_x

        new_rel_trailing_x = -rel_trailing_y
        new_rel_trailing_y = rel_trailing_x

        # Convert to mm
        new_rel_leading_x_mm = new_rel_leading_x / pixel_per_mm
        new_rel_leading_y_mm = new_rel_leading_y / pixel_per_mm
        new_rel_trailing_x_mm = new_rel_trailing_x / pixel_per_mm
        new_rel_trailing_y_mm = new_rel_trailing_y / pixel_per_mm

        # Append all data to list
        data_list.append([
            filename,
            index * timestep,
            droplet_volume1_nL,
            droplet_volume2_nL,
            new_relative_X1_mm,
            new_relative_Y1_mm,
            new_relative_X2_mm,
            new_relative_Y2_mm,
            new_rel_leading_x_mm,
            new_rel_leading_y_mm,
            new_rel_trailing_x_mm,
            new_rel_trailing_y_mm
        ])

        # 畫圖與儲存
        image_with_contour = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 1)  # Green for largest contour
        if second_largest_contour is not largest_contour:
            cv2.drawContours(image_with_contour, [second_largest_contour], -1, (0, 255, 255), 1)  # Yellow for second largest contour
        cv2.circle(image_with_contour, (cX1, cY1), 5, (0, 0, 255), -1)
        cv2.circle(image_with_contour, (cX2, cY2), 5, (0, 0, 255), -1)
        cv2.circle(image_with_contour, (origin_x, origin_y), 5, (255, 0, 0), -1)
        cv2.circle(image_with_contour, leading_point_int, 5, (255,0,255), -1)
        cv2.circle(image_with_contour, trailing_point_int, 5, (255,255,0), -1)

        # # Rotate the image 90 degrees clockwise after drawing all contours and points
        # h, w = image_with_contour.shape[:2]
        # # For 90 degree clockwise rotation, we need to swap width and height
        # rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), -90, 1.0)
        # # Adjust the translation to keep the image centered
        # rotation_matrix[0, 2] += (h - w) / 2
        # rotation_matrix[1, 2] += (w - h) / 2
        # image_with_contour = cv2.warpAffine(image_with_contour, rotation_matrix, (h, w))

        cv2.drawContours(image_with_contour, overlap1_cnt, -1, (255, 0, 0), 1)
        cv2.drawContours(image_with_contour, overlap2_cnt, -1, (255, 0, 0), 1)

        output_path = os.path.join(output_folder, f"processed_{filename}")
        cv2.imwrite(output_path, image_with_contour)
        # print(index, droplet_volume1_nL, droplet_volume2_nL)


# 輸出 Excel
df = pd.DataFrame(
    data_list,
    columns=[
        "Filename",
        "Time",
        "Volume1_nL",
        "Volume2_nL",
        "Centroid1_X_mm",
        "Centroid1_Y_mm",
        "Centroid2_X_mm",
        "Centroid2_Y_mm",
        "Leading_X_mm",
        "Leading_Y_mm",
        "Trailing_X_mm",
        "Trailing_Y_mm"
    ]
)
excel_output = os.path.join(output_folder, "data_1khz_new_volume.xlsx")
df.to_excel(excel_output, index=False)
print(f"✅ 已輸出質心結果至 Excel：{excel_output}")