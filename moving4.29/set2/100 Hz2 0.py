import cv2
import numpy as np
import os
import pandas as pd
from scipy import interpolate

# ====== 使用者參數設定 ======
folder_path = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\moving4.29\set2\100 Hz2 0"

image_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
image_files.sort()

output_folder = os.path.join(folder_path, "output 100 Hz2 0")
os.makedirs(output_folder, exist_ok=True)

excel_output = os.path.join(output_folder, "100_Hz2_0.xlsx")

rotation_deg = 0  # 旋轉角度 (度)

crop_topleft = (38, 570)  # 裁切區域左上角 (X, Y)
crop_bottomright = (610, 860)  # 裁切區域右上角 (X, Y)
origin_uncrop = (crop_topleft[0], int((crop_bottomright[1]+crop_topleft[1])/2))  # 原始未裁切的原點 (X, Y)
subtract_mask_x_uncrop = 334  # 用於減去液滴的區域 X 座標
subtract_mask_x = subtract_mask_x_uncrop - crop_topleft[0]  # 減去液滴的區域 X 座標在裁切後圖片中的位置

# 計算原點在裁切後圖片中的位置
origin_x = origin_uncrop[0] - crop_topleft[0]
origin_y = origin_uncrop[1] - crop_topleft[1]

pixel_per_mm = (584-334)/0.4  # 每毫米多少像素
droplet_height_mm = 0.06 # 液滴高度 (毫米)
area_threshold = 66000  # 面積閾值，若液滴輪廓小于此值將被认为未框选完全

fps = 2000  # 每秒幀數
timestep = 1 / fps  # 每幀的時間間隔 (秒)
actuate_frame = 8091  # 开通电极的帧
actuate_time_on_graph = 0.2 # 图表上开通电极的时刻 (秒)
start_frame = actuate_frame - int(actuate_time_on_graph / timestep)  # 开始处理的帧
if start_frame < 0:
    start_frame = 0  # 确保开始帧不小于0
    print("⚠️ 开始帧小于0，已调整为0")

# ====== 使用者參數設定結束 ======


def find_largest_contour_from_binary(binary_img, filename):
    '''先找封闭曲线，如果一个封闭曲线套了小的封闭曲线，只保留最外层的（cv.RETR_EXTERNAL）；然后找面积最大的封闭曲线，也就是液滴轮廓'''
    cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in cnts:
        if len(cnt) >= 5:
            (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(cnt)
            if major_axis >= 50 and minor_axis >= 50:
                valid_contours.append(cnt)

    if valid_contours:
        return max(valid_contours, key=cv2.contourArea)
    else:
        print(f"❌ No valid contours: {filename}")
        return None



data_list = []
for index, filename in enumerate(image_files[start_frame:]):
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"❌ 無法讀取圖片: {filename}")
        continue

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 計算逆時針旋轉 1 度的旋轉矩陣
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1)
    # 使用旋轉矩陣來旋轉圖片
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # cv2.imshow("Rotated Image", rotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 裁切圖片
    cropped_image = rotated_image[crop_topleft[1]:crop_bottomright[1], crop_topleft[0]:crop_bottomright[0]]

    # cv2.imshow("Cropped Image", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    

    # 模糊：用bilateral而不用gaussian，这样可以避免液滴边缘被模糊
    blurred_image = cv2.bilateralFilter(cropped_image, d=9, sigmaColor=75, sigmaSpace=15)

    # cv2.imshow("Blurred Image", blurred_image)
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()

    # 检测边缘
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(np.absolute(sobel_edges))
    
    # 挑选主要边缘，生成黑白图像
    _, binary_image = cv2.threshold(sobel_edges, 7, 255, cv2.THRESH_BINARY)

    # cv2.imshow("Binary Image", binary_image)
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()

    # 找面积最大的封闭曲线，也就是液滴轮廓
    largest_contour = find_largest_contour_from_binary(binary_image, filename)
    if largest_contour is None:
        continue

    # 画最大封闭曲线的mask
    largest_contour_mask = np.zeros_like(cropped_image)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # print(f"largest contour area: {cv2.contourArea(largest_contour)}")
    # cv2.imshow("Largest Contour Mask 0", largest_contour_mask)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    
    kernel_size = 3
    t = 0
    while cv2.contourArea(largest_contour) < area_threshold and t < 5:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        largest_contour_mask = cv2.morphologyEx(largest_contour_mask, cv2.MORPH_CLOSE, kernel)
        largest_contour = find_largest_contour_from_binary(largest_contour_mask, filename)
        if largest_contour is None:
            break
        kernel_size += 2
        t += 1

        # cv2.imshow("Largest Contour Mask " + str(t), largest_contour_mask)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()

    if largest_contour is None or t >= 5:
        print(f"❌ No valid contours after morph close: {filename}")
        continue

    filled_droplet = np.zeros_like(cropped_image)
    cv2.drawContours(filled_droplet, [largest_contour], -1, 255, thickness=cv2.FILLED)

    kernel = np.ones((9, 9), np.uint8)
    filled_droplet = cv2.morphologyEx(filled_droplet, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("Smoothed", filled_droplet)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    largest_contour = find_largest_contour_from_binary(filled_droplet, filename)
    if largest_contour is None:
        print(f"❌ No valid contours after morph open: {filename}")
        continue
    filled_droplet = np.zeros_like(cropped_image)
    cv2.drawContours(filled_droplet, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    image_with_contour = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 1)
    
    # cv2.imshow("Image with Contour", image_with_contour)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    (h, w) = cropped_image.shape[:2]
    subtract_mask = np.zeros_like(cropped_image)
    subtract_mask = 255 - subtract_mask
    cv2.rectangle(subtract_mask, (0,0), (subtract_mask_x, h), 0, -1) # todo
    overlap = cv2.bitwise_and(filled_droplet, subtract_mask)
    overlap_px = cv2.countNonZero(overlap)
    overlap_mm2 = overlap_px / (pixel_per_mm ** 2)
    droplet_volume_mm3 = overlap_mm2 * droplet_height_mm
    droplet_volume_nL = droplet_volume_mm3 * 1000

    image_with_overlap_cnt = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    overlap_cnts, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_with_overlap_cnt, overlap_cnts, -1, (255, 0, 0), 1)
    
    # cv2.putText(image_with_overlap_cnt, f"Overlap: {overlap_px:.2f} px", (50, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.imshow("Image with Overlap Contour", image_with_overlap_cnt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        relative_cX = cX - origin_x
        relative_cY = cY - origin_y

        new_relative_X = relative_cX
        new_relative_Y = relative_cY

        new_relative_X_mm = new_relative_X / pixel_per_mm
        new_relative_Y_mm = new_relative_Y / pixel_per_mm

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
        num_samples = 2000   # how many points you want
        s_uniform = np.linspace(0, s[-1], num_samples)
        x_uniform = fx(s_uniform)
        y_uniform = fy(s_uniform)

        # Stack back into an array of points
        dense_pts = np.vstack([x_uniform, y_uniform]).T

        y_min = origin_y - 1
        y_max = origin_y + 1

        mask = (dense_pts[:,1] >= y_min) & (dense_pts[:,1] <= y_max)
        filtered_pts = dense_pts[mask]

        if len(filtered_pts) <= 1:
            print(f"No leading/trailing points found in specified range: {filename}")
            # if this happens, try increasing num_samples or adjusting x_min/x_max to a wider range
            continue

        min_idx = np.argmin(filtered_pts[:,0])
        max_idx = np.argmax(filtered_pts[:,0])

        trailing_point = filtered_pts[min_idx]
        leading_point = filtered_pts[max_idx]

        trailing_point_int = tuple(np.round(trailing_point).astype(int))
        leading_point_int = tuple(np.round(leading_point).astype(int))

        
        rel_leading_x = leading_point[0] - origin_x
        rel_leading_y = leading_point[1] - origin_y
        rel_trailing_x = trailing_point[0] - origin_x
        rel_trailing_y = trailing_point[1] - origin_y
        
        new_rel_leading_x = rel_leading_x
        new_rel_leading_y = rel_leading_y

        new_rel_trailing_x = rel_trailing_x
        new_rel_trailing_y = rel_trailing_y

        # Convert to mm
        new_rel_leading_x_mm = new_rel_leading_x / pixel_per_mm
        new_rel_leading_y_mm = new_rel_leading_y / pixel_per_mm
        new_rel_trailing_x_mm = new_rel_trailing_x / pixel_per_mm
        new_rel_trailing_y_mm = new_rel_trailing_y / pixel_per_mm

        # Append all data to list
        data_list.append([
            filename,
            index * timestep,
            new_relative_X_mm,
            new_relative_Y_mm,
            new_rel_leading_x_mm,
            new_rel_leading_y_mm,
            new_rel_trailing_x_mm,
            new_rel_trailing_y_mm,
            droplet_volume_nL
        ])

        # 畫圖與儲存
        cv2.circle(image_with_contour, (cX, cY), 5, (0, 0, 255), -1)
        cv2.circle(image_with_contour, (origin_x, origin_y), 5, (255, 0, 0), -1)
        cv2.circle(image_with_contour, leading_point_int, 5, (255,0,255), -1)
        cv2.circle(image_with_contour, trailing_point_int, 5, (255,255,0), -1)

        output_path = os.path.join(output_folder, f"processed_{filename}")
        cv2.imwrite(output_path, image_with_contour)
    # print(filename, "processed successfully.")

# 輸出 Excel
df = pd.DataFrame(
    data_list,
    columns=[
        "Filename",
        "Time",
        "Centroid_X_mm",
        "Centroid_Y_mm",
        "Leading_X_mm",
        "Leading_Y_mm",
        "Trailing_X_mm",
        "Trailing_Y_mm",
        "Volume_nL"
    ]
)
df.to_excel(excel_output, index=False)
print(f"✅ 已輸出質心結果至 Excel：{excel_output}")