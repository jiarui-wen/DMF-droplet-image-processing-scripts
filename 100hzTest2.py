import cv2 as cv
import numpy as np
import copy
import os
import string


def find_largest_contour(cnts, filename):
    '''找面积最大的封闭曲线，也就是液滴轮廓'''
    valid_contours = []
    for cnt in cnts:
        if len(cnt) >= 5:
            (_, _), (major_axis, minor_axis), _ = cv.fitEllipse(cnt)
            if major_axis >= 50 and minor_axis >= 50:
                valid_contours.append(cnt)

    if valid_contours:
        return max(valid_contours, key=cv.contourArea)
    else:
        print(f"❌ No valid contours: {filename}")
        return None

def fix_broken_contours(binary_img, save_dir, step_name):
    '''修复断裂的轮廓'''
    # 方法1：形态学闭运算
    closed_7x7 = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    closed_5x5 = cv.morphologyEx(closed_7x7, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # 方法2：膨胀后腐蚀（开运算的逆操作）
    dilated = cv.dilate(binary_img, np.ones((3, 3), np.uint8), iterations=2)
    eroded = cv.erode(dilated, np.ones((3, 3), np.uint8), iterations=1)
    
    # 方法3：组合方法
    combined = cv.bitwise_or(closed_5x5, eroded)
    
    # 方法4：迭代闭运算 - 逐步增加核大小直到轮廓闭合
    iterative_closed = binary_img.copy()
    kernel_sizes = [3, 5, 7, 9, 11]
    for kernel_size in kernel_sizes:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        iterative_closed = cv.morphologyEx(iterative_closed, cv.MORPH_CLOSE, kernel)
        # 检查是否形成了闭合轮廓
        contours, _ = cv.findContours(iterative_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_area = max([cv.contourArea(cnt) for cnt in contours if len(cnt) >= 5])
            if largest_area > 50000:  # 如果面积足够大，认为轮廓已闭合
                break
    
    # 方法5：距离变换 + 分水岭方法
    # 先进行距离变换
    dist_transform = cv.distanceTransform(binary_img, cv.DIST_L2, 5)
    # 归一化距离变换
    dist_transform = cv.normalize(dist_transform, None, 0, 1.0, cv.NORM_MINMAX)
    # 阈值化得到种子点
    _, sure_fg = cv.threshold(dist_transform, 0.3, 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    # 膨胀种子点
    sure_fg = cv.dilate(sure_fg, np.ones((3, 3), np.uint8), iterations=2)
    # 形态学闭运算连接
    watershed_closed = cv.morphologyEx(sure_fg, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    
    # 方法6：轮廓连接算法 - 查找轮廓端点并连接
    contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_connected = np.zeros_like(binary_img)
    
    if contours:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv.contourArea)
        # 绘制轮廓
        cv.drawContours(contour_connected, [largest_contour], -1, 255, 1)
        
        # 查找轮廓的端点（轮廓的起点和终点）
        if len(largest_contour) > 2:
            start_point = tuple(largest_contour[0][0])
            end_point = tuple(largest_contour[-1][0])
            
            # 计算端点之间的距离
            distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            
            # 如果端点距离较近，用直线连接
            if distance < 50:  # 阈值可调整
                cv.line(contour_connected, start_point, end_point, 255, 2)
        
        # 对连接的轮廓进行填充
        contour_connected = cv.morphologyEx(contour_connected, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # 方法7：多尺度形态学操作
    multi_scale = binary_img.copy()
    # 使用不同大小的核进行多次闭运算
    for kernel_size in [3, 5, 7, 9]:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        multi_scale = cv.morphologyEx(multi_scale, cv.MORPH_CLOSE, kernel)
        # 每次操作后稍微膨胀以保持连接
        multi_scale = cv.dilate(multi_scale, np.ones((2, 2), np.uint8), iterations=1)
    
    # 方法8：Flood Fill连接 - 在轮廓内部进行填充
    flood_fill = binary_img.copy()
    # 创建掩码用于flood fill
    h, w = flood_fill.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # 找到图像中心点作为种子点
    center_x, center_y = w//2, h//2
    
    # 尝试从中心点进行flood fill
    try:
        cv.floodFill(flood_fill, mask, (center_x, center_y), 255)
    except:
        # 如果中心点失败，尝试其他点
        for x in range(0, w, 50):
            for y in range(0, h, 50):
                if flood_fill[y, x] == 0:  # 如果这个点是背景
                    try:
                        cv.floodFill(flood_fill, mask, (x, y), 255)
                        break
                    except:
                        continue
    
    # 对flood fill结果进行形态学操作
    flood_fill = cv.morphologyEx(flood_fill, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    
    # 方法9：霍夫线变换连接断裂边缘
    hough_connected = binary_img.copy()
    
    # 使用霍夫线变换检测直线
    lines = cv.HoughLinesP(hough_connected, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=20)
    
    if lines is not None:
        # 创建线图像
        line_img = np.zeros_like(hough_connected)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(line_img, (x1, y1), (x2, y2), 255, 2)
        
        # 将检测到的线与原始图像结合
        hough_connected = cv.bitwise_or(hough_connected, line_img)
        
        # 形态学操作连接
        hough_connected = cv.morphologyEx(hough_connected, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # 保存所有方法的结果供比较
    cv.imwrite(os.path.join(save_dir, f'{step_name}_method1_closed.png'), closed_5x5)
    cv.imwrite(os.path.join(save_dir, f'{step_name}_method2_dilate_erode.png'), eroded)
    cv.imwrite(os.path.join(save_dir, f'{step_name}_method3_combined.png'), combined)
    cv.imwrite(os.path.join(save_dir, f'{step_name}_method4_iterative.png'), iterative_closed)
    cv.imwrite(os.path.join(save_dir, f'{step_name}_method5_watershed.png'), watershed_closed)
    cv.imwrite(os.path.join(save_dir, f'{step_name}_method6_contour_connect.png'), contour_connected)
    cv.imwrite(os.path.join(save_dir, f'{step_name}_method7_multiscale.png'), multi_scale)
    cv.imwrite(os.path.join(save_dir, f'{step_name}_method8_flood_fill.png'), flood_fill)
    cv.imwrite(os.path.join(save_dir, f'{step_name}_method9_hough.png'), hough_connected)
    
    return closed_5x5, eroded, combined, iterative_closed, watershed_closed, contour_connected, multi_scale, flood_fill, hough_connected

save_dir = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\100 Hz\process images"
os.makedirs(save_dir, exist_ok=True)

img_path = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\100 Hz\frames\frame_0000.png"
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
cv.imwrite(os.path.join(save_dir, '00_original.png'), img)

origin_x = 20 # 新的原點 X 座標 (像素)  (40, 390)
origin_y = 220  # 新的原點 Y 座標 (像素) 
pixel_per_mm = 630  # 每毫米多少像素

# 裁切 (770, 550) (20, 170)
cropped_image = img[170:580, 20:770]

cv.imshow("cropped", cropped_image)
cv.imwrite(os.path.join(save_dir, '01_cropped.png'), cropped_image)
cv.waitKey(0)
cv.destroyAllWindows()

# After cropping
img_eq = cv.equalizeHist(cropped_image)
cv.imshow("equalized hist", img_eq)
cv.imwrite(os.path.join(save_dir, '01a_equalized_hist.png'), img_eq)
cv.waitKey(0)
cv.destroyAllWindows()
# or
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(cropped_image)
cv.imshow("CLAHE", img_clahe)
cv.imwrite(os.path.join(save_dir, '01b_CLAHE.png'), img_clahe)
cv.waitKey(0)
cv.destroyAllWindows()
# or
img_contrast = cv.convertScaleAbs(cropped_image, alpha=1.5, beta=0)
cv.imshow("contrast adjusted", img_contrast)
cv.imwrite(os.path.join(save_dir, '01c_contrast_adjusted.png'), img_contrast)
cv.waitKey(0)
cv.destroyAllWindows()

cropped_image = img_contrast

# 模糊：用bilateral而不用gaussian，这样可以避免液滴边缘被模糊
# blurred_image = cv.GaussianBlur(cropped_image, (7,7), 20)
blurred_image = cv.bilateralFilter(cropped_image, d=9, sigmaColor=75, sigmaSpace=15)

cv.imshow("blurred", blurred_image)
cv.imwrite(os.path.join(save_dir, '02_blurred.png'), blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()

# 检测边缘
# sobel_x = cv.Sobel(blurred_image, cv.CV_64F, 1, 0, ksize=3)
# sobel_y = cv.Sobel(blurred_image, cv.CV_64F, 0, 1, ksize=3)
sobel_x = cv.Scharr(blurred_image, cv.CV_64F, 1, 0)
sobel_y = cv.Scharr(blurred_image, cv.CV_64F, 0, 1)
sobel_edges = cv.magnitude(sobel_x, sobel_y)
sobel_edges = np.uint8(np.absolute(sobel_edges))
sobel_edges_norm = cv.normalize(sobel_edges, None, 0, 255, cv.NORM_MINMAX)
sobel_edges_visual = np.uint8(sobel_edges_norm)


cv.imshow("sobel edges", sobel_edges_visual)
cv.imwrite(os.path.join(save_dir, '03_sobel_edges.png'), sobel_edges_visual)
cv.waitKey(0)
cv.destroyAllWindows()

# # 尝试使用Canny边缘检测作为替代方案
# canny = cv.Canny(blurred_image, 5, 9, L2gradient=True)
# cv.imshow("canny original", canny)
# cv.imwrite(os.path.join(save_dir, '03a_canny_original.png'), canny)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 对Canny结果进行形态学操作连接断裂
# canny_closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
# cv.imshow("canny after close", canny_closed)
# cv.imwrite(os.path.join(save_dir, '03b_canny_after_close.png'), canny_closed)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 可选：使用Canny结果替代Sobel结果
# # binary_image = canny_closed.copy()

# # 组合Sobel和Canny结果以获得更好的边缘连接
# combined_edges = cv.bitwise_or(sobel_edges_visual, canny_closed)
# cv.imshow("combined edges", combined_edges)
# cv.imwrite(os.path.join(save_dir, '03c_combined_edges.png'), combined_edges)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 对组合结果进行阈值处理
# _, combined_binary = cv.threshold(combined_edges, 5, 255, cv.THRESH_BINARY)
# cv.imshow("combined binary", combined_binary)
# cv.imwrite(os.path.join(save_dir, '03d_combined_binary.png'), combined_binary)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 可选：使用组合结果替代原始binary_image
# # binary_image = combined_binary.copy()


# 挑选主要边缘，生成黑白图像
_, binary_image = cv.threshold(sobel_edges, 30, 255, cv.THRESH_BINARY)

cv.imshow("binary img", binary_image)
cv.imwrite(os.path.join(save_dir, '04_binary.png'), binary_image)
cv.waitKey(0)
cv.destroyAllWindows()

# # 修复断裂的轮廓：使用形态学闭运算连接断裂的边缘
# # 先用较大的核进行闭运算，连接主要断裂
# binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
# cv.imshow("binary img after close 7x7", binary_image)
# cv.imwrite(os.path.join(save_dir, '04a_binary_after_close_7x7.png'), binary_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 再用较小的核进行精细连接
# binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
# cv.imshow("binary img after close 5x5", binary_image)
# cv.imwrite(os.path.join(save_dir, '04b_binary_after_close_5x5.png'), binary_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 可选：使用膨胀操作进一步连接边缘
# binary_image = cv.dilate(binary_image, np.ones((3, 3), np.uint8), iterations=1)
# cv.imshow("binary img after dilate", binary_image)
# cv.imwrite(os.path.join(save_dir, '04c_binary_after_dilate.png'), binary_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 使用修复函数处理断裂的轮廓
# print("尝试不同的轮廓修复方法...")
# method1, method2, method3, method4, method5, method6, method7, method8, method9 = fix_broken_contours(binary_image, save_dir, '04_fix')

# # 显示所有方法的结果供选择
# cv.imshow("Method 1: Morphological Close", method1)
# cv.imshow("Method 2: Dilate + Erode", method2)
# cv.imshow("Method 3: Combined", method3)
# cv.imshow("Method 4: Iterative Close", method4)
# cv.imshow("Method 5: Watershed", method5)
# cv.imshow("Method 6: Contour Connect", method6)
# cv.imshow("Method 7: Multi-scale", method7)
# cv.imshow("Method 8: Flood Fill", method8)
# cv.imshow("Method 9: Hough Lines", method9)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 选择最佳方法（可以根据结果手动调整）
# # binary_image = method1  # 形态学闭运算
# # binary_image = method2  # 膨胀+腐蚀
# # binary_image = method3  # 组合方法
# # binary_image = method4  # 迭代闭运算
# # binary_image = method5  # 距离变换+分水岭
# # binary_image = method6  # 轮廓连接
# # binary_image = method7  # 多尺度形态学
# # binary_image = method8  # Flood Fill方法
# binary_image = method9  # 霍夫线变换方法

# cv.imshow("Final binary image", binary_image)
# cv.imwrite(os.path.join(save_dir, '04_final_binary.png'), binary_image)
# cv.waitKey(0)
# cv.destroyAllWindows()


# 找封闭曲线：如果一个封闭曲线套了小的封闭曲线，只保留最外层的（cv.RETR_EXTERNAL）
contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# contoured_img = np.zeros_like(cropped_image)
# cv.drawContours(contoured_img, contours, -1, (255,0,0), 1)

# cv.imshow("contoured img", contoured_img)
# cv.imwrite(os.path.join(save_dir, '05_contoured.png'), contoured_img)
# cv.waitKey(0)
# cv.destroyAllWindows()


# 找面积最大的封闭曲线，也就是液滴轮廓
largest_contour = find_largest_contour(contours, img_path[-8:])

# 画最大封闭曲线的mask
largest_contour_mask_filled = np.zeros_like(cropped_image)
cv.drawContours(largest_contour_mask_filled, [largest_contour], -1, (255, 255, 255), thickness=cv.FILLED)

cv.imshow("largest contour mask 0 filled", largest_contour_mask_filled)
cv.imwrite(os.path.join(save_dir, '06b_largest_contour_mask_0_filled.png'), largest_contour_mask_filled)
cv.waitKey(0)
cv.destroyAllWindows()

largest_contour_mask = np.zeros_like(cropped_image)
cv.drawContours(largest_contour_mask, [largest_contour], -1, (255, 255, 255), 1)

cv.imshow("largest contour mask 0", largest_contour_mask)
cv.imwrite(os.path.join(save_dir, '06a_largest_contour_mask_0.png'), largest_contour_mask)
cv.waitKey(0)
cv.destroyAllWindows()

t = 1
kernel_size = 3
while cv.contourArea(largest_contour) < 80000:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # largest_contour_mask_filled = cv.morphologyEx(largest_contour_mask_filled, cv.MORPH_CLOSE, kernel)
    largest_contour_mask_filled = cv.erode(largest_contour_mask_filled, kernel, iterations=1)
    contours, _ = cv.findContours(largest_contour_mask_filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours, img_path[-8:])
    letter = string.ascii_lowercase[t-1]  # 'a', 'b', 'c', ...
    cv.imshow("largest contour mask" + str(t), largest_contour_mask_filled)
    cv.imwrite(os.path.join(save_dir, f'07{letter}_largest_contour_mask_filled_{t}.png'), largest_contour_mask_filled)
    cv.waitKey(0)
    cv.destroyAllWindows()
    t += 1
    kernel_size += 2
    print(cv.contourArea(largest_contour))

# image_with_contour = cv.cvtColor(cropped_image, cv.COLOR_GRAY2BGR)
# cv.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 1)
# cv.imshow("largest contour" + str(t), image_with_contour)
# cv.imwrite(os.path.join(save_dir, f'largest_contour_{t}.png'), image_with_contour)
# cv.waitKey(0)
# cv.destroyAllWindows()

filled_droplet = np.zeros_like(cropped_image)
cv.drawContours(filled_droplet, [largest_contour], -1, 255, thickness=cv.FILLED)
cv.imshow("filled droplet", filled_droplet)
cv.imwrite(os.path.join(save_dir, '08b_filled_droplet.png'), filled_droplet)
cv.waitKey(0)
cv.destroyAllWindows()

closed_droplet = np.zeros_like(cropped_image)
cv.drawContours(closed_droplet, [largest_contour], -1, 255, 1)
cv.imwrite(os.path.join(save_dir, '08a_closed_droplet.png'), closed_droplet)

kernel = np.ones((9, 9), np.uint8)
filled_droplet = cv.morphologyEx(filled_droplet, cv.MORPH_OPEN, kernel)
cv.imshow("less fuzzy droplet", filled_droplet)
cv.imwrite(os.path.join(save_dir, '09_less_fuzzy_droplet.png'), filled_droplet)
cv.waitKey(0)
cv.destroyAllWindows()

contours, _ = cv.findContours(filled_droplet, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
largest_contour = find_largest_contour(contours, img_path[-8:])
image_with_contour = cv.cvtColor(cropped_image, cv.COLOR_GRAY2BGR)
cv.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 1)
cv.imshow("largest contour less fuzzy", image_with_contour)
cv.imwrite(os.path.join(save_dir, '10_largest_contour_less_fuzzy.png'), image_with_contour)
cv.waitKey(0)
cv.destroyAllWindows()

# electrode = np.zeros_like(cropped_image)
# cv.rectangle(electrode, (384,74), (692, 382), 255, -1)
# overlap = cv.bitwise_and(filled_droplet, electrode)
subtract_mask = np.zeros_like(cropped_image)
subtract_mask = 255 - subtract_mask
cv.rectangle(subtract_mask, (0,0), (384, 450), 0, -1)
overlap = cv.bitwise_and(filled_droplet, subtract_mask)
cv.imshow("overlap", overlap)
cv.imwrite(os.path.join(save_dir, '11_overlap.png'), overlap)
cv.waitKey(0)
cv.destroyAllWindows()

overlap_cnts, _ = cv.findContours(overlap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
overlap_cnt = find_largest_contour(overlap_cnts, img_path[-8:])
image_with_overlap_cnt = cv.cvtColor(cropped_image, cv.COLOR_GRAY2BGR)
cv.drawContours(image_with_overlap_cnt, [overlap_cnt], -1, (0, 255, 0), 1)
cv.imshow("overlap contour", image_with_overlap_cnt)
cv.imwrite(os.path.join(save_dir, '12_overlap_contour.png'), image_with_overlap_cnt)
cv.waitKey(0)
cv.destroyAllWindows()



print("contour size:", cv.contourArea(largest_contour))
print("filled droplet size:", cv.countNonZero(filled_droplet))
print("overlap size:", cv.countNonZero(overlap))
# # 对最大封闭曲线的mask进行morph close，连接断裂的部分
# kernel = np.ones((5, 5), np.uint8)
# closed_img = cv.morphologyEx(largest_contour_mask, cv.MORPH_CLOSE, kernel)

# cv.imshow("closed", closed_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 找morph close之后mask的轮廓，取最大面积的轮廓
# 按理说不用再从valid_contours找largest_contour，因为刚找的contours里面只有一个contour
# 但是考虑到一次morph close不一定能完全封闭，找largest_contour还是有必要的，以此来计算面积
# todo: 以面积作为条件进行while loop (while cv.contourArea(largest_contour) < 100000:)
# contours, _ = cv.findContours(closed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# valid_contours = []
# for cnt in contours:
#     if len(cnt) >= 5:
#         (_, _), (major_axis, minor_axis), _ = cv.fitEllipse(cnt)
#         if major_axis >= 50 and minor_axis >= 50:
#             valid_contours.append(cnt)
# if valid_contours:
#         largest_contour = max(valid_contours, key=cv.contourArea)
# largest_contour_mask = np.zeros_like(cropped_image)
# cv.drawContours(largest_contour_mask, [contours[0]], -1, (255, 255, 255), 1)  

# cv.imshow("largest", largest_contour_mask)
# cv.waitKey(0)
# cv.destroyAllWindows()






# 下面的代码没有用
# kernel = np.ones((3, 3), np.uint8)
# closed_img = copy.copy(binary_image)
# while cv.contourArea(largest_contour) < 100000:
#     closed_img = cv.morphologyEx(closed_img, cv.MORPH_CLOSE, kernel)
#     cv.imshow("closed img", closed_img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

#     contours, _ = cv.findContours(closed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     valid_contours = []
#     for cnt in contours:
#         if len(cnt) >= 5:
#             (_, _), (major_axis, minor_axis), _ = cv.fitEllipse(cnt)
#             if major_axis >= 50 and minor_axis >= 50:
#                 valid_contours.append(cnt)
#     if valid_contours:
#             largest_contour = max(valid_contours, key=cv.contourArea)

# largest_contour_img = copy.deepcopy(cropped_image)
# cv.drawContours(largest_contour_img, [largest_contour], -1, (0, 255, 0), 3)

# cv.imshow("largest contour", largest_contour_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

     




     

# # cv.floodFill()
# floodfill_img = copy.deepcopy(largest_contour)
# h, w = binary_image.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)
# mask = 255-mask
# cv.floodFill(floodfill_img, mask, (344,226), 0)

# cv.imshow("floodfill", floodfill_img)
# cv.waitKey(0)
# cv.destroyAllWindows()