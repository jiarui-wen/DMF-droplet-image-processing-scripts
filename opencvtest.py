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

save_dir = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\process images 1 kHz 0_00001879.tif"
os.makedirs(save_dir, exist_ok=True)

# only problematic frame: 1 kHz 0_00000865.tif
# img_path = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\1 kHz 0 new\1 kHz 0_00001879.tif"
img_path = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\DMF-droplet-image-processing-scripts\1 kHz 0_00001879.tif"
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
cv.imwrite(os.path.join(save_dir, '00_original.png'), img)

origin_x = 10 # 新的原點 X 座標 (像素)
origin_y = 230  # 新的原點 Y 座標 (像素) 
pixel_per_mm = 630  # 每毫米多少像素

# 裁切
cropped_image = img[170:620, 250:1000]

cv.imshow("cropped", cropped_image)
cv.imwrite(os.path.join(save_dir, '01_cropped.png'), cropped_image)
cv.waitKey(0)
cv.destroyAllWindows()

# 模糊：用bilateral而不用gaussian，这样可以避免液滴边缘被模糊
# blurred_image = cv.GaussianBlur(cropped_image, (7,7), 20)
blurred_image = cv.bilateralFilter(cropped_image, d=9, sigmaColor=75, sigmaSpace=15)

cv.imshow("blurred", blurred_image)
cv.imwrite(os.path.join(save_dir, '02_blurred.png'), blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()

# 检测边缘
sobel_x = cv.Sobel(blurred_image, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(blurred_image, cv.CV_64F, 0, 1, ksize=3)
# sobel_x = cv.Scharr(blurred_image, cv.CV_64F, 1, 0)
# sobel_y = cv.Scharr(blurred_image, cv.CV_64F, 0, 1)
sobel_edges = cv.magnitude(sobel_x, sobel_y)
sobel_edges = np.uint8(np.absolute(sobel_edges))
sobel_edges_norm = cv.normalize(sobel_edges, None, 0, 255, cv.NORM_MINMAX)
sobel_edges_visual = np.uint8(sobel_edges_norm)


cv.imshow("sobel edges", sobel_edges_visual)
cv.imwrite(os.path.join(save_dir, '03_sobel_edges.png'), sobel_edges_visual)
cv.waitKey(0)
cv.destroyAllWindows()

# canny = cv.Canny(blurred_image, 5, 9, L2gradient=True)
# cv.imshow("canny", canny)
# cv.waitKey(0)
# cv.destroyAllWindows()


# canny_mask = cv.morphologyEx(canny, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
# cv.imshow("canny mask morph close", canny_mask)
# cv.waitKey(0)
# cv.destroyAllWindows()


# cnts, _ = cv.findContours(canny_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# largest_contour = max(cnts, key=cv.contourArea)
# canny_largest_contour = np.zeros_like(cropped_image)
# cv.drawContours(canny_largest_contour, [largest_contour], -1, (255, 255, 255), thickness=cv.FILLED)
# cv.imshow("largest contour canny", canny_largest_contour)
# cv.waitKey(0)
# cv.destroyAllWindows()

# t = 1
# kernel_size = 3
# while cv.contourArea(largest_contour) < 120000:
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     canny_mask = cv.morphologyEx(canny_mask, cv.MORPH_CLOSE, kernel)
#     contours, _ = cv.findContours(canny_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     largest_contour = max(contours, key=cv.contourArea)
#     cv.drawContours(canny_mask, [largest_contour], -1, (255, 255, 255), thickness=cv.FILLED)
#     cv.imshow("largest canny mask filled" + str(t), canny_mask)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#     t += 1
#     kernel_size += 2
#     print(cv.contourArea(largest_contour))

# 挑选主要边缘，生成黑白图像
_, binary_image = cv.threshold(sobel_edges, 5, 255, cv.THRESH_BINARY)

cv.imshow("binary img", binary_image)
cv.imwrite(os.path.join(save_dir, '04_binary.png'), binary_image)
cv.waitKey(0)
cv.destroyAllWindows()

# 找封闭曲线：如果一个封闭曲线套了小的封闭曲线，只保留最外层的（cv.RETR_EXTERNAL）
contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# contoured_img = copy.deepcopy(cropped_image)
contoured_img = np.zeros_like(cropped_image)
cv.drawContours(contoured_img, contours, -1, (255,0,0), 1)

cv.imshow("contoured img", contoured_img)
cv.imwrite(os.path.join(save_dir, '05_contoured.png'), contoured_img)
cv.waitKey(0)
cv.destroyAllWindows()

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
while cv.contourArea(largest_contour) < 123000:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    largest_contour_mask_filled = cv.morphologyEx(largest_contour_mask_filled, cv.MORPH_CLOSE, kernel)
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