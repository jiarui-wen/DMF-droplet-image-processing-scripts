import cv2 as cv
import numpy as np
import copy

img_path = r"C:\Users\wjrwe\Documents\NTU2025ImageProcessing\image processing practice\1 kHz 0 volume graph\1 kHz 0_00001879.tif"
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

origin_x = 10 # 新的原點 X 座標 (像素)
origin_y = 230  # 新的原點 Y 座標 (像素) 
pixel_per_mm = 630  # 每毫米多少像素

# 裁切
cropped_image = img[170:620, 250:1000]

cv.imshow("cropped", cropped_image)
cv.waitKey(0)
cv.destroyAllWindows()

# 模糊：用bilateral而不用gaussian，这样可以避免液滴边缘被模糊
# blurred_image = cv.GaussianBlur(cropped_image, (7,7), 20)
blurred_image = cv.bilateralFilter(cropped_image, d=9, sigmaColor=75, sigmaSpace=15)

cv.imshow("blurred", blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()

# 检测边缘
sobel_x = cv.Sobel(blurred_image, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(blurred_image, cv.CV_64F, 0, 1, ksize=3)
sobel_edges = cv.magnitude(sobel_x, sobel_y)
sobel_edges = np.uint8(np.absolute(sobel_edges))

# 挑选主要边缘，生成黑白图像
_, binary_image = cv.threshold(sobel_edges, 5, 255, cv.THRESH_BINARY)

cv.imshow("binary img", binary_image)
cv.waitKey(0)
cv.destroyAllWindows()

# 找封闭曲线：如果一个封闭曲线套了小的封闭曲线，只保留最外层的（cv.RETR_EXTERNAL）。保留曲线上的所有点（cv.CHAIN_APPROX_NONE）
contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contoured_img = copy.deepcopy(cropped_image)
cv.drawContours(contoured_img, contours, -1, (255,0,0), 3)

cv.imshow("contoured img", contoured_img)
cv.waitKey(0)
cv.destroyAllWindows()

# 找面积最大的封闭曲线，也就是液滴轮廓
valid_contours = []
for cnt in contours:
    if len(cnt) >= 5:
        (_, _), (major_axis, minor_axis), _ = cv.fitEllipse(cnt)
        if major_axis >= 50 and minor_axis >= 50:
            valid_contours.append(cnt)
if valid_contours:
        largest_contour = max(valid_contours, key=cv.contourArea)

# 画最大封闭曲线的mask
largest_contour_mask = np.zeros_like(cropped_image)
cv.drawContours(largest_contour_mask, [largest_contour], -1, (255, 255, 255), 1)

cv.imshow("largest contour mask", largest_contour_mask)
cv.waitKey(0)
cv.destroyAllWindows()

# 对最大封闭曲线的mask进行morph close，连接断裂的部分
kernel = np.ones((5, 5), np.uint8)
closed_img = cv.morphologyEx(largest_contour_mask, cv.MORPH_CLOSE, kernel)

cv.imshow("closed", closed_img)
cv.waitKey(0)
cv.destroyAllWindows()

# 找morph close之后mask的轮廓，取最大面积的轮廓
# 按理说不用再从valid_contours找largest_contour，因为刚找的contours里面只有一个contour
# 但是考虑到一次morph close不一定能完全封闭，找largest_contour还是有必要的，以此来计算面积
# todo: 以面积作为条件进行while loop (while cv.contourArea(largest_contour) < 100000:)
contours, _ = cv.findContours(closed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# valid_contours = []
# for cnt in contours:
#     if len(cnt) >= 5:
#         (_, _), (major_axis, minor_axis), _ = cv.fitEllipse(cnt)
#         if major_axis >= 50 and minor_axis >= 50:
#             valid_contours.append(cnt)
# if valid_contours:
#         largest_contour = max(valid_contours, key=cv.contourArea)
largest_contour_mask = np.zeros_like(cropped_image)
cv.drawContours(largest_contour_mask, [contours[0]], -1, (255, 255, 255), 1)  

cv.imshow("largest", largest_contour_mask)
cv.waitKey(0)
cv.destroyAllWindows()





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