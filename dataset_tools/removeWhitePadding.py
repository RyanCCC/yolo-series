import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
import os
import numpy as np

'''
去除图像中的白边
'''

def showImg(img, window_name = 'Test'):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = './TODO_CJL/'
pattern = path+'*.jpg'
for item in tqdm(glob(pattern)):
    img = cv2.imread(item)
    # 对图像进行二值化，先转换成灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
    canny_img = cv2.Canny(gray_img, 50, 150)
    # showImg(canny_img)
    # 遍历像素0：黑色，255为白色
    pixels_x = []
    pixels_y = []
    width, height = canny_img.shape
    for i in range(width):
        for j in range(height):
            if canny_img[i][j] == 255:
                pixels_x.append(i)
                pixels_y.append(j)
    x_min = min(pixels_x)
    x_max = max(pixels_x)
    y_min = min(pixels_y)
    y_max = max(pixels_y)
    ROI = img[x_min:x_max, y_min:y_max]
    # showImg(ROI)
    cv2.imwrite(item, ROI)