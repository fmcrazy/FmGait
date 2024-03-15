import cv2
import numpy as np

def clothing_augmentation(image_path, iterations=1, kernel_size=3):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 创建一个椭圆形的核（用于膨胀操作）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 对人的轮廓部分进行膨胀操作
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)

    return dilated_image

class RansDilated(object):
    def __init__(self, iterations=1, kernel_size=3):
        self.iterations = iterations
        self.kernel_size = kernel_size

    def __call__(self, seq):
        # 创建一个椭圆形的核（用于膨胀操作）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 对人的轮廓部分进行膨胀操作
        dilated_seq = cv2.dilate(seq, kernel, iterations=iterations)

        return dilated_seq

