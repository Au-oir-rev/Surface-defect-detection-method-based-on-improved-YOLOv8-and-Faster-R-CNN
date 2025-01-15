import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def build_laplacian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)

    laplacian_pyramid = []
    for i in range(levels, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)

    return laplacian_pyramid

def enhance_image_with_pyramid(image, levels=3, a=5):
    # 对8位图像进行处理，需要缩放到0-1范围便于操作，然后返回到8位
    image = image.astype(np.float32) / 255.0  # 将图像缩放为0到1的范围

    # 低通滤波
    X1 = cv2.GaussianBlur(image, (5, 5), 0)

    # 高频部分
    high_freq = cv2.subtract(image, X1)

    # 构建拉普拉斯金字塔
    laplacian_pyramid = build_laplacian_pyramid(image, levels)

    # 初始化增强图像
    enhanced_image = image.copy()

    for layer in laplacian_pyramid:
        # 如果层的大小与增强图像不匹配，调整大小
        if layer.shape != enhanced_image.shape:
            layer = cv2.resize(layer, (enhanced_image.shape[1], enhanced_image.shape[0]))

        # 增强图像加上拉普拉斯层的细节
        enhanced_image = cv2.add(enhanced_image, a * layer)

    # 应用增强公式: Y = X + a * (X - X1)
    Y = cv2.addWeighted(enhanced_image, 1.0, high_freq, a, 0.0)

    # 将增强后的图像缩放回8位范围
    Y = np.clip(Y * 255.0, 0, 255).astype(np.uint8)

    return Y

def process_and_save_comparison_image(input_path, output_path, levels=3, a=5):
    """
    读取图像，进行增强，并生成原图与增强后的图像对比图，并保存。
    """
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")
    if image.dtype != np.uint8:
        raise ValueError("输入图像必须是 8 位灰度图像")

    # 图像增强（包含拉普拉斯金字塔的细节增强）
    enhanced_image = enhance_image_with_pyramid(image, levels=levels, a=a)

    # 合并原图和增强图（横向拼接）
    comparison_image = np.hstack((image, enhanced_image))

    # 保存对比图
    cv2.imwrite(output_path, comparison_image)
    print(f"Comparison image saved to {output_path}")

# 示例：读取一张图像并保存原图与增强图的对比图
input_path = "./NEU-DET/IMAGES/scratches_1.jpg" # 输入图像路径
output_path = "comparison_image6.jpg"  # 输出对比图路径

process_and_save_comparison_image(input_path, output_path, levels=3, a=0.5)
