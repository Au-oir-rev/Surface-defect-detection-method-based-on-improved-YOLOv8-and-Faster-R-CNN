import numpy as np
import cv2
import os

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



def process_ct_image(input_path, output_path, levels=3, a=5):
    """
    对单张图像进行增强并保存。
    """
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")
    if image.dtype != np.uint8:
        raise ValueError("输入图像必须是 8 位灰度图像")

    # 图像增强（包含拉普拉斯金字塔的细节增强）
    final_image = enhance_image_with_pyramid(image, levels=levels, a=a)

    # 保存增强后的图像
    cv2.imwrite(output_path, final_image)
    print(f"Enhanced image saved to {output_path}")

def process_images_in_folder(input_folder, output_folder, levels=3, a=5):
    """
    批量处理文件夹中的所有图像文件。
    支持 .jpg、.jpeg、.png、.tif、.tiff 格式。
    """
    supported_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                process_ct_image(input_path, output_path, levels=levels, a=a)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# 确定文件夹路径
input_folder = r"C:\Users\88983\Desktop\NEU\NEU-DET\IMAGES" # 输入文件夹路径
output_folder = r"C:\Users\88983\Desktop\NEU\NEUimg_enhance"  # 输出文件夹路径

# 批量处理文件夹中的所有图片
process_images_in_folder(input_folder, output_folder, levels=3, a=0.5)
