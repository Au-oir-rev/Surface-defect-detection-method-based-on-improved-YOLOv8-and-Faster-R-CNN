import os
import json
import torch
from torch.utils.data import Dataset
import cv2
import logging
from PIL import Image


class WeldDefectDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        """
        Args:
            images_dir (str): 图像目录的路径（例如，'images/train' 或 'images/val'）。
            annotations_file (str): COCO 格式标注文件的路径。
            transforms (callable, optional): 图像的转换函数。
        """
        self.images_dir = images_dir  # 直接使用 images_dir，不再拼接 split
        self.transforms = transforms

        # 加载 COCO 格式标注文件
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.coco = json.load(f)

        # 创建 image_id 到标注的映射
        self.image_id_to_annotations = {image['id']: [] for image in self.coco['images']}
        for annotation in self.coco['annotations']:
            self.image_id_to_annotations[annotation['image_id']].append(annotation)

        # 图像文件列表
        self.image_files = [
            {
                "file_name": img['file_name'],
                "id": img['id'],
                "width": img['width'],
                "height": img['height']
            }
            for img in self.coco['images']
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_info = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        # 调试信息：打印要加载的图像路径
        logging.debug(f"Loading image: {img_path}")

        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # 使用 cv2 读取16位单通道TIFF图像
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        # 检查是否是16位单通道图像
        if image.dtype != 'uint16':
            raise ValueError(f"Expected 16-bit image, but got {image.dtype}.")

        # 归一化到0-1并转换为8位
        image = (image / 65535.0 * 255).astype('uint8')  # 归一化到0-255

        # 将图像从单通道转换为三通道RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 将图像转换为 PIL 格式，以便与转换函数兼容
        image = Image.fromarray(image)

        # 应用转换（如有）
        if self.transforms:
            image = self.transforms(image)

        annotations = self.image_id_to_annotations[img_info['id']]
        boxes = []
        labels = []
        for ann in annotations:
            x_min, y_min, width, height = ann['bbox']
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_info['id']])  # 确保包含 image_id

        return image, target
