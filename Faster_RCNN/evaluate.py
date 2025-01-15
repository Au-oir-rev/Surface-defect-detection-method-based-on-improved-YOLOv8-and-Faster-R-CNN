# src/predict.py

import torch
import logging
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import tifffile as tiff
from torchvision.ops import nms
import json
from collections import defaultdict
from model import get_model


def setup_logging(log_file='predict.log'):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', log_file)),
            logging.StreamHandler()
        ]
    )


# 自定义DataLoader
class WeldDefectTestDataset(Dataset):
    def __init__(self, images_dir, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.tiff')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = tiff.imread(img_path)
        if img.ndim == 3:
            img = img[:, :, 0]  # 取第一个通道
        elif img.ndim == 2:
            pass  # 已经是单通道
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        img = img.astype(np.float32) / 65535.0  # 归一化到0-1范围

        if self.transforms:
            img = self.transforms(img)

        return img, img_name


def calculate_iou(box1, box2):
    """计算两个框的 IoU"""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    # 计算交集部分
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    # 判断是否有交集
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    # 计算并集
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_intersection_area(box1, box2):
    """计算两个框的交集部分的面积"""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    # 计算交集部分
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    # 判断是否有交集
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    return inter_area


def load_coco_annotations(coco_json_path):
    """加载COCO格式的注释并按图像名称映射"""
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # 创建图像ID到文件名的映射
    img_id_to_name = {img['id']: img['file_name'] for img in coco['images']}

    # 创建文件名到真实框的映射
    gt = defaultdict(list)
    for ann in coco['annotations']:
        img_name = img_id_to_name[ann['image_id']]
        bbox = ann['bbox']  # COCO格式的bbox是 [x, y, width, height]
        # 转换为 [x1, y1, x2, y2]
        x1, y1, width, height = bbox
        x2 = x1 + width
        y2 = y1 + height
        gt[img_name].append({
            'bbox': [x1, y1, x2, y2],
            'category_id': ann['category_id']
        })
    return gt


def load_model(model_path, num_classes, device):
    """
    加载模型用于预测。

    参数：
    - model_path (str): 模型文件路径。
    - num_classes (int): 类别数（包括背景类）。
    - device (torch.device): 设备（CPU或GPU）。

    返回：
    - model (torch.nn.Module): 加载好的模型。
    """
    model = get_model(num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 加载完整检查点，提取 model_state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded model state_dict from checkpoint '{model_path}'.")
    else:
        # 直接加载 state_dict
        model.load_state_dict(checkpoint)
        logging.info(f"Loaded model state_dict from '{model_path}'.")

    model.to(device)
    model.eval()
    return model


def predict_and_save(model, test_loader, device, output_dir, no_detection_output_dir, predictions_dict,
                     score_threshold=0.5,
                     intersection_threshold=0.5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(no_detection_output_dir, exist_ok=True)  # 创建没有检测到框的图像输出文件夹

    for images, img_names in tqdm(test_loader, desc="Predicting"):
        images = list(image.to(device) for image in images)

        with torch.no_grad():
            predictions = model(images)

        for img_tensor, prediction, img_name in zip(images, predictions, img_names):
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()

            # 检查预测是否为空
            if len(boxes) == 0:
                logging.info(f"No predictions for {img_name}")
                # 保存原图到 no_detection_output_dir
                original_img_path = os.path.join(test_loader.dataset.images_dir, img_name)
                original_img = tiff.imread(original_img_path)
                if original_img.ndim == 3:
                    original_img = original_img[:, :, 0]  # 第一个通道
                elif original_img.ndim == 2:
                    pass  # 已经是单通道
                else:
                    raise ValueError(f"Unsupported image shape: {original_img.shape}")

                if np.max(original_img) > 0:
                    img_normalized = (original_img / np.max(original_img)) * 255
                else:
                    img_normalized = original_img
                img_normalized = img_normalized.astype(np.uint8)

                if img_normalized.ndim == 3:
                    img_normalized = img_normalized.squeeze()
                elif img_normalized.ndim == 1:
                    img_normalized = img_normalized[np.newaxis, :]
                elif img_normalized.ndim != 2:
                    raise ValueError(f"Cannot handle image with shape: {img_normalized.shape}")

                pil_img = Image.fromarray(img_normalized, mode='L')
                no_detection_annotated_img_path = os.path.join(no_detection_output_dir, f"no_detection_{img_name}")
                pil_img.save(no_detection_annotated_img_path)
                logging.info(f"No detection for {img_name}, saved to {no_detection_annotated_img_path}")

                # 记录预测结果为空
                predictions_dict[img_name] = {
                    'boxes': [],
                    'labels': [],
                    'scores': []
                }
                continue  # 跳过后续处理

            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            # NMS: 去除冗余框
            keep = nms(boxes_tensor, scores_tensor, intersection_threshold)

            # 将 keep 转换为 NumPy 数组
            keep = keep.cpu().numpy().astype(int)

            # 检查 keep 是否为空
            if len(keep) == 0:
                logging.info(f"No boxes kept after NMS for {img_name}")
                # 保存原图到 no_detection_output_dir
                original_img_path = os.path.join(test_loader.dataset.images_dir, img_name)
                original_img = tiff.imread(original_img_path)
                if original_img.ndim == 3:
                    original_img = original_img[:, :, 0]  # 第一个通道
                elif original_img.ndim == 2:
                    pass  # 已经是单通道
                else:
                    raise ValueError(f"Unsupported image shape: {original_img.shape}")

                if np.max(original_img) > 0:
                    img_normalized = (original_img / np.max(original_img)) * 255
                else:
                    img_normalized = original_img
                img_normalized = img_normalized.astype(np.uint8)

                if img_normalized.ndim == 3:
                    img_normalized = img_normalized.squeeze()
                elif img_normalized.ndim == 1:
                    img_normalized = img_normalized[np.newaxis, :]
                elif img_normalized.ndim != 2:
                    raise ValueError(f"Cannot handle image with shape: {img_normalized.shape}")

                pil_img = Image.fromarray(img_normalized, mode='L')
                no_detection_annotated_img_path = os.path.join(no_detection_output_dir, f"no_detection_{img_name}")
                pil_img.save(no_detection_annotated_img_path)
                logging.info(f"No detection after NMS for {img_name}, saved to {no_detection_annotated_img_path}")

                # 记录预测结果为空
                predictions_dict[img_name] = {
                    'boxes': [],
                    'labels': [],
                    'scores': []
                }
                continue  # 跳过后续处理

            # 保留经过 NMS 后的框、标签和分数
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            # 处理框之间的交集：如果两个框之间有交集，保留置信度更高的框
            filtered_boxes = []
            filtered_labels = []
            filtered_scores = []

            for i in range(len(boxes)):
                current_box = boxes[i]
                current_score = scores[i]
                current_label = labels[i]
                is_duplicated = False

                for j in range(len(filtered_boxes)):
                    # 计算两个框的交集面积
                    intersection_area = calculate_intersection_area(current_box, filtered_boxes[j])
                    if intersection_area > 0:  # 有交集
                        # 保留置信度更高的框
                        if current_score > filtered_scores[j]:
                            filtered_boxes[j] = current_box
                            filtered_labels[j] = current_label
                            filtered_scores[j] = current_score
                        is_duplicated = True
                        break

                if not is_duplicated:
                    filtered_boxes.append(current_box)
                    filtered_labels.append(current_label)
                    filtered_scores.append(current_score)

            # 过滤掉低于置信度阈值的框
            valid_boxes = []
            valid_labels = []
            valid_scores = []
            for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
                if score >= score_threshold:
                    valid_boxes.append(box)
                    valid_labels.append(label)
                    valid_scores.append(score)

            # 记录预测结果
            predictions_dict[img_name] = {
                'boxes': valid_boxes,
                'labels': valid_labels,
                'scores': valid_scores
            }

            # 处理图像并绘制
            original_img_path = os.path.join(test_loader.dataset.images_dir, img_name)
            original_img = tiff.imread(original_img_path)
            if original_img.ndim == 3:
                original_img = original_img[:, :, 0]  # 第一个通道
            elif original_img.ndim == 2:
                pass  # 已经是单通道
            else:
                raise ValueError(f"Unsupported image shape: {original_img.shape}")

            if np.max(original_img) > 0:
                img_normalized = (original_img / np.max(original_img)) * 255
            else:
                img_normalized = original_img
            img_normalized = img_normalized.astype(np.uint8)

            if img_normalized.ndim == 3:
                img_normalized = img_normalized.squeeze()
            elif img_normalized.ndim == 1:
                img_normalized = img_normalized[np.newaxis, :]
            elif img_normalized.ndim != 2:
                raise ValueError(f"Cannot handle image with shape: {img_normalized.shape}")

            pil_img = Image.fromarray(img_normalized, mode='L')
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()

            # 如果没有有效框，将图像保存到另一个文件夹
            if len(valid_boxes) == 0:
                no_detection_annotated_img_path = os.path.join(no_detection_output_dir, f"no_detection_{img_name}")
                pil_img.save(no_detection_annotated_img_path)
                logging.info(
                    f"No detection (after filtering) for {img_name}, saved to {no_detection_annotated_img_path}")
            else:
                # 绘制有效框
                for box, label, score in zip(valid_boxes, valid_labels, valid_scores):
                    draw.rectangle(box, outline="red", width=2)
                    text = f"Class {label}, {score:.2f}"
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle([box[0], box[1] - text_size[1], box[0] + text_size[0], box[1]], fill="red")
                    draw.text((box[0], box[1] - text_size[1]), text, fill="white", font=font)

                annotated_img_path = os.path.join(output_dir, f"annotated_{img_name}")
                pil_img.save(annotated_img_path)
                logging.info(f"Annotated image saved to {annotated_img_path}")


def evaluate_predictions(predictions, ground_truths, iou_threshold=0.2):
    """
    评估预测结果，计算每一类的Precision和Recall
    """
    # 获取所有类别
    classes = set()
    for gt_boxes in ground_truths.values():
        for ann in gt_boxes:
            classes.add(ann['category_id'])
    for pred_boxes in predictions.values():
        for label in pred_boxes['labels']:
            classes.add(label)
    classes = sorted(list(classes))

    # 初始化每个类别的TP, FP, FN
    metrics = {cls: {'TP': 0, 'FP': 0, 'FN': 0} for cls in classes}

    for img_name in ground_truths:
        gt = ground_truths[img_name]
        pred = predictions.get(img_name, {'boxes': [], 'labels': [], 'scores': []})

        gt_matched = [False] * len(gt)
        pred_matched = [False] * len(pred['boxes'])

        for pred_idx, (pred_box, pred_label) in enumerate(zip(pred['boxes'], pred['labels'])):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_ann in enumerate(gt):
                if gt_ann['category_id'] != pred_label:
                    continue
                iou = calculate_iou(pred_box, gt_ann['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                metrics[pred_label]['TP'] += 1
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True
            else:
                metrics[pred_label]['FP'] += 1

        # 计算FN
        for gt_idx, gt_ann in enumerate(gt):
            if not gt_matched[gt_idx]:
                metrics[gt_ann['category_id']]['FN'] += 1

    # 计算Precision和Recall
    results = {}
    for cls in classes:
        TP = metrics[cls]['TP']
        FP = metrics[cls]['FP']
        FN = metrics[cls]['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        results[cls] = {
            'Precision': precision,
            'Recall': recall
        }

    # 输出结果
    logging.info("Evaluation Results:")
    print("类别\tPrecision\tRecall")
    for cls in classes:
        print(f"{cls}\t{results[cls]['Precision']:.4f}\t\t{results[cls]['Recall']:.4f}")
        logging.info(f"Class {cls}: Precision={results[cls]['Precision']:.4f}, Recall={results[cls]['Recall']:.4f}")


def main():
    setup_logging()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"Using device: {device}")

    # 设置文件路径
    test_images_dir = r"C:\Users\w10\Desktop\CNN\data\images\val"
    output_dir = r"C:\Users\w10\Desktop\CNN\outputs\model s_V2"
    model_path = r"C:\Users\w10\Desktop\CNN\src\outputs\model s_5\best_model.pth"
    no_predict = r"C:\Users\w10\Desktop\CNN\outputs\no_predicts"
    coco_annotations_path = r"C:\Users\w10\Desktop\CNN\data\annotations\val_coco.json"

    # 加载模型
    num_classes = 6  # 确保与训练时一致
    model = load_model(model_path, num_classes=num_classes, device=device)

    logging.info("Model loaded and set to evaluation mode.")

    # 数据处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
    ])

    test_dataset = WeldDefectTestDataset(test_images_dir, transforms=transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 禁用多进程
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    logging.info(f"Number of test samples: {len(test_dataset)}")

    # 加载真实标签
    ground_truths = load_coco_annotations(coco_annotations_path)
    logging.info("Ground truth annotations loaded.")

    # 初始化预测结果字典
    predictions_dict = {}

    # 执行预测并保存结果
    predict_and_save(model, test_loader, device, output_dir, no_predict, predictions_dict)
    logging.info("Prediction and annotation completed.")

    # 评估预测结果
    evaluate_predictions(predictions_dict, ground_truths, iou_threshold=0.4)
    logging.info("Evaluation completed.")


if __name__ == "__main__":
    main()
