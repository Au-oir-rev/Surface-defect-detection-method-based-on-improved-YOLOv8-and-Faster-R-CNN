from ultralytics import YOLO
import os
from PIL import Image
import gc
import torch


# 1. 数据集配置
dataset_yaml = "./dataset/dataset.yaml"  # 数据集配置文件路径
# 2. 模型配置
model_config = "afpn_s.yaml"  # YOLOv8 模型配置文件路径 (e.g., yolov8n.yaml)
model_output = "./runs/detect/train"  # 模型保存路径

def train_yolo():
    # 初始化 YOLO 模型

    gc.collect()  # 释放Python层面不再引用的对象    torch.cuda.empty_cache()  # 释放已分配但未被使用的GPU缓存

    model = YOLO("./runs/detect/train/custom_yolov87/weights/last.pt")
    for param in model.parameters():
        param.requires_grad = True

    # 开始训练，
    model.train(
        data="./dataset/cropped/dataset.yaml",  # 使用裁剪后数据集的配置文件
        epochs=500,             # 训练轮数
        imgsz=224,             #
        val_imgsz=224,       # 新增加缩放验证集大小
        batch=8,                # 批量大小
        device=0,               # 指定设备 (0 表示使用 GPU，"cpu" 表示使用 CPU)
        workers=4,              # 数据加载的线程数量
        project=model_output,   # 保存模型的项目目录
        name="custom_yolov8",   # 模型名称,断点训练要加上对应的，而正常训练就是custom_yolov8就行
        rect=True,              # 使用矩形训练模式（与 ROI 裁剪匹配）
        hsv_h=0.0,              # 关闭色相调整
        hsv_s=0.0,              # 关闭饱和度调整
        hsv_v=0.0,              # 关闭亮度调整
        scale=0.1,              # 缩放
        resume=True,            # 断点训练
        exist_ok=True,           # 允许在已存在目录中继续训练
        patience = 150,
        cache=True,
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        mosaic=0.0,  # 启用 Mosaic 增强
    )


if __name__ == "__main__":
    train_yolo()
