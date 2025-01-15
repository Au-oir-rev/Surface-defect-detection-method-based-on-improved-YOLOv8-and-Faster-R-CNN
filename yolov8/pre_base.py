from ultralytics import YOLO
import os
from PIL import Image

dataset_yaml = "./dataset/dataset.yaml"  # 数据集配置文件路径
# 2. 模型配置
model_config = "yolov8n.yaml"  # YOLOv8 模型配置文件路径 (e.g., yolov8n.yaml)
model_output = "./runs/detect/train"  # 模型保存路径

# 2. 裁剪图片和标签到 ROI


# 3. 从头训练模型
def train_yolo():
    # 3.1. 初始化自定义的 AFPN 模型
    # 3.7. 开始训练，使用裁剪后的数据
    model = YOLO("yolov8n.pt")
    model.train(
        data=dataset_yaml,  # 数据集配置文件路径
        epochs=500,  # 训练轮数
        imgsz=224,  # 输入图片大小
        val_imgsz=224,  # 验证图片大小
        batch=8,  # 批量大小
        device=0,  # 指定设备 (0 表示使用 GPU，"cpu" 表示使用 CPU)
        workers=4,  # 数据加载的线程数量
        project=model_output,  # 保存模型的项目目录
        name="custom_yolov8",  # 模型名称
        rect=True,  # 使用矩形训练模式（与 ROI 裁剪匹配）
        hsv_h=0.0,  # 关闭色相调整
        hsv_s=0.0,  # 关闭饱和度调整
        hsv_v=0.0,  # 关闭亮度调整
        scale=0.1,  # 缩放
        patience=150,
        cache=True,
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        mosaic=0.0,  # 启用 Mosaic 增强
        lr0=1e-4,  # 初始学习率
        lrf=0.2,  # 学习率衰减因子
    )

if __name__ == "__main__":

    train_yolo()
