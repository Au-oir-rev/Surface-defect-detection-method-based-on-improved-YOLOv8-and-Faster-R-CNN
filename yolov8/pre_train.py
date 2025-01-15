from ultralytics import YOLO
import torch

dataset_yaml = "./dataset/dataset.yaml"  # 数据集配置文件路径
model_config = "afpn.yaml"  # 自定义 YOLOv8 模型配置文件路径
model_output = "./runs/detect/train"  # 模型保存路径

def train_yolo():
    # 1. 初始化自定义模型
    #custom_model = YOLO(model_config).load("yolov8n.pt")  # 使用自定义配置文件初始化模型
    model = YOLO('afpn.yaml')
    model.load('yolov8n.pt')
    for param in model.parameters():
        param.requires_grad = True

    # 8. 开始训练
    model.train(
        data=dataset_yaml,       # 数据集配置文件路径
        epochs=500,              # 训练轮数
        imgsz=224,               # 输入图片大小
        val_imgsz=224,
        batch=8,                 # 批量大小
        device=0,                # 指定设备 (0 表示使用 GPU，"cpu" 表示使用 CPU)
        workers=4,               # 数据加载的线程数量
        project=model_output,    # 保存模型的项目目录
        name="custom_yolov8",    # 模型名称
        rect=True,               # 使用矩形训练模式（与 ROI 裁剪匹配）
        hsv_h=0.0,               # 关闭色相调整
        hsv_s=0.0,               # 关闭饱和度调整
        hsv_v=0.0,               # 关闭亮度调整
        scale=0.1,               # 缩放
        patience=150,
        freeze=0,
        cache=True,
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
    )
    print("Training started.")

if __name__ == "__main__":
    train_yolo()
