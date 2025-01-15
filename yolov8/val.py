from ultralytics import YOLO

import matplotlib
from matplotlib import pyplot as plt

# 设置字体为 SimHei（黑体）
matplotlib.rc("font", family='SimHei')

# 验证是否加载成功

def validate_model():
    # 数据集配置
    dataset_yaml = "./dataset/dataset.yaml"  # 数据集配置文件路径
    model_weights = "./runs/detect/train/custom_yolov86/weights/best.pt"  # 已训练好的权重路径

    # 加载已训练的模型
    model = YOLO(model_weights)

    # 验证阶段
    model.val(
        data=dataset_yaml,  # 数据集配置文件
        imgsz=224,         # 验证阶段使用的分辨率
        val_imgsz=224,
        batch=16,            # 验证批量大小
        device=0,           # 使用 GPU（0 表示 GPU 0）
        conf=0.1,          # 置信度阈值
        iou=0.1,            # IoU 阈值   0.7
        save_json=True,     # 保存验证结果为 JSON 文件（COCO 格式）
        verbose=True,        # 输出详细日志信息
        cache=True,
    )

if __name__ == '__main__':
    validate_model()


#0.1,0.3    522
#