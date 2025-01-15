import torch
from ultralytics import YOLO

# 1. 加载指定尺寸的 YOLOv8 模型 (n, s, m, l, x)
model = YOLO("yolov8n.pt")

# 2. 准备一个测试输入 (batch_size=1, 3通道, 尺寸640x640)
dummy_input = torch.randn(1, 3, 640, 640)

# 3. 只通过骨干 (backbone) 获取特征图
#   注意：YOLOv8 内部模型通常分为 backbone + neck + head
features = model.model.backbone(dummy_input)

# 4. 打印各阶段特征图尺寸
for i, feat in enumerate(features):
    print(f"Feature {i} shape: {feat.shape}")
