# src/utils.py

import torchvision.transforms as T

def get_transform(train):
    """
    获取图像转换函数。

    Args:
        train (bool): 是否为训练模式。

    Returns:
        transforms (torchvision.transforms.Compose): 图像转换序列。
    """
    transforms = []
    transforms.append(T.ToTensor())  # 将图像转换为张量，并归一化到[0,1]

    if train:
        # 添加数据增强（如随机水平翻转）
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)
