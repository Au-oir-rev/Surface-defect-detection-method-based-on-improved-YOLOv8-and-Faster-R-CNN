# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

class SelfAttention(nn.Module):
    """
    优化后的自注意力模块，适用于2D特征图，包含下采样以减少显存占用，并确保输出尺寸与输入一致。
    """

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # 减少中间通道数
        reduced_channels = max(1, in_channels // 32)  # 调整为更小的比例，防止除数为0
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        # 添加下采样和上采样
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 下采样因子为2
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样回原分辨率

    def forward(self, x):
        batch_size, C, H, W = x.size()
        original_size = (H, W)

        # 下采样
        x_down = self.downsample(x)
        _, _, H_down, W_down = x_down.size()
        N = H_down * W_down

        # 生成查询、键、值
        proj_query = self.query_conv(x_down).view(batch_size, -1, H_down * W_down).permute(0, 2, 1)  # [B, N, C']
        proj_key = self.key_conv(x_down).view(batch_size, -1, H_down * W_down)  # [B, C', N]
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N]
        attention = self.softmax(energy)  # [B, N, N]

        proj_value = self.value_conv(x_down).view(batch_size, -1, H_down * W_down)  # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch_size, C, H_down, W_down)  # [B, C, H_down, W_down]

        # 上采样回原分辨率
        out = self.upsample(out)  # [B, C, H*2, W*2]

        # 如果原始尺寸为奇数，调整上采样后的尺寸以匹配
        if out.size(2) != original_size[0] or out.size(3) != original_size[1]:
            out = F.interpolate(out, size=original_size, mode='nearest')

        out = self.gamma * out + x  # 残差连接
        return out


class CustomRPNHead(nn.Module):
    """
    自定义的 RPN 头，集成优化后的自注意力模块。
    """

    def __init__(self, in_channels, num_anchors):
        super(CustomRPNHead, self).__init__()
        # 原始的 RPN 头包含一个卷积层
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 添加优化后的自注意力模块
        self.self_attention = SelfAttention(in_channels)

        # 分类和回归层
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        # 初始化权重
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        """
        对每个特征图应用 RPN 头，并返回分类和回归结果的列表。

        Args:
            features (List[Tensor]): 来自 FPN 的特征图列表，每个特征图的形状为 [B, C, H, W]

        Returns:
            logits (List[Tensor]): 分类 logits 列表
            bbox_reg (List[Tensor]): 边界框回归参数列表
        """
        logits = []
        bbox_reg = []
        for x in features:
            x = self.conv(x)
            x = self.relu(x)
            x = self.self_attention(x)  # 应用优化后的自注意力
            logits.append(self.cls_logits(x))
            bbox_reg.append(self.bbox_pred(x))
        return logits, bbox_reg


def get_model(num_classes):
    """
    加载预训练的 Faster R-CNN 模型，替换分类头，并在 RPN 中添加优化后的自注意力机制。

    Args:
        num_classes (int): 类别数（包括背景类）。

    Returns:
        model (torch.nn.Module): 修改后的模型。
    """
    # 加载预训练模型，使用 'DEFAULT' 权重
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights,trainable_backbone_layers=5)
    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    '''
    # 获取 RPN 的参数以创建自定义 RPN 头
    # 假设所有 FPN 层的输出通道数相同
    in_channels = model.backbone.out_channels
    # 获取每个位置的 anchor 数量（假设所有 FPN 层的 anchor 数量相同）
    num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]

    # 替换 RPN 头为自定义的 RPN 头
    model.rpn.head = CustomRPNHead(in_channels, num_anchors)
    '''




    return model
