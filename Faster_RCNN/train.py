# src/train.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import WeldDefectDataset
from model import get_model
from utils import get_transform
import logging
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import csv  # 新增：用于处理CSV文件
from torch import nn

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, preds, targets):
        inter_x1 = torch.max(preds[:, 0], targets[:, 0])
        inter_y1 = torch.max(preds[:, 1], targets[:, 1])
        inter_x2 = torch.min(preds[:, 2], targets[:, 2])
        inter_y2 = torch.min(preds[:, 3], targets[:, 3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        area_preds = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
        area_targets = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
        union_area = area_preds + area_targets - inter_area

        iou = inter_area / union_area.clamp(min=1e-6)
        loss = 1 - iou
        return loss.mean()

def collate_fn(batch):
    """自定义的 collate 函数，用于 DataLoader。"""
    return tuple(zip(*batch))


def setup_logging(log_dir):
    """
    设置日志记录。

    参数：
    - log_dir (str): 日志文件目录。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def calculate_iou(pred_box, true_box):
    """
    计算预测框与真实框的IoU（Intersection over Union）。

    参数：
    - pred_box (tuple/list): 预测的边界框，格式为 (x_min, y_min, x_max, y_max)
    - true_box (tuple/list): 真实的边界框，格式为 (x_min, y_min, x_max, y_max)

    返回：
    - float: IoU值
    """
    x_min_inter = max(pred_box[0], true_box[0])
    y_min_inter = max(pred_box[1], true_box[1])
    x_max_inter = min(pred_box[2], true_box[2])
    y_max_inter = min(pred_box[3], true_box[3])

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection_area = inter_width * inter_height

    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
    union_area = pred_area + true_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def evaluate_precision_recall(
    model,
    data_loader,
    device,
    iou_threshold=0.5,
    confidence_threshold=0.5
):
    """
    在验证集上评估模型性能，计算 Precision 和 Recall。

    参数：
    - model (torch.nn.Module): 训练中的模型。
    - data_loader (DataLoader): 验证集的数据加载器。
    - device (torch.device): 设备（CPU或GPU）。
    - iou_threshold (float): IoU阈值，用于确认TP。
    - confidence_threshold (float): 置信度阈值，过滤低置信度的预测。

    返回：
    - precision (float)
    - recall (float)
    """
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                # 取出预测框
                pred_boxes = output["boxes"]
                pred_scores = output["scores"]
                pred_labels = output["labels"]  # 如果需要区分类别，也可以加以利用

                # 过滤预测分数小于 confidence_threshold 的框
                keep_idx = pred_scores >= confidence_threshold
                pred_boxes = pred_boxes[keep_idx]
                pred_scores = pred_scores[keep_idx]
                pred_labels = pred_labels[keep_idx]

                # 取出真实框
                gt_boxes = target["boxes"]
                # 如果需要使用标签做更严格的匹配，也可取 gt_labels = target["labels"]

                # 对每个预测框找出是否有匹配的真实框（IoU>=阈值）
                matched_gt = set()  # 存放已匹配的 gt box 索引，防止一个 GT 被多个预测重复匹配
                for pbox in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        iou = calculate_iou(pbox, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    # 如果满足 IoU 阈值，并且该 gt 尚未被匹配过，则记为 TP，否则记为 FP
                    if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_fp += 1

                # 未被匹配到的 GT 记为 FN
                total_fn += (len(gt_boxes) - len(matched_gt))

    # 计算 Precision 和 Recall
    precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )

    model.train()
    return precision, recall


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    scaler,
    epoch,
    writer,
    grad_clip=None,
    lambda_iou=1.0,  # IoU Loss 的权重
):
    """
    训练模型的一个epoch。
    增加 IoU 损失，与原有的回归损失（Smooth L1）结合。
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

    iou_criterion = IoULoss().to(device)  # 我们在代码顶部已经定义的 IoU 损失类

    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # loss_dict 包含:
            # 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'
            loss_dict = model(images, targets)

            # 原始回归损失
            loss_box_reg = loss_dict.get('loss_box_reg', 0.0)

            # 这里示例只演示在 ROI Head 上计算 IoU
            # 如果要结合 RPN 的回归，也可类似处理
            iou_loss_val = 0.0
            # 尝试获取预测框和真实框来计算 IoU Loss
            # 由于官方代码在训练模式下不返回预测框，
            # 我们需要通过替换/改写 forward 或者做一些 trick 获取
            # 这里给出“伪代码”思路，如果您想要更准确地实现，
            # 需要深入修改或在 model forward 里获取解码后的 bbox。

            # ---------- 伪代码示例开始 ------------
            # pred_boxes_batch, gt_boxes_batch = _extract_boxes_for_iou(model, images, targets)
            # if pred_boxes_batch is not None and gt_boxes_batch is not None:
            #     iou_loss_val = iou_criterion(pred_boxes_batch, gt_boxes_batch)
            # ---------- 伪代码示例结束 ------------

            # 此处就简单用 0 模拟
            # 如果无法轻易获取预测框，可考虑先用 RPN 里 proposals 与 targets 的 IoU 做个近似
            # 或仅仅在 ROI Head forward 里把解码后的框保存下来
            iou_loss_val = 0.0  # 占位

            # 合并回归损失 + IoU Loss
            total_box_loss = loss_box_reg + lambda_iou * iou_loss_val

            # 替换 loss_box_reg
            losses = 0.0
            for k, v in loss_dict.items():
                if k == 'loss_box_reg':
                    losses += total_box_loss
                else:
                    losses += v

        scaler.scale(losses).backward()

        if grad_clip:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        epoch_loss += losses.item()
        num_batches += 1
        progress_bar.set_postfix({'Loss': losses.item()})

    avg_loss = epoch_loss / num_batches
    logging.info(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")
    writer.add_scalar('Train/Loss', avg_loss, epoch + 1)

    return avg_loss



def save_checkpoint(state, filename):
    """
    保存训练检查点。

    参数：
    - state (dict): 要保存的状态字典。
    - filename (str): 检查点文件路径。
    """
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}.")


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint_path,
    device
):
    """
    从检查点加载训练状态。

    参数：
    - model (torch.nn.Module): 模型。
    - optimizer (torch.optim.Optimizer): 优化器。
    - scheduler (torch.optim.lr_scheduler): 学习率调度器。
    - scaler (GradScaler): 混合精度的梯度缩放器。
    - checkpoint_path (str): 检查点文件路径。
    - device (torch.device): 设备（CPU或GPU）。

    返回：
    - int: 恢复的epoch数。
    - float: 恢复的最佳F1（或其他评估标准）。
    - list: 训练损失的历史记录。
    - list: 验证precision历史记录。
    - list: 验证recall历史记录。
    """
    if os.path.isfile(checkpoint_path):
        logging.info(f"Loading checkpoint '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        train_losses = checkpoint.get('train_losses', [])
        val_precision_scores = checkpoint.get('val_precision_scores', [])
        val_recall_scores = checkpoint.get('val_recall_scores', [])
        logging.info(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        return start_epoch, best_score, train_losses, val_precision_scores, val_recall_scores
    else:
        logging.warning(f"No checkpoint found at '{checkpoint_path}'. Starting from scratch.")
        return 0, 0.0, [], [], []


def parse_args():
    """
    解析命令行参数。

    返回：
    - argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description="Weld Defect Detection Training Script")
    parser.add_argument('--train_images_dir', type=str,
                        default="C:/Users/w10/Desktop/CNN/data/images/train",
                        help='路径到训练图像目录')
    parser.add_argument('--train_annotations_file', type=str,
                        default="C:/Users/w10/Desktop/CNN/data/annotations/train_coco.json",
                        help='路径到训练注释文件')
    parser.add_argument('--val_images_dir', type=str,
                        default="C:/Users/w10/Desktop/CNN/data/images/val",
                        help='路径到验证图像目录')
    parser.add_argument('--val_annotations_file', type=str,
                        default="C:/Users/w10/Desktop/CNN/data/annotations/val_coco.json",
                        help='路径到验证注释文件')
    parser.add_argument('--num_classes', type=int, default=6, help='类别数（包括背景）')
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作线程数')
    parser.add_argument('--num_epochs', type=int, default=10000, help='训练周期数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减参数')
    parser.add_argument('--early_stop_patience', type=int, default=800, help='早停耐心')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--log_dir', type=str, default='logs_5', help='日志目录')
    parser.add_argument('--model_save_dir', type=str, default='outputs/model s_5', help='模型保存目录')
    parser.add_argument('--metrics_save_dir', type=str, default='outputs/metrics s_5', help='指标保存目录')
    parser.add_argument('--save_every', type=int, default=50, help='每多少个epoch保存一次模型')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 设置日志
    setup_logging(args.log_dir)

    # 初始化 TensorBoard
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, current_time))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 创建保存目录
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.metrics_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 加载数据集
    train_dataset = WeldDefectDataset(
        args.train_images_dir,
        args.train_annotations_file,
        transforms=get_transform(train=True)
    )
    val_dataset = WeldDefectDataset(
        args.val_images_dir,
        args.val_annotations_file,
        transforms=get_transform(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    logging.info(f"Number of training samples: {len(train_dataset)}")
    logging.info(f"Number of validation samples: {len(val_dataset)}")

    # 加载模型
    model = get_model(args.num_classes)
    model.to(device)
    logging.info("Model loaded and moved to device.")

    # 使用 AdamW 优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 学习率调度器（以最大化某指标为目标，这里用F1作为step的监控指标）
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # 混合精度
    scaler = GradScaler()

    # 初始化训练状态
    start_epoch = 0
    # 在这里我们用 F1 作为衡量 “最佳模型” 的标准，也可以根据需要改成纯 precision 或纯 recall
    best_score = 0.0
    train_losses = []
    val_precision_scores = []
    val_recall_scores = []




    # 如果有 last_model.pth，就自动加载
    last_model_path = os.path.join(args.model_save_dir, "last_model.pth")
    if os.path.isfile(last_model_path):
        (start_epoch,
         best_score,
         train_losses,
         val_precision_scores,
         val_recall_scores) = load_checkpoint(
            model, optimizer, lr_scheduler, scaler, last_model_path, device
        )

    # 定义两个固定名字的模型文件
    best_model_path = os.path.join(args.model_save_dir, "best_model.pth")
    last_model_path = os.path.join(args.model_save_dir, "last_model.pth")

    # 定义 CSV 文件路径
    metrics_csv_path = os.path.join(args.metrics_save_dir, "metrics.csv")

    # 初始化或续写 CSV 文件
    if start_epoch > 0 and os.path.isfile(metrics_csv_path):
        # 假设 metrics.csv 已经包含了之前的训练数据，直接追加新数据
        pass
    else:
        # 创建新的 CSV 文件并写入表头
        with open(metrics_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['epoch', 'train_loss', 'precision', 'recall', 'f1'])

    epochs_no_improve = 0

    try:
        for epoch in range(start_epoch, args.num_epochs):
            logging.info(f"Epoch {epoch + 1}/{args.num_epochs} started.")

            # 训练一个epoch
            train_loss = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                scaler,
                epoch,
                writer,
                grad_clip=args.grad_clip
            )

            # 每个epoch结束后进行验证（Precision 和 Recall）
            precision, recall = evaluate_precision_recall(
                model,
                val_loader,
                device,
                iou_threshold=0.5,
                confidence_threshold=0.5
            )
            # 计算 F1
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            logging.info(f"New best model found with F1: {f1:.4f}. "
                         f"precision ：{precision:.4f}."
                         f"recall ：{recall:.4f}.")

            # 记录到 TensorBoard
            writer.add_scalar('Validation/Precision', precision, epoch + 1)
            writer.add_scalar('Validation/Recall', recall, epoch + 1)
            writer.add_scalar('Validation/F1', f1, epoch + 1)

            # 更新学习率调度器
            lr_scheduler.step(f1)

            # 保存训练损失和验证 P/R 到 CSV
            train_losses.append(train_loss)
            val_precision_scores.append(precision)
            val_recall_scores.append(recall)
            with open(metrics_csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([epoch + 1, train_loss, precision, recall, f1])
            logging.info(f"Training metrics saved to {metrics_csv_path}.")

            # 每个 epoch 都更新 last_model.pth
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_score': best_score,  # 保存历史最佳F1
                'train_losses': train_losses,
                'val_precision_scores': val_precision_scores,
                'val_recall_scores': val_recall_scores
            }
            save_checkpoint(checkpoint_state, last_model_path)

            # 若当前 f1 更优，则更新 best_model.pth
            if f1 > best_score:
                best_score = f1
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"New best model found with F1: {best_score:.4f}. "
                             f"Model saved to {best_model_path}.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement in F1 for {epochs_no_improve} epoch(s).")

            # 如果满足 save_every，则额外保存一下带 epoch 标记的模型（可选）
            if (epoch + 1) % args.save_every == 0:
                epoch_model_path = os.path.join(
                    args.model_save_dir,
                    f"model_epoch_{epoch + 1}_{current_time}.pth"
                )
                save_checkpoint(checkpoint_state, epoch_model_path)

            # 早停判断
            if epochs_no_improve >= args.early_stop_patience:
                logging.info("Early stopping triggered.")
                break

    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving current model and metrics...")

        # 保存中断时的检查点
        epoch_model_path = os.path.join(
            args.model_save_dir,
            f"interrupted_model_epoch_{epoch + 1}_{current_time}.pth"
        )
        save_checkpoint(checkpoint_state, epoch_model_path)

        # 保存训练损失和验证指标到 CSV
        with open(metrics_csv_path, mode='a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([epoch + 1, train_loss, precision, recall, f1])
        logging.info(f"Training metrics saved to {metrics_csv_path}.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise e

    else:
        logging.info("Training completed successfully.")

    finally:
        writer.close()
        logging.info("Exiting training process.")


if __name__ == "__main__":
    main()
