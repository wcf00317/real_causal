# 文件: engine/evaluator.py
# 版本：【LibMTL NYUv2 Alignment Version - Cleaned & Fixed】

import os
import logging
import torch
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, stage, data_type,mask_zeros):
    """
    评估模型，并返回所有任务及重构任务的指标。
    """
    model.eval()

    # --- 1. 初始化所有指标对象 ---
    # [FIX] 优先从 config 获取类别数，兼容新旧模型架构
    if hasattr(model, 'config'):
        num_seg_classes = model.config.get('num_seg_classes', 40)
    elif hasattr(model, 'head_seg'):
        # CausalMTLModel (New): head_seg.head[-1] 是 Conv2d
        num_seg_classes = model.head_seg.head[-1].out_channels
    elif hasattr(model, 'predictor_seg'):
        # Legacy
        num_seg_classes = getattr(model.predictor_seg, 'output_channels', 40)
    else:
        num_seg_classes = 40  # Default fallback

    # [FIXED] ignore_index=-1 匹配 LibMTL 数据格式
    miou_metric = torchmetrics.classification.MulticlassJaccardIndex(
        num_classes=num_seg_classes, ignore_index=-1).to(device)

    pixel_acc_metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes=num_seg_classes, average='micro', ignore_index=-1).to(device)

    # LibMTL 对齐指标
    depth_metric = DepthMetric(mask_zeros=mask_zeros).to(device)
    normal_metric = NormalMetric().to(device)

    # --- 2. 跟踪损失 ---
    total_val_loss = 0.0
    total_recon_geom_loss = 0.0
    total_recon_app_loss = 0.0
    total_independence_loss = 0.0

    # CKA Tracking
    total_cka_ind = 0.0

    pbar = tqdm(val_loader, desc="Evaluating", leave=False)

    for batch in pbar:
        rgb = batch['rgb'].to(device)
        targets_on_device = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

        # 前向传播
        outputs = model(rgb, stage=stage)

        # 计算 Loss
        crit_out = criterion(outputs, targets_on_device)
        if isinstance(crit_out, (tuple, list)):
            _, loss_dict = crit_out
        else:
            loss_dict = crit_out

        # 累加 Loss
        total_val_loss += loss_dict.get('total_loss', torch.tensor(0.0)).item()
        total_recon_geom_loss += loss_dict.get('recon_geom_loss', torch.tensor(0.0)).item()
        total_recon_app_loss += loss_dict.get('recon_app_loss', torch.tensor(0.0)).item()
        total_independence_loss += loss_dict.get('independence_loss', torch.tensor(0.0)).item()

        # 累加 CKA (新版只有一个 independence_loss)
        total_cka_ind += loss_dict.get('independence_loss', torch.tensor(0.0)).item()

        # 更新任务指标
        if 'pred_seg' in outputs:
            miou_metric.update(outputs['pred_seg'], targets_on_device['segmentation'])
            pixel_acc_metric.update(outputs['pred_seg'], targets_on_device['segmentation'])

        # 法线指标更新 (注意：pred_normal 是新版 key)
        if 'normal' in targets_on_device and 'pred_normal' in outputs:
            normal_metric.update(outputs['pred_normal'], targets_on_device['normal'])
        elif 'normal' in targets_on_device and 'normals' in outputs:  # 兼容旧 Key
            normal_metric.update(outputs['normals'], targets_on_device['normal'])

        # 深度指标更新
        if 'depth' in targets_on_device and 'pred_depth' in outputs:
            depth_metric.update(outputs['pred_depth'], targets_on_device['depth'])

    # --- 3. 平均 ---
    num_batches = max(1, len(val_loader))
    avg_val_loss = total_val_loss / num_batches
    avg_recon_geom_loss = total_recon_geom_loss / num_batches
    avg_recon_app_loss = total_recon_app_loss / num_batches
    avg_independence_loss = total_independence_loss / num_batches
    avg_cka_ind = total_cka_ind / num_batches

    # --- 4. 任务指标计算 ---
    final_miou = miou_metric.compute().item()
    final_pixel_acc = pixel_acc_metric.compute().item()

    final_abs_err, final_rel_err = depth_metric.compute()

    if len(normal_metric.record) > 0:
        mean_angle, median_angle, acc_11, acc_22, acc_30 = normal_metric.compute()
    else:
        mean_angle, median_angle, acc_11, acc_22, acc_30 = 0.0, 0.0, 0.0, 0.0, 0.0

    # --- 5. 打印报告 (Log) ---
    logging.info("\n--- Validation Results ---")

    # 打印 Loss 概览
    log_loss = (f"Avg Loss: {avg_val_loss:.4f} | "
                f"Indep(CKA): {avg_independence_loss:.4f} | "
                f"Recon: G={avg_recon_geom_loss:.4f}, A={avg_recon_app_loss:.4f}")
    logging.info(log_loss)

    logging.info("-- Downstream Task Metrics --")

    # 1. Seg
    logging.info(f"[Seg   ] mIoU:     {final_miou:<7.4f} | Pixel Acc: {final_pixel_acc:<7.4f}")

    # 2. Depth
    # [MODIFIED] 始终打印深度指标，不再屏蔽 GTA5
    logging.info(f"[Depth ] Abs Err:  {final_abs_err:<7.4f} | Rel Err:   {final_rel_err:<7.4f}")

    # 3. Normal
    if 'nyuv2' in str(data_type).lower():
        logging.info(f"[Normal] Mean Ang: {mean_angle:<7.4f}° | Median:    {median_angle:<7.4f}°")
        logging.info(f"         Acc 11°:  {acc_11:<7.4f} | Acc 22°:   {acc_22:<7.4f} | Acc 30°: {acc_30:<7.4f}")
    logging.info("-" * 60)

    # Reset metrics
    miou_metric.reset()
    pixel_acc_metric.reset()
    depth_metric.reset()
    normal_metric.reset()

    return {
        'val_loss': avg_val_loss,
        'recon_geom_loss': avg_recon_geom_loss,
        'recon_app_loss': avg_recon_app_loss,
        'seg_miou': final_miou,
        'seg_pixel_acc': final_pixel_acc,
        'depth_abs_err': final_abs_err,
        'depth_rel_err': final_rel_err,
        'normal_mean_angle': mean_angle,
        'normal_median_angle': median_angle,
        'normal_acc_11': acc_11,
        'normal_acc_22': acc_22,
        'normal_acc_30': acc_30
    }


# ==========================================================
# [NEW] LibMTL Aligned Metric Classes (for Depth & Normal)
# ==========================================================

class AbsMetric(object):
    """LibMTL AbsMetric 抽象基类"""

    def __init__(self):
        self.bs = []

    def update(self, *args):
        self.update_fun(*args)

    def compute(self):
        return self.score_fun()

    def to(self, device):
        return self

    def reinit(self):
        self.bs = []
        if hasattr(self, 'abs_record'): self.abs_record = []
        if hasattr(self, 'rel_record'): self.rel_record = []
        if hasattr(self, 'record'): self.record = []

    def reset(self):
        self.reinit()


class DepthMetric(AbsMetric):
    """
    对齐 LibMTL 的 DepthMetric，计算 Abs Err (MAE) 和 Rel Err。
    """

    def __init__(self,mask_zeros):
        super(DepthMetric, self).__init__()
        self.abs_record = []
        self.rel_record = []
        self.bs = []
        self.mask_zeros=mask_zeros

    def update_fun(self, pred, gt):
        # 1. 根据保存的策略生成 Mask
        if self.mask_zeros:
            # Cityscapes / NYUv2: 必须过滤 0 (通常 > 0.001)
            valid_mask = (gt > 1e-3)
        else:
            # GTA5: 全图有效
            valid_mask = torch.ones_like(gt, dtype=torch.bool)

        # 全图无效则跳过
        if valid_mask.sum() == 0:
            return

        # 2. 提取像素
        p = pred[valid_mask]
        g = gt[valid_mask]

        # 3. 计算误差
        abs_err = torch.abs(p - g)

        # 相对误差计算
        if self.mask_zeros:
            # 既然已经过滤了 > 1e-3，直接除 g 即可，绝对安全
            rel_err = torch.abs(p - g) / g
        else:
            # GTA 模式下 g 可能为 0，必须 clamp 分母防止除 0
            rel_err = torch.abs(p - g) / torch.clamp(g, min=1e-6)

        self.abs_record.append(abs_err.mean().item())
        self.rel_record.append(rel_err.mean().item())
        self.bs.append(p.numel())

    def score_fun(self):
        if not self.bs:
            return [0.0, 0.0]

        records = np.stack([np.array(self.abs_record), np.array(self.rel_record)])
        batch_size = np.array(self.bs)

        total_pixels = sum(batch_size)
        if total_pixels == 0:
            return [0.0, 0.0]

        # 计算加权平均 (误差 * 像素数 / 总像素数)
        weighted_abs_err = (records[0] * batch_size).sum() / total_pixels
        weighted_rel_err = (records[1] * batch_size).sum() / total_pixels

        return [float(weighted_abs_err), float(weighted_rel_err)]


class NormalMetric(AbsMetric):
    """
    对齐 LibMTL 的 NormalMetric，计算角度误差指标。
    """

    def __init__(self):
        super(NormalMetric, self).__init__()
        self.record = []  # 记录所有有效像素的角度误差 (度)

    def update_fun(self, pred, gt):
        # pred, gt 形状应为 [B, 3, H, W]

        # 1. 法线归一化 (pred)
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)

        # 2. 掩码
        binary_mask = (torch.sum(gt, dim=1) != 0)

        # 3. 计算点积 (cos(theta))
        dot_product = torch.sum(pred * gt, 1).masked_select(binary_mask)

        # 4. 角度误差 (acos(dot_product))
        error_rad = torch.acos(torch.clamp(dot_product, -1, 1))

        # 转换为角度 (度)
        error_deg = torch.rad2deg(error_rad).detach().cpu().numpy()

        self.record.append(error_deg)

    def score_fun(self):
        if not self.record:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        records = np.concatenate(self.record)

        # 5. 计算指标
        mean_angle = np.mean(records)
        median_angle = np.median(records)

        # 准确率 (Acc@T)
        acc_11 = np.mean((records < 11.25) * 1.0)
        acc_22 = np.mean((records < 22.5) * 1.0)
        acc_30 = np.mean((records < 30) * 1.0)

        return [float(mean_angle), float(median_angle), float(acc_11), float(acc_22), float(acc_30)]