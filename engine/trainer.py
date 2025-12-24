import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import os, logging
import numpy as np
from .evaluator import evaluate
from utils.general_utils import save_checkpoint


# ----------------------------
# utils
# ----------------------------
def _set_requires_grad(module, requires_grad: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = requires_grad


def _switch_stage_freeze(model, stage: int):
    """
    针对新版 CausalMTLModel 的冻结策略。
    Model 结构:
      - encoder, projector_s, projector_p (Shared)
      - head_seg, head_depth, head_normal (Task Heads, depend on Z_s)
      - decoder_app, albedo_head, etc. (Recon/Decomp, depend on Z_p)
    """
    # 基础检查，防止传入不兼容的模型 (如 RawMTL)
    if not hasattr(model, 'projector_p'):
        return

    # === Stage 0: 分解预热 (Decomposition Warmup) ===
    # 目标：强迫 Encoder 学习物理分解 (Z_p -> Albedo, Z_s -> Normal_phys)，
    #      而不被下游任务的梯度干扰。
    if stage == 0:
        # 1. 冻结下游任务头 (只接收 Z_s)
        _set_requires_grad(model.head_seg, False)
        _set_requires_grad(model.head_depth, False)
        _set_requires_grad(model.head_normal, False)

        # 2. 确保分解头和投影层是解冻的
        _set_requires_grad(model.projector_s, True)
        _set_requires_grad(model.projector_p, True)
        _set_requires_grad(model.albedo_head, True)
        _set_requires_grad(model.normal_head, True)
        _set_requires_grad(model.light_head, True)
        _set_requires_grad(model.decoder_app, True)

        logging.info("❄️ Stage-0: Decomposition Warmup. Task Heads FROZEN. Training Recon/Decomp only.")

    # === Stage 1 & 2: 联合训练 (Joint Training) ===
    # 解冻所有部分，开始训练下游任务
    else:
        _set_requires_grad(model.head_seg, True)
        _set_requires_grad(model.head_depth, True)
        _set_requires_grad(model.head_normal, True)

        # 确保其他部分也是解冻的
        _set_requires_grad(model.projector_s, True)
        _set_requires_grad(model.projector_p, True)

        logging.info(f"🔥 Stage-{stage}: Full Joint Training. All components UNFROZEN.")


def _get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg.get("lr", None)


def _set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _build_scheduler(optimizer, train_cfg):
    """
    自动构建调度器：Cosine 或 Step
    """
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    sched_cfg = train_cfg.get("lr_scheduler", {}) or {}
    sched_type = str(sched_cfg.get("type", "cosine")).lower()

    if sched_type == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 3))
        min_lr_factor = float(sched_cfg.get("min_lr_factor", 0.1))
        total_epochs = int(train_cfg.get("epochs", 30))
        t_max = int(sched_cfg.get("T_max", max(1, total_epochs - warmup_epochs)))
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=base_lr * min_lr_factor
        )
        return {
            "type": "cosine",
            "warmup_epochs": warmup_epochs,
            "base_lr": base_lr,
            "cosine": cosine
        }

    # fallback: StepLR
    step_size = int(sched_cfg.get("step_size", 100))
    gamma = float(sched_cfg.get("gamma", 0.5))
    step = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return {
        "type": "step",
        "step": step
    }


# ----------------------------
# train loops
# ----------------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, stage: int):
    model.train()
    total_train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]", leave=False)

    # 累积梯度步数
    target_bs = 16
    physical_bs = train_loader.batch_size
    accumulation_steps = max(1, target_bs // physical_bs)

    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(pbar):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        rgb = batch['rgb']

        # 前向传播 (FP32)
        outputs = model(rgb, stage=stage)

        crit_out = criterion(outputs, batch)

        # 解析 Loss 返回值
        if isinstance(crit_out, (tuple, list)):
            total_loss, loss_dict = crit_out[0], crit_out[1]
        elif isinstance(crit_out, dict):
            loss_dict = crit_out
            total_loss = loss_dict.get('total_loss')
        else:
            raise ValueError("criterion must return dict or (total_loss, dict).")

        # 打印最后一个 batch 的 loss
        if i == len(pbar) - 1:
            # 简化 log，避免刷屏
            simple_log = {k: float(f"{v:.4f}") for k, v in loss_dict.items() if
                          isinstance(v, (float, torch.Tensor)) and v > 0.001}
            logging.info(f"Epoch {epoch + 1} Loss: {simple_log}")

        loss_normalized = total_loss / accumulation_steps
        loss_normalized.backward()

        # 梯度更新
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 记录 Loss
        loss_val = total_loss.item()
        if not np.isfinite(loss_val):
            logging.info(f"⚠️ Warning: Non-finite loss {loss_val} at step {i}")

        total_train_loss += float(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    # 处理剩余梯度
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_train_loss = total_train_loss / max(1, len(train_loader))
    logging.info(f"Epoch {epoch + 1} - Avg Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def calculate_improvement(base_metrics, current_metrics, data_type='nyuv2'):
    """相对提升率计算"""
    improvement = 0
    count = 0
    # 1=越大越好, 0=越小越好
    metric_meta = {
        'seg_miou': 1, 'seg_pixel_acc': 1,
        'depth_abs_err': 0, 'depth_rel_err': 0,
        'normal_mean_angle': 0, 'normal_acc_30': 1,
        'normal_median_angle': 0, 'normal_acc_11': 1, 'normal_acc_22': 1
    }

    if 'gta5' in data_type:
        valid_keys = {'seg_miou', 'seg_pixel_acc'}
    elif data_type == 'cityscapes':
        valid_keys = {'seg_miou', 'seg_pixel_acc', 'depth_abs_err', 'depth_rel_err'}
    else:
        valid_keys = set(metric_meta.keys())

    for k, direction in metric_meta.items():
        if k not in valid_keys: continue
        if k in base_metrics and k in current_metrics:
            base = base_metrics[k]
            curr = current_metrics[k]
            if base == 0: continue

            if direction == 1:
                imp = (curr - base) / base
            else:
                imp = (base - curr) / base
            improvement += imp
            count += 1

    return improvement / max(1, count)


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, config, device,
          checkpoint_dir='checkpoints'):
    data_type = config['data'].get('type', 'nyuv2').lower()
    train_cfg = config['training']

    stage0_epochs = int(train_cfg.get('stage0_epochs', 0))
    stage1_epochs = int(train_cfg.get('stage1_epochs', 0))
    total_epochs = int(train_cfg.get('epochs', 30))
    base_lr = float(train_cfg.get("learning_rate", 1e-4))

    ind_warmup_epochs = int(train_cfg.get('ind_warmup_epochs', 0))
    target_ind_lambda = float(config['losses'].get('lambda_independence', 0.0))

    best_relative_score = -float('inf')
    baseline_metrics = None
    best_epoch = 0
    best_metrics_details = {}

    # [NEW] 计算 Stage 2 正式开始的 Epoch 索引
    stage2_start_epoch = stage0_epochs + stage1_epochs

    sched = _build_scheduler(optimizer, train_cfg)
    logging.info(f"[LR Scheduler] {sched['type']}; base_lr={base_lr}")

    for epoch in range(total_epochs):
        # --- Stage Logic ---
        if epoch < stage0_epochs:
            stage = 0
        elif epoch < stage1_epochs + stage0_epochs:
            stage = 1
        else:
            stage = 2

        # 每个 Epoch 检查是否需要切换冻结状态
        _switch_stage_freeze(model, stage)

        if epoch == 0:
            logging.info(f"🔍 [Training Strategy] Total Epochs: {total_epochs}")
            logging.info(f"   Stage 0 (Decomp Only): 0 -> {stage0_epochs}")
            logging.info(f"   Stage 2 (Joint Train): {stage0_epochs} -> {total_epochs}")
            logging.info(f"   Baseline will be FIXED at Epoch {stage2_start_epoch + 1} (Start of Stage 2)")

        # --- Lambda Decay/Warmup Strategy ---
        current_ind_lambda = target_ind_lambda

        # 在 Stage 0 强制 lambda 为 0
        if stage == 0:
            current_ind_lambda = 0.0
        # 如果还在 Warmup 期间
        elif ind_warmup_epochs > 0 and (epoch - stage0_epochs) < ind_warmup_epochs:
            progress = epoch - stage0_epochs
            ratio = min(1.0, max(0.0, progress / float(ind_warmup_epochs)))
            current_ind_lambda = target_ind_lambda * ratio
            logging.info(f"   > Ind Loss Warmup: {progress}/{ind_warmup_epochs} = {ratio:.2f}")

        # 更新 Criterion 中的权重
        real_criterion = criterion.module if hasattr(criterion, 'module') else criterion
        if hasattr(real_criterion, 'weights'):
            real_criterion.weights['lambda_independence'] = torch.tensor(current_ind_lambda, device=device)

        # ---- LR Warmup (Cosine only) ----
        if sched["type"] == "cosine":
            warmup_epochs = sched["warmup_epochs"]
            if epoch < warmup_epochs:
                warmup_start = 0.1 * base_lr
                ratio = float(epoch + 1) / float(max(1, warmup_epochs))
                lr_now = warmup_start + (base_lr - warmup_start) * ratio
                _set_lr(optimizer, lr_now)

        cur_lr = _get_lr(optimizer)
        logging.info(f"\n----- Epoch {epoch + 1}/{total_epochs} (Stage {stage}) | lr={cur_lr:.6f} -----")

        # --- Train & Validate ---
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, stage=stage)
        val_metrics = evaluate(model, val_loader, criterion, device, stage=stage, data_type=data_type)

        # --- Scheduler Step ---
        if sched["type"] == "cosine":
            if epoch >= sched["warmup_epochs"]:
                sched["cosine"].step()
        else:
            sched["step"].step()

        # --- Best Model Selection (Fixed Baseline Logic) ---
        is_best = False
        score = 0.0

        # Stage 0 & 1: 不计算 Score，跳过
        if stage < 2:
            logging.info(f"  -> Stage {stage} (Pre-training) - Skipping improvement calculation.")
            baseline_metrics = None  # 确保不使用预训练阶段作为基准

            # 依然保存 checkpoint，方便调试
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_relative_score,
            }, False, checkpoint_dir=checkpoint_dir)

        else:
            # Stage 2: 只有进入联合训练阶段才开始评测
            if epoch == stage2_start_epoch:
                # 刚进入 Stage 2 的第一轮 -> 强制锁定为 Baseline
                baseline_metrics = val_metrics
                logging.info(f"  -> 🏁 Stage 2 Started. Setting FIXED BASELINE from current epoch.")

                # 保存一份作为 Stage 2 起点的存档
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_score': 0.0,
                    'baseline_metrics': baseline_metrics,
                }, False, checkpoint_dir=checkpoint_dir, filename='checkpoint_stage2_start.pth.tar')

            elif baseline_metrics is not None:
                # Stage 2 后续轮次 -> 与锁定的 Baseline 比较
                score = calculate_improvement(baseline_metrics, val_metrics, data_type=data_type)
                is_best = (score > best_relative_score)

                if is_best:
                    best_relative_score = score
                    best_epoch = epoch + 1
                    best_metrics_details = val_metrics.copy()
                    logging.info(f"  -> 🏆 Best Model (vs Stage2-Start)! Score: {score:.2%}")

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_score': best_relative_score,
                    'baseline_metrics': baseline_metrics,
                }, is_best, checkpoint_dir=checkpoint_dir)
            else:
                # 防御性代码：如果是断点续训且没加载到 baseline，以当前为准
                logging.info("  -> Warning: No baseline found (resumed?), setting current as baseline.")
                baseline_metrics = val_metrics

    logging.info(f"\n✅ Training Finished. Best Epoch: {best_epoch}, Score: {best_relative_score:.2%}")