import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Project Modules ---
# 确保这些模块路径在您的项目中是正确的
from models.causal_celeba_model import CausalCelebAModel
from data_utils.celeba_dataset import CelebADataset
from utils.general_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Subspace Orthogonality (Zs vs Zp)")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to model checkpoint")
    parser.add_argument('--output', type=str, default='orthogonality_check.png', help="Output plot filename")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda/cpu)")
    parser.add_argument('--batches', type=int, default=10, help="Number of batches to evaluate")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. Load Config
    print(f"📂 Loading config: {args.config}")
    config = load_config(args.config)

    # [Fix] 强制设置 num_attributes 为 40，以匹配 Checkpoint 的权重形状
    # 这是为了解决之前遇到的权重加载报错问题
    print("🔧 Forcing num_attributes=40 to match standard CelebA checkpoint...")
    config['model']['num_attributes'] = 40

    set_seed(config['training'].get('seed', 2024))

    # 2. Dataset
    data_cfg = config['data']
    print(f"📚 Initializing CelebA Dataset (Val split)...")
    # 这里强制 num_attributes=40，确保数据与模型头对齐
    dataset = CelebADataset(
        root_dir=data_cfg['dataset_path'],
        split='val',
        img_size=data_cfg.get('img_size', [128, 128]),
        num_attributes=40,
        augmentation=False
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 3. Model
    print("🧠 Initializing CausalCelebAModel...")
    model = CausalCelebAModel(config['model']).to(device)

    # 4. Load Checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📥 Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ No checkpoint provided! Results will be random.")

    model.eval()

    # Losses
    criterion_attr = nn.BCEWithLogitsLoss()
    criterion_recon = nn.L1Loss()

    cosine_sims = []

    print(f"⚡ Running orthogonality check on {args.batches} batches...")

    for i, batch in enumerate(tqdm(loader, total=args.batches)):
        if i >= args.batches:
            break

        imgs = batch['image'].to(device)
        attrs = batch['attributes'].to(device)  # [B, 40]

        # 即使在 eval 模式下，也启用梯度计算，因为我们需要对 input 求导（或特征层求导）
        with torch.set_grad_enabled(True):
            imgs.requires_grad_(True)

            # --- 手动分步前向传播 ---
            # 1. Encoder 提取特征
            feat_map = model.encoder(imgs)
            feat_map.retain_grad()  # 关键步骤：保留中间层特征的梯度

            # 2. 分支投影
            zs_map = model.proj_zs(feat_map)
            zp_map = model.proj_zp(feat_map)

            # 3. 属性任务 (Zs 路径)
            zs_vec = F.adaptive_avg_pool2d(zs_map, (1, 1)).flatten(1)
            pred_attr = model.attr_head(zs_vec)

            # 4. 重构任务 (Zs + Zp 路径)
            z_combined = torch.cat([zs_map, zp_map], dim=1)
            recon_img = model.decoder(z_combined)

            # --- 梯度计算 ---

            # A. 计算 Attribute Loss 对 Encoder 特征的梯度
            model.zero_grad()
            if feat_map.grad is not None: feat_map.grad.zero_()

            loss_attr = criterion_attr(pred_attr, attrs)
            loss_attr.backward(retain_graph=True)  # 保留图以进行第二次 backward

            grad_attr = feat_map.grad.clone().detach()  # [B, C, H, W]

            # B. 计算 Recon Loss 对 Encoder 特征的梯度
            model.zero_grad()
            feat_map.grad.zero_()

            loss_recon = criterion_recon(recon_img, imgs)
            loss_recon.backward()

            grad_recon = feat_map.grad.clone().detach()  # [B, C, H, W]

            # --- 计算余弦相似度 ---
            # 展平为 [B, D]
            g_a_flat = grad_attr.view(grad_attr.size(0), -1)
            g_r_flat = grad_recon.view(grad_recon.size(0), -1)

            # 计算两组梯度的余弦相似度
            # sim = (A . B) / (|A| |B|)
            sim = F.cosine_similarity(g_a_flat, g_r_flat, dim=1, eps=1e-8)
            cosine_sims.append(sim.cpu().numpy())

    # 汇总所有 Batch
    all_sims = np.concatenate(cosine_sims)
    mean_sim = np.mean(all_sims)
    std_sim = np.std(all_sims)

    print(f"\n📊 Results:")
    print(f"  Mean Cosine Similarity: {mean_sim:.4f}")
    print(f"  Std Dev: {std_sim:.4f}")
    print(f"  (Ideal value is close to 0, indicating orthogonality)")

    # 5. 可视化
    plt.figure(figsize=(10, 6))
    sns.histplot(all_sims, bins=50, kde=True, color='purple', alpha=0.6)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Orthogonality')
    plt.axvline(mean_sim, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean_sim:.4f}')

    plt.title("Subspace Orthogonality Check: $\\nabla \mathcal{L}_{attr}$ vs $\\nabla \mathcal{L}_{recon}$",
              fontsize=16)
    plt.xlabel("Cosine Similarity", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"✅ Plot saved to {args.output}")


if __name__ == '__main__':
    main()