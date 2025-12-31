import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools

# --- 项目模块导入 ---
from models.causal_model import CausalMTLModel
from data_utils.gta5_dataset import GTA5Dataset
from data_utils.cityscapes_dataset import CityscapesDataset
from utils.general_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Domain Intervention: GTA5 Structure + Cityscapes Style")
    parser.add_argument('--config', type=str, required=True, help="Config file path")
    parser.add_argument('--checkpoint', type=str, required=True, help="Model checkpoint path")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    parser.add_argument('--samples', type=int, default=500, help="Number of samples to evaluate")
    return parser.parse_args()


def relative_l2_distance(z_orig, z_new):
    """
    计算相对 L2 漂移: ||z_new - z_orig|| / ||z_orig||
    """
    B = z_orig.shape[0]
    z_o_flat = z_orig.view(B, -1)
    z_n_flat = z_new.view(B, -1)

    diff = torch.norm(z_o_flat - z_n_flat, p=2, dim=1)
    base = torch.norm(z_o_flat, p=2, dim=1) + 1e-8

    return (diff / base).cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Cross-Domain Intervention (GTA5 -> Cityscapes) on {device}")

    # 1. Load Config & Model
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    set_seed(config['training']['seed'])

    print("🧠 Loading CausalMTLModel...")
    model = CausalMTLModel(config['model'], config['data']).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. Prepare Two Datasets
    print("📚 Initializing Source (GTA5) & Target (Cityscapes)...")

    # Source: GTA5 (提供结构 Zs)
    ds_src = GTA5Dataset(
        root_dir=config['data']['train_dataset_path'],
        img_size=config['data']['img_size'],
        augmentation=False
    )

    # Target: Cityscapes (提供风格 Zp)
    # 使用 val_dataset_path 因为它通常在 config 里配好了
    ds_tgt = CityscapesDataset(
        root_dir=config['data']['val_dataset_path'],
        split='val'
    )

    loader_src = DataLoader(ds_src, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    loader_tgt = DataLoader(ds_tgt, batch_size=16, shuffle=True, num_workers=4, drop_last=True)

    # 3. Metrics Storage
    delta_zs_list = []  # 结构稳定性 (越小越好)
    delta_zp_drift_list = []  # 风格响应性 (相对于原图风格的变化，越大越好)
    delta_zp_consistency_list = []  # 风格一致性 (相对于目标风格的距离，越小越好 - 可选指标)

    count = 0
    print(f"\n⚡ Starting Cross-Domain Intervention Loop (Max {args.samples} samples)...")

    # 使用 zip 同时遍历两个数据集
    with torch.no_grad():
        for batch_src, batch_tgt in tqdm(zip(loader_src, loader_tgt), total=args.samples // 16):
            if count >= args.samples:
                break

            # --- A. 准备数据 ---
            # Image A (GTA5)
            img_a = batch_src['rgb'].to(device)
            # Image B (Cityscapes)
            img_b = batch_tgt['rgb'].to(device)

            B_size = img_a.size(0)
            if img_b.size(0) != B_size: break  # 最后一批对齐

            # --- B. 提取特征 ---
            # 1. Source (GTA5) -> Get Zs_A, Zp_A
            out_a = model(img_a, stage=2)
            zs_a_orig = out_a['z_s_map']
            zp_a_orig = out_a['z_p']  # Vector

            # 2. Target (Cityscapes) -> Get Zp_B
            out_b = model(img_b, stage=2)
            zp_b_target = out_b['z_p']  # Vector

            # --- C. 跨域干预 (Cross-Domain Intervention) ---
            # 生成: Structure from GTA5 + Style from Cityscapes
            # 注意: generate_counterfactual_image 需要 z_p 为 map 或 vector，根据你的实现调整
            # 这里的 zp_b_target 是 vector [B, C]，我们需要传给 decoder

            # 我们的 model.generate_counterfactual_image 内部逻辑是:
            # I_cfa = decoder_app(z_s, z_p_vec)
            # 所以我们不需要手动 shuffle，直接传入 batch B 的 Zp 即可
            # (这意味着 img_a[i] 将获得 img_b[i] 的风格)

            # 直接调用 decoder (比用 generate_counterfactual_image 更直接，因为那个函数内部有 shuffle 逻辑)
            # 我们手动模拟 "Mixing":
            recon_logits, _ = model.decoder_app(zs_a_orig, zp_b_target)
            I_cfa = model.final_app_activation(recon_logits)

            # --- D. 重编码 (Re-encoding) ---
            out_cfa = model(I_cfa, stage=2)
            zs_new = out_cfa['z_s_map']
            zp_new = out_cfa['z_p']

            # --- E. 测量指标 ---

            # 1. Metric: Structure Stability (Zs A vs Zs New)
            # 即使换了 Cityscapes 的皮，GTA5 的骨架还在吗？
            d_zs = relative_l2_distance(zs_a_orig, zs_new)
            delta_zs_list.append(d_zs)

            # 2. Metric: Style Responsiveness (Zp A vs Zp New)
            # 新的风格(Cityscapes)是否这就导致它离原来的风格(GTA5)非常远？
            # 预期：这个值应该很大，因为 GTA5 和 CS 风格差异巨大
            d_zp_drift = relative_l2_distance(zp_a_orig, zp_new)
            delta_zp_drift_list.append(d_zp_drift)

            count += B_size

    # 4. Analysis & Visualization
    delta_zs = np.concatenate(delta_zs_list)
    delta_zp = np.concatenate(delta_zp_drift_list)

    mean_zs_shift = np.mean(delta_zs)
    mean_zp_shift = np.mean(delta_zp)
    ratio = mean_zp_shift / (mean_zs_shift + 1e-6)

    print("\n" + "=" * 70)
    print("📊 Cross-Domain Intervention Results (GTA5 Structure + Cityscapes Style)")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Mean Shift':<15} | {'Expectation'}")
    print("-" * 70)
    print(f"{'ΔZs (Structure Stability)':<35} | {mean_zs_shift:.4f}          | Low (Stable)")
    print(f"{'ΔZp (Domain Gap Responsiveness)':<35} | {mean_zp_shift:.4f}          | Very High (>0.8?)")
    print("-" * 70)
    print(f"📈 Causal Robustness Ratio: {ratio:.2f}")
    print("=" * 70)

    # 绘图
    plt.figure(figsize=(10, 6))

    import seaborn as sns
    sns.kdeplot(delta_zs, fill=True, color='green', label=r'$\Delta Z_s$ (Structure Stability)', clip=(0, None))
    sns.kdeplot(delta_zp, fill=True, color='purple', label=r'$\Delta Z_p$ (Domain Gap)', clip=(0, None))

    plt.title(f"Cross-Domain Stability: GTA5 $\\to$ Cityscapes Intervention\nRatio = {ratio:.2f}", fontsize=14)
    plt.xlabel("Relative Feature Shift (L2)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = "cross_domain_stability.png"
    plt.savefig(save_path, dpi=150)
    print(f"✅ Plot saved to {save_path}")


if __name__ == "__main__":
    main()