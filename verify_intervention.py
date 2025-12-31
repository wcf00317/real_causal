import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 项目模块导入 ---
from models.causal_model import CausalMTLModel
from data_utils.gta5_dataset import GTA5Dataset
from utils.general_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Interventional Stability Analysis (The 'Crash Test')")
    parser.add_argument('--config', type=str, required=True, help="Config file path")
    parser.add_argument('--checkpoint', type=str, required=True, help="Model checkpoint path")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    parser.add_argument('--samples', type=int, default=500, help="Number of samples to evaluate")
    return parser.parse_args()


def relative_l2_distance(z_orig, z_new):
    """
    计算相对 L2 距离: ||z - z_hat|| / ||z||
    输入 shape: [B, ...] 会被展平为 [B, D] 计算
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
    print(f"🚀 Running Interventional Stability Analysis on {device}")

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

    # 2. Data Loader (Use GTA5 Source for 'Crash Test')
    # 我们需要在源域上测试：改变源域图片的风格，看源域的结构特征是否崩坏
    ds = GTA5Dataset(
        root_dir=config['data']['train_dataset_path'],
        img_size=config['data']['img_size'],
        augmentation=False
    )
    # Batch size 必须 > 1 才能进行 shuffle
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)

    # 3. Metrics Storage
    delta_zs_list = []
    delta_zp_list = []

    count = 0
    print(f"\n⚡ Starting Intervention Loop (Target Samples: {args.samples})...")

    with torch.no_grad():
        for batch in tqdm(loader):
            if count >= args.samples:
                break

            imgs = batch['rgb'].to(device)
            B = imgs.size(0)

            # --- Step 1: 原始编码 (Original Encoding) ---
            outputs_orig = model(imgs, stage=2)
            z_s_orig = outputs_orig['z_s_map']  # [B, C, H, W]
            z_p_orig = outputs_orig['z_p']  # [B, C] (Vector)

            # --- Step 2: 干预 (Intervention) ---
            # 使用 model 内置的 generate_counterfactual_image 做 Batch 内随机 Shuffle
            # 这模拟了：保持 z_s 不变，强行把环境 z_p 换成别人的
            # I_cfa 是 "结构保持 + 风格迁移" 后的图像
            I_cfa, _ = model.generate_counterfactual_image(
                z_s_orig, outputs_orig['z_p_map'], strategy='global'
            )

            # --- Step 3: 重编码 (Re-encoding) ---
            # 把生成出来的反事实图像再喂给模型，看它提取出什么特征
            outputs_cfa = model(I_cfa, stage=2)
            z_s_new = outputs_cfa['z_s_map']
            z_p_new = outputs_cfa['z_p']

            # --- Step 4: 测量 (Measurement) ---

            # Metric A: Stability of Zs (期望越小越好)
            # 逻辑：虽然图片变成了别人的风格，但模型提取出的 Zs 应该和原来的一模一样
            d_zs = relative_l2_distance(z_s_orig, z_s_new)
            delta_zs_list.append(d_zs)

            # Metric B: Responsiveness of Zp (期望越大越好)
            # 逻辑：既然图片换了风格，模型提取出的 Zp 应该变了 (变成别人的 Zp)，
            # 所以它应该离原来的 Zp 很远。
            d_zp = relative_l2_distance(z_p_orig, z_p_new)
            delta_zp_list.append(d_zp)

            count += B

    # 4. Analysis & Visualization
    delta_zs = np.concatenate(delta_zs_list)
    delta_zp = np.concatenate(delta_zp_list)

    mean_zs_shift = np.mean(delta_zs)
    mean_zp_shift = np.mean(delta_zp)

    print("\n" + "=" * 60)
    print("📊 Interventional Stability Results (System Log Analysis)")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Mean Relative Shift':<20} | {'Expectation'}")
    print("-" * 60)
    print(f"{'ΔZs (Structure Stability)':<30} | {mean_zs_shift:.4f}               | Low (Stable)")
    print(f"{'ΔZp (Style Responsiveness)':<30} | {mean_zp_shift:.4f}               | High (Changed)")
    print("-" * 60)

    # 计算比率：响应性 / 稳定性 (信噪比)
    ratio = mean_zp_shift / (mean_zs_shift + 1e-6)
    print(f"📈 Causal Robustness Ratio (Zp/Zs): {ratio:.2f} (Higher is better)")
    print("=" * 60)

    # 绘图
    plt.figure(figsize=(10, 6))

    # 绘制分布直方图
    sns_plot = True
    try:
        import seaborn as sns
    except ImportError:
        sns_plot = False

    if sns_plot:
        sns.kdeplot(delta_zs, fill=True, color='green', label=r'$\Delta Z_s$ (Structure)', clip=(0, None))
        sns.kdeplot(delta_zp, fill=True, color='red', label=r'$\Delta Z_p$ (Style)', clip=(0, None))
    else:
        plt.hist(delta_zs, bins=50, alpha=0.5, color='green', label=r'$\Delta Z_s$ (Structure)', density=True)
        plt.hist(delta_zp, bins=50, alpha=0.5, color='red', label=r'$\Delta Z_p$ (Style)', density=True)

    plt.title("Interventional Stability: Feature Drift under Style Intervention", fontsize=14)
    plt.xlabel("Relative L2 Distance (Drift)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = "interventional_stability.png"
    plt.savefig(save_path, dpi=150)
    print(f"✅ Plot saved to {save_path}")
    print("\nInterpretation:")
    print(" - Green curve (Zs) should be clustered near 0 (Immutable).")
    print(" - Red curve (Zp) should be distributed far from 0 (Mutable).")


if __name__ == "__main__":
    main()