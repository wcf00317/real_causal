import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# --- 项目模块导入 ---
# 确保这些文件在同级目录或 Python 路径下
from models.causal_model import CausalMTLModel
from data_utils.gta5_dataset import GTA5Dataset
from data_utils.cityscapes_dataset import CityscapesDataset
from utils.general_utils import set_seed

# ==============================================================================
# 配置区域
# ==============================================================================
# 定义要可视化的语义类别 (简化版，防止颜色太多看不清)
# GTA5 mapping: {7: 'road', 8: 'sidewalk', 11: 'building', ...}
# 这里我们选取几个具有代表性的核心类别进行可视化
INTERESTING_CLASSES = [0, 1, 2]  # 例如: 0=Road, 1=Building, 2=Car (具体取决于你的 mapping)
CLASS_NAMES = {0: 'Road', 1: 'Building', 2: 'Object'}  # 仅作图例显示用，可根据你的 dataset mapping 修改


# ==============================================================================
# 1. 数据提取函数
# ==============================================================================

@torch.no_grad()
def extract_tsne_data(model, loader, device, domain_name, max_images=100, points_per_image=50):
    """
    从数据集中提取用于 t-SNE 的向量。
    - Zs: 像素级采样 (Pixel-level sampling) -> 用于验证语义聚类
    - Zp: 图像级向量 (Image-level vector) -> 用于验证域分离
    """
    model.eval()

    # 存储 Zs 数据 (Points)
    zs_vectors = []
    zs_labels = []  # 语义标签 (0, 1, 2...)
    zs_domains = []  # 域标签 (String: 'GTA5' or 'Cityscapes')

    # 存储 Zp 数据 (Images)
    zp_vectors = []
    zp_domains = []

    count_img = 0

    for batch in tqdm(loader, desc=f"Extracting {domain_name}", leave=False):
        if count_img >= max_images:
            break

        imgs = batch['rgb'].to(device)
        masks = batch['segmentation'].to(device)  # [B, H, W]

        # 1. Forward
        outputs = model(imgs, stage=2)

        # 2. Extract Zp (Image Level)
        # 假设 outputs['z_p'] 是 [B, C] 向量
        z_p = outputs['z_p'].cpu().numpy()
        zp_vectors.append(z_p)
        zp_domains.extend([domain_name] * imgs.size(0))

        # 3. Extract Zs (Pixel Level)
        # z_s_map: [B, C, H', W']
        z_s_map = outputs['z_s_map']
        B, C, H_feat, W_feat = z_s_map.shape

        # 将 Mask 下采样到 Feature Map 大小以便对应
        masks_resized = torch.nn.functional.interpolate(
            masks.unsqueeze(1).float(),
            size=(H_feat, W_feat),
            mode='nearest'
        ).squeeze(1).long()

        # 遍历 Batch 中的每张图进行采样
        z_s_perm = z_s_map.permute(0, 2, 3, 1)  # [B, H, W, C]

        for b in range(B):
            # 获取当前图的所有像素特征和标签
            feat_flat = z_s_perm[b].reshape(-1, C)  # [N_pixels, C]
            mask_flat = masks_resized[b].reshape(-1)  # [N_pixels]

            # 过滤掉 ignore_index (-1 或 255) 和 不感兴趣的类别
            # 假设我们只关心 INTERESTING_CLASSES
            valid_idx = []
            for cls_id in INTERESTING_CLASSES:
                idx = (mask_flat == cls_id).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    valid_idx.append(idx)

            if len(valid_idx) == 0:
                continue

            valid_idx = torch.cat(valid_idx)

            # 随机采样 N 个点
            if len(valid_idx) > points_per_image:
                perm = torch.randperm(len(valid_idx))[:points_per_image]
                sampled_idx = valid_idx[perm]
            else:
                sampled_idx = valid_idx

            # 收集数据
            sampled_feats = feat_flat[sampled_idx].cpu().numpy()
            sampled_lbls = mask_flat[sampled_idx].cpu().numpy()

            zs_vectors.append(sampled_feats)
            zs_labels.append(sampled_lbls)
            zs_domains.extend([domain_name] * len(sampled_lbls))

        count_img += B

    return {
        'zs': np.concatenate(zs_vectors, axis=0) if zs_vectors else None,
        'zs_lbl': np.concatenate(zs_labels, axis=0) if zs_labels else None,
        'zs_dom': np.array(zs_domains),
        'zp': np.concatenate(zp_vectors, axis=0) if zp_vectors else None,
        'zp_dom': np.array(zp_domains)
    }


# ==============================================================================
# 2. t-SNE 计算与绘图
# ==============================================================================

def run_tsne_and_plot(data_gta, data_cs, save_path="tsne_analysis.png"):
    print("Computing t-SNE... (This may take a while)")

    # --- 准备 Zs 数据 (Concatenate) ---
    X_zs = np.concatenate([data_gta['zs'], data_cs['zs']], axis=0)
    y_zs_cls = np.concatenate([data_gta['zs_lbl'], data_cs['zs_lbl']], axis=0)
    y_zs_dom = np.concatenate([data_gta['zs_dom'], data_cs['zs_dom']], axis=0)

    # 限制 Zs 点的数量，防止绘图太慢或太乱 (例如最多只画 2000 个点)
    if len(X_zs) > 2000:
        idx = np.random.choice(len(X_zs), 2000, replace=False)
        X_zs = X_zs[idx]
        y_zs_cls = y_zs_cls[idx]
        y_zs_dom = y_zs_dom[idx]

    # Run t-SNE for Zs
    tsne_zs = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_zs_2d = tsne_zs.fit_transform(X_zs)

    # --- 准备 Zp 数据 (Concatenate) ---
    X_zp = np.concatenate([data_gta['zp'], data_cs['zp']], axis=0)
    y_zp_dom = np.concatenate([data_gta['zp_dom'], data_cs['zp_dom']], axis=0)

    # Run t-SNE for Zp
    tsne_zp = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_zp_2d = tsne_zp.fit_transform(X_zp)

    # --- 绘图 ---
    print(f"Plotting results to {save_path}...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ----------------------
    # Plot A: Zs Distribution
    # Color=Semantic, Marker=Domain
    # ----------------------
    ax = axes[0]
    # 定义域对应的 Marker
    markers = {'GTA5': 'o', 'Cityscapes': '^'}
    # 定义类别对应的颜色 (可用 colormap)
    colors = ['r', 'b', 'g', 'c', 'm', 'y']

    for dom in ['GTA5', 'Cityscapes']:
        for i, cls_id in enumerate(INTERESTING_CLASSES):
            # 筛选：既属于该域，又属于该类
            mask = (y_zs_dom == dom) & (y_zs_cls == cls_id)
            if mask.sum() == 0: continue

            ax.scatter(
                X_zs_2d[mask, 0], X_zs_2d[mask, 1],
                c=colors[i % len(colors)],
                marker=markers[dom],
                label=f"{dom}-{CLASS_NAMES.get(cls_id, cls_id)}",
                alpha=0.6, s=30, edgecolors='k', linewidth=0.3
            )

    ax.set_title(r"(A) $Z_s$ Distribution" + "\n(Color=Class, Shape=Domain)\nExpect: Mixed Shapes, Clustered Colors")
    ax.legend(fontsize='small', loc='best')
    ax.grid(True, alpha=0.3)

    # ----------------------
    # Plot B: Zp Distribution
    # Color=Domain
    # ----------------------
    ax = axes[1]
    dom_colors = {'GTA5': 'red', 'Cityscapes': 'blue'}

    for dom in ['GTA5', 'Cityscapes']:
        mask = (y_zp_dom == dom)
        ax.scatter(
            X_zp_2d[mask, 0], X_zp_2d[mask, 1],
            c=dom_colors[dom],
            label=dom,
            alpha=0.7, s=40, edgecolors='w', linewidth=0.5
        )

    ax.set_title(r"(B) $Z_p$ Distribution" + "\n(Color=Domain)\nExpect: Separated Clusters")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print("✅ Done.")


# ==============================================================================
# 3. 主流程
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--samples', type=int, default=200, help="Number of images per domain")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    set_seed(config['training']['seed'])

    # Load Model
    model = CausalMTLModel(config['model'], config['data']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    # Load Data (Subsets for speed)
    print("Initializing Datasets...")
    # GTA5 (Train set usually)
    ds_gta = GTA5Dataset(
        root_dir=config['data']['train_dataset_path'],
        img_size=config['data']['img_size'],
        augmentation=False
    )
    # Cityscapes (Val set usually)
    ds_cs = CityscapesDataset(
        root_dir=config['data']['val_dataset_path'],
        split='val'
    )

    loader_gta = DataLoader(ds_gta, batch_size=8, shuffle=True, num_workers=4)
    loader_cs = DataLoader(ds_cs, batch_size=8, shuffle=True, num_workers=4)

    # Extract
    data_gta = extract_tsne_data(model, loader_gta, device, "GTA5", max_images=args.samples)
    data_cs = extract_tsne_data(model, loader_cs, device, "Cityscapes", max_images=args.samples)

    # Plot
    run_tsne_and_plot(data_gta, data_cs, save_path="tsne_distribution.png")


if __name__ == '__main__':
    main()