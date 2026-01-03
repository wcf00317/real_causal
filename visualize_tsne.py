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
from models.causal_model import CausalMTLModel
from data_utils.gta5_dataset import GTA5Dataset
from data_utils.cityscapes_dataset import CityscapesDataset
from utils.general_utils import set_seed

# ==============================================================================
# 配置区域
# ==============================================================================
INTERESTING_CLASSES = [0, 1, 2]  # 0=Road, 1=Building, 2=Object
CLASS_NAMES = {0: 'Road', 1: 'Building', 2: 'Object'}
SAVE_NAME = "tsne_distribution.png"  # 固定文件名


# ==============================================================================
# 1. 数据提取函数 (加入精确截断逻辑)
# ==============================================================================
@torch.no_grad()
def extract_tsne_data(model, loader, device, domain_name, max_images, points_per_image=20):
    model.eval()
    zs_vectors = []
    zs_labels = []
    zs_domains = []
    zp_vectors = []
    zp_domains = []

    count_img = 0
    pbar = tqdm(loader, desc=f"Extracting {domain_name}", leave=False)

    for batch in pbar:
        needed = max_images - count_img
        if needed <= 0:
            break

        imgs = batch['rgb']
        masks = batch['segmentation'].long()

        # [Fix] 精确切片，防止多读一个 Batch 导致数量溢出
        if imgs.size(0) > needed:
            imgs = imgs[:needed]
            masks = masks[:needed]

        imgs = imgs.to(device)
        masks = masks.to(device)
        B = imgs.size(0)

        outputs = model(imgs, stage=2)

        # --- Zp ---
        z_p = outputs['z_p'].detach().cpu().numpy()
        zp_vectors.append(z_p)
        zp_domains.extend([domain_name] * B)

        # --- Zs ---
        z_s_map = outputs['z_s_map'].detach()
        # Resize mask to feature map size
        H_feat, W_feat = z_s_map.shape[-2:]
        masks_resized = torch.nn.functional.interpolate(
            masks.unsqueeze(1).float(), size=(H_feat, W_feat), mode='nearest'
        ).squeeze(1).long()

        z_s_perm = z_s_map.permute(0, 2, 3, 1)  # [B, H, W, C]
        C = z_s_perm.shape[-1]

        for b in range(B):
            feat_flat = z_s_perm[b].reshape(-1, C)
            mask_flat = masks_resized[b].reshape(-1)

            valid_idx = []
            for cls_id in INTERESTING_CLASSES:
                idx = (mask_flat == cls_id).nonzero(as_tuple=True)[0]
                if len(idx) > 0: valid_idx.append(idx)

            if len(valid_idx) == 0: continue
            valid_idx = torch.cat(valid_idx)

            if len(valid_idx) > points_per_image:
                perm = torch.randperm(len(valid_idx))[:points_per_image]
                sampled_idx = valid_idx[perm]
            else:
                sampled_idx = valid_idx

            zs_vectors.append(feat_flat[sampled_idx].cpu().numpy())
            zs_labels.append(mask_flat[sampled_idx].cpu().numpy())
            zs_domains.extend([domain_name] * len(sampled_idx))

        count_img += B

    return {
        'zs': np.concatenate(zs_vectors, axis=0) if zs_vectors else None,
        'zs_lbl': np.concatenate(zs_labels, axis=0) if zs_labels else None,
        'zs_dom': np.array(zs_domains) if zs_vectors else None,
        'zp': np.concatenate(zp_vectors, axis=0) if zp_vectors else None,
        'zp_dom': np.array(zp_domains) if zp_vectors else None
    }


# ==============================================================================
# 2. t-SNE 计算与绘图 (Visual Consistency Fixed)
# ==============================================================================
def run_tsne_and_plot(data_gta, data_cs, target_zs_total):
    print("\nComputing t-SNE...")

    # --- Zs ---
    if data_gta['zs'] is None or data_cs['zs'] is None:
        print("Error: No Zs data extracted.")
        return

    X_zs = np.concatenate([data_gta['zs'], data_cs['zs']], axis=0)
    y_zs_cls = np.concatenate([data_gta['zs_lbl'], data_cs['zs_lbl']], axis=0)
    y_zs_dom = np.concatenate([data_gta['zs_dom'], data_cs['zs_dom']], axis=0)

    # 数量控制
    if len(X_zs) > target_zs_total:
        print(f"  -> Downsampling Zs from {len(X_zs)} to {target_zs_total}")
        idx = np.random.choice(len(X_zs), target_zs_total, replace=False)
        X_zs = X_zs[idx]
        y_zs_cls = y_zs_cls[idx]
        y_zs_dom = y_zs_dom[idx]

    tsne_zs = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_zs_2d = tsne_zs.fit_transform(X_zs)

    # --- Zp ---
    X_zp = np.concatenate([data_gta['zp'], data_cs['zp']], axis=0)
    y_zp_dom = np.concatenate([data_gta['zp_dom'], data_cs['zp_dom']], axis=0)
    tsne_zp = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_zp_2d = tsne_zp.fit_transform(X_zp)

    # --- 绘图 ---
    print(f"Plotting to {SAVE_NAME}...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 统一颜色定义: Color = Domain
    domain_colors = {'GTA5': 'tab:red', 'Cityscapes': 'tab:blue'}

    # =========================================================
    # Plot A: Zs Distribution
    # Request: Color=Domain (Match Right), Shape=Class
    # =========================================================
    ax = axes[0]

    # 定义形状 (Represent Class)
    class_markers = {0: 'o', 1: 's', 2: '^'}  # 0=Circle, 1=Square, 2=Triangle

    plot_idx = np.arange(len(X_zs))
    np.random.shuffle(plot_idx)
    legend_handles = {}

    for i in plot_idx:
        dom = y_zs_dom[i]
        cls_id = y_zs_cls[i]
        if cls_id not in INTERESTING_CLASSES: continue

        # Logic: Color from Domain, Marker from Class
        c = domain_colors[dom]
        m = class_markers[cls_id]

        lbl_cls_name = CLASS_NAMES.get(cls_id, cls_id)
        # Legend Label ex: "Road (GTA5)"
        lbl = f"{lbl_cls_name} ({dom})"

        sc = ax.scatter(
            X_zs_2d[i, 0], X_zs_2d[i, 1],
            c=c, marker=m,
            alpha=0.6, s=40, edgecolors='k', linewidth=0.3
        )
        if lbl not in legend_handles: legend_handles[lbl] = sc

    ax.set_title(r"(A) $Z_s$ Distribution" + f"\n(N={len(X_zs)}, Color=Domain, Shape=Class)")

    # Legend Sorting (Group by Class for readability)
    sorted_keys = sorted(legend_handles.keys())
    sorted_handles = [legend_handles[k] for k in sorted_keys]

    ax.legend(handles=sorted_handles, labels=sorted_keys,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
    ax.grid(True, alpha=0.3)

    # =========================================================
    # Plot B: Zp Distribution
    # Request: Color=Domain
    # =========================================================
    ax = axes[1]

    for dom in ['GTA5', 'Cityscapes']:
        mask = (y_zp_dom == dom)
        ax.scatter(
            X_zp_2d[mask, 0], X_zp_2d[mask, 1],
            c=domain_colors[dom],
            label=dom,
            alpha=0.75, s=50, edgecolors='w', linewidth=0.5
        )

    ax.set_title(r"(B) $Z_p$ Distribution" + f"\n(N={len(X_zp)}, Color=Domain)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_NAME, dpi=150)
    print("✅ Done.")


# ==============================================================================
# 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    # 两个参数分开控制
    parser.add_argument('--samples_zp', type=int, default=200, help="Total samples for Zp (Right Plot)")
    parser.add_argument('--samples_zs', type=int, default=500, help="Total samples for Zs (Left Plot)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    set_seed(config['training']['seed'])

    model = CausalMTLModel(config['model'], config['data']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, strict=False)

    print("Initializing Datasets...")
    ds_gta = GTA5Dataset(config['data']['train_dataset_path'], config['data']['img_size'], augmentation=False)
    ds_cs = CityscapesDataset(config['data']['val_dataset_path'], split='val')
    loader_gta = DataLoader(ds_gta, batch_size=16, shuffle=True, num_workers=4)
    loader_cs = DataLoader(ds_cs, batch_size=16, shuffle=True, num_workers=4)

    # 逻辑修正：samples_zp 是总数，所以每个域取一半
    per_domain_img = args.samples_zp // 2
    print(f"Strategy: Extracting {per_domain_img} images per domain (Total Zp={args.samples_zp})...")

    data_gta = extract_tsne_data(model, loader_gta, device, "GTA5", max_images=per_domain_img)
    data_cs = extract_tsne_data(model, loader_cs, device, "Cityscapes", max_images=per_domain_img)

    # 画图
    run_tsne_and_plot(data_gta, data_cs, target_zs_total=args.samples_zs)


if __name__ == '__main__':
    main()