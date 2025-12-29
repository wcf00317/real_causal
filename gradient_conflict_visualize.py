import os
import argparse
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader

# --- 项目模块导入 ---
# 请确保这些文件都在正确的目录下
from models.causal_celeba_model import CausalCelebAModel
from data_utils.celeba_dataset import CelebADataset
from utils.general_utils import set_seed

# CelebA 40个属性的标准名称列表
CELEBA_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Gradient Conflict Heatmap for CelebA Attributes")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the config file (e.g., configs/resnet/5celeba.yaml)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to the trained model checkpoint (optional)")
    parser.add_argument('--output', type=str, default='gradient_conflict_heatmap.png',
                        help="Output filename for the heatmap")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (cuda/cpu)")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_cosine_similarity(vecs):
    """
    计算一组向量的两两余弦相似度
    Args:
        vecs: [N, D] tensor
    Returns:
        sim_matrix: [N, N] numpy array
    """
    # 归一化
    norm = torch.norm(vecs, p=2, dim=1, keepdim=True)
    vecs_normalized = vecs / (norm + 1e-8)
    # 矩阵乘法计算 Cosine Similarity
    sim_matrix = torch.mm(vecs_normalized, vecs_normalized.t())
    return sim_matrix.cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 加载配置
    print(f"📂 Loading config: {args.config}")
    config = load_config(args.config)

    # 设置随机种子以保证复现性
    set_seed(config['training'].get('seed', 2024))

    # 2. 初始化数据集 (仅需一个 Batch)
    data_cfg = config['data']
    print(f"📚 Initializing CelebA Dataset from {data_cfg['dataset_path']}...")

    # 强制使用 train 集，因为我们看的是训练时的梯度冲突
    dataset = CelebADataset(
        root_dir=data_cfg['dataset_path'],
        split='train',
        img_size=data_cfg.get('img_size', [128, 128]),
        num_attributes=40,  # 强制分析所有40个属性
        augmentation=False  # 关闭增强以减少随机性干扰分析
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 获取一个 Batch 的数据
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("❌ Error: Dataset is empty.")
        return

    imgs = batch['image'].to(device)
    attrs = batch['attributes'].to(device)  # [B, 40]

    print(f"✅ Data loaded. Batch shape: {imgs.shape}")

    # 3. 初始化模型
    print("🧠 Initializing CausalCelebAModel...")
    model = CausalCelebAModel(config['model']).to(device)

    # 4. 加载权重 (如果有)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📥 Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ No checkpoint provided or found. Using initialized weights (Analysis might be random).")

    # 5. 准备梯度分析
    model.eval()  # 设为 eval 模式主要是为了 fix BN，但我们需要 grad

    # 确保所有参数都需要梯度 (虽然 eval 模式不影响 requires_grad，但保险起见)
    for param in model.parameters():
        param.requires_grad = True

    # === 关键：定位 Shared Encoder 的最后一层 ===
    # ResNet18: encoder -> features (Sequential) -> [..., layer4 (Sequential)]
    # 我们取 layer4 的最后一个 BasicBlock 的第二个卷积层 (conv2) 的权重
    # 理由：这是特征进入 Task Heads 分叉前，最后一个包含可学习参数的层
    try:
        # model.encoder 是 ResNet18Encoder
        # model.encoder.features 是 Sequential
        # 最后一个是 layer4
        # layer4 是 BasicBlock 的列表
        target_layer = model.encoder.backbone.layer4[-1].conv2
        target_param = target_layer.weight
        print(f"🎯 Target Shared Layer: model.encoder.backbone.layer4[-1].conv2 ({target_param.shape})")
    except AttributeError:
        print("❌ Error: Could not locate the specific ResNet layer. Please check model structure.")
        return

    # Loss 函数 (单属性 BCE)
    # 注意：这里我们不加 Sigmoid，因为 AttributeHead 输出的是 Logits，用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    grads = []
    print("\n⚡ Computing gradients for each of the 40 attributes...")

    # 6. 循环计算每个属性的梯度
    for i in range(40):
        # 清零梯度
        model.zero_grad()

        # 前向传播
        outputs = model(imgs)
        pred_logits = outputs['pred_attr']  # [B, 40]

        # 提取当前任务的预测和标签
        task_pred = pred_logits[:, i]
        task_gt = attrs[:, i].float()

        # 计算该任务的 Loss
        loss = criterion(task_pred, task_gt)

        # 反向传播
        loss.backward()

        # 获取目标层的梯度并展平
        if target_param.grad is not None:
            g = target_param.grad.view(-1).clone().detach()
            grads.append(g)
        else:
            print(f"⚠️ Warning: No gradient for attribute {i}")
            grads.append(torch.zeros_like(target_param.view(-1)).detach())

    # 7. 计算相似度矩阵
    grads_stack = torch.stack(grads)  # [40, D]
    sim_matrix = compute_cosine_similarity(grads_stack)  # [40, 40] numpy

    # 8. 绘图
    print(f"🎨 Plotting heatmap to {args.output}...")
    plt.figure(figsize=(20, 16))

    # 使用 Seaborn 绘制热力图
    # vmin=-1 (完全冲突), vmax=1 (完全一致), center=0 (正交)
    ax = sns.heatmap(
        sim_matrix,
        cmap='RdBu_r',  # 蓝色=负相关(冲突), 红色=正相关(协同)
        vmin=-1, vmax=1, center=0,
        square=True,
        xticklabels=CELEBA_ATTRIBUTES,
        yticklabels=CELEBA_ATTRIBUTES
    )

    plt.title(f"Gradient Conflict Heatmap (Shared Encoder Last Layer)\nModel: Ours (CausalMTL)", fontsize=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"✅ Done. Heatmap saved.")


if __name__ == '__main__':
    main()