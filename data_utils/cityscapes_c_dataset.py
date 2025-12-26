import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CityscapesCDataset(Dataset):
    """
    [Final Fixed Version]
    针对 Cityscapes-C 的泛化测试 Dataset。
    逻辑严格对齐 LibMTL 跑通版本：
    1. Image: Resize(256, 128) -> float32 -> / 255.0 (无 ImageNet Mean/Std 归一化)
    2. Label: 直接读取预处理好的 .npy (值域 -1~6，无需额外映射)
    3. Index: 严格依赖文件排序索引 (idx) 来对齐 Image 和 GT
    """

    def __init__(self, root_dir, gt_dir, corruption='fog', severity=5, img_size=(128, 256)):
        """
        Args:
            root_dir: Cityscapes-C 根目录 (包含 fog, noise 等文件夹)
            gt_dir: 预处理后的 Cityscapes GT 根目录 (包含 val/label/*.npy)
            corruption: 腐蚀类型 (如 'fog', 'gaussian_noise')
            severity: 严重程度 (1-5)
            img_size: (H, W) 默认 (128, 256)。注意：代码内部会自动适配 PIL 的 (W, H)。
        """
        super().__init__()
        self.gt_dir = gt_dir
        # Config 传入通常是 [H, W], PIL Resize 需要 (W, H)
        self.target_size = (img_size[1], img_size[0])

        # 1. 构造 Cityscapes-C 图片路径
        # 路径结构: root_dir/corruption/severity/city/image.png
        self.image_dir = os.path.join(root_dir, corruption, str(severity))
        self.img_paths = []

        if os.path.exists(self.image_dir):
            # 严格按照字典序读取，确保第 i 张图对应 gt_dir 里的 i.npy
            # 先遍历城市文件夹
            subfolders = sorted([d for d in os.listdir(self.image_dir)
                                 if os.path.isdir(os.path.join(self.image_dir, d))])

            if len(subfolders) > 0:
                for city in subfolders:
                    city_path = os.path.join(self.image_dir, city)
                    files = sorted([f for f in os.listdir(city_path) if f.endswith('.png')])
                    for f in files:
                        self.img_paths.append(os.path.join(city_path, f))
            else:
                # 兼容部分扁平结构 (直接在 severity 目录下放图片)
                files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
                for f in files:
                    self.img_paths.append(os.path.join(self.image_dir, f))

        self.num_samples = len(self.img_paths)
        print(f"[Cityscapes-C] {corruption} (Sev {severity}): Found {self.num_samples} images.")

        if self.num_samples == 0:
            print(f"⚠️ Warning: No images found in {self.image_dir}. Please check path.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ==================================================
        # 1. 图片处理 (严格对齐 test_cityscape_c.py)
        # ==================================================
        img_path = self.img_paths[idx]
        img_pil = Image.open(img_path).convert('RGB')

        # Resize: PIL使用 (W, H)
        # 你的测试: img_resized = img_pil.resize((256, 128), resample=Image.BILINEAR)
        img_resized = img_pil.resize(self.target_size, resample=Image.BILINEAR)

        # Normalize: 仅 / 255.0，不做减均值除方差
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        # ToTensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # ==================================================
        # 2. GT 读取 (读取预处理好的 .npy)
        # ==================================================
        # 假设 gt_dir 下结构为 val/label/idx.npy
        label_path = os.path.join(self.gt_dir, 'val', 'label', f'{idx}.npy')
        depth_path = os.path.join(self.gt_dir, 'val', 'depth', f'{idx}.npy')

        # 默认值 (H, W)
        H, W = self.target_size[1], self.target_size[0]
        label = torch.zeros((H, W), dtype=torch.long)
        depth = torch.zeros((1, H, W), dtype=torch.float)

        # 读取 Label
        if os.path.exists(label_path):
            try:
                # 直接读取，你的 npy 已经是 [-1, 0...6]，直接用
                lbl_np = np.load(label_path)
                label = torch.from_numpy(lbl_np).long()
            except Exception:
                pass

        # 读取 Depth (如果存在)
        if os.path.exists(depth_path):
            try:
                dep_np = np.load(depth_path)
                # 维度修正: 确保是 [1, H, W]
                if dep_np.ndim == 2:
                    dep_np = dep_np[np.newaxis, ...]
                elif dep_np.ndim == 3 and dep_np.shape[2] == 1:
                    dep_np = np.moveaxis(dep_np, -1, 0)

                depth = torch.from_numpy(dep_np).float()
            except Exception:
                pass

        # ==================================================
        # 3. 返回字典 (适配 main.py 接口)
        # ==================================================
        return {
            'rgb': img_tensor,
            'segmentation': label,
            'depth': depth,
            # 兼容性字段：Causal Model 可能需要 target 进行重构 loss 计算
            'appearance_target': img_tensor,
            'scene_type': torch.tensor(0, dtype=torch.long)
        }