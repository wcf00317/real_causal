import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

GTA5_TO_7_CLASSES = {
    0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1,
    7: 0, 8: 0, 9: -1, 10: -1, 11: 1, 12: 1, 13: 1,
    14: -1, 15: -1, 16: -1, 17: 2, 18: -1, 19: 2, 20: 2,
    21: 3, 22: 3, 23: 4, 24: 5, 25: 5, 26: 6, 27: 6,
    28: 6, 29: -1, 30: -1, 31: 6, 32: 6, 33: 6
}

class GTA5Dataset(Dataset):
    def __init__(self, root_dir, img_size, augmentation=False):
        """
        新增 augmentation 参数控制是否翻转
        """
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.augmentation = augmentation  # <--- 新增

        self.images_dir = os.path.join(root_dir, 'images')
        if not os.path.exists(self.images_dir):
            self.images_dir = os.path.join(root_dir, 'Images')

        self.labels_dir = os.path.join(root_dir, 'labels')
        if not os.path.exists(self.labels_dir):
            self.labels_dir = os.path.join(root_dir, 'Labels')

        self.images = []
        self.targets = []

        search_pattern = os.path.join(self.images_dir, "*.png")
        files = glob.glob(search_pattern)

        for img_path in files:
            file_name = os.path.basename(img_path)
            label_path = os.path.join(self.labels_dir, file_name)
            if os.path.exists(label_path):
                self.images.append(img_path)
                self.targets.append(label_path)

        self.images.sort()
        self.targets.sort()

        self.mapping = np.zeros(256, dtype=np.int64) - 1
        for k, v in GTA5_TO_7_CLASSES.items():
            if k >= 0:
                self.mapping[k] = v

        print(f"[GTA5] Found {len(self.images)} samples in {root_dir} (Aug={self.augmentation})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.targets[idx]

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # Resize
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        label = label.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        # Augmentation (仅在 augmentation=True 时执行)
        if self.augmentation and random.random() < 0.5:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)

        to_tensor = transforms.ToTensor()
        rgb_tensor_unnormalized = to_tensor(img).float()

        label_np = np.array(label, dtype=np.int64)
        label_np[label_np > 255] = 255
        label_mapped = self.mapping[label_np]
        seg_tensor = torch.from_numpy(label_mapped).long()

        depth_tensor = torch.zeros((1, self.img_size[1], self.img_size[0]), dtype=torch.float32)

        return {
            'rgb': rgb_tensor_unnormalized,
            'depth': depth_tensor,
            'segmentation': seg_tensor,
            'scene_type': torch.tensor(0),
            'appearance_target': rgb_tensor_unnormalized
        }