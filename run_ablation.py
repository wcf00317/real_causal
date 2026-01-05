import argparse
import logging
import os
import sys
import shutil
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==============================================================================
#  👇 1. 项目内模块导入
# ==============================================================================
from data_utils.nyuv2_dataset import NYUv2Dataset
from data_utils.gta5_dataset import GTA5Dataset
from data_utils.cityscapes_dataset import CityscapesDataset
from data_utils.cityscapes_c_dataset import CityscapesCDataset

from models.baselines import RawMTLModel, SingleTaskModel
from models.causal_model import CausalMTLModel

from losses.mtl_loss import MTLLoss
from losses.composite_loss import CompositeLoss

from engine.trainer import train
from utils.general_utils import setup_logging

# ==============================================================================
#  👇 2. 实验列表配置
# ==============================================================================

BASE_CONFIG_PATH = "config.yaml"
COMMON_LOSS_WEIGHTS = {
    # 任务权重
    "lambda_seg": 20.0,
    "lambda_depth": 10.0,
    "lambda_normal": 10.0,
    "lambda_edge_consistency": 0.1,

    # 重构与物理约束
    "alpha_recon_geom": 2.0,
    "beta_recon_app": 2.0,
    "lambda_l1_recon": 1.0,
    "lambda_img": 1.0,
    "lambda_alb_tv": 0.1,
    "lambda_sh_gray": 0.001,
    "lambda_xcov": 0.5
}
ABLATION_EXPERIMENTS = [
    # --- STL Normal ---
    {
        "name": "01_EW_Baseline",
        "override": {
            "model": {"type": "raw_mtl"},
            "losses": {
                "lambda_seg": 20.0,
                "lambda_depth": 10.0,
                "lambda_normal": 10.0,
                "lambda_independence": 0.0
            },
            "training": {"cfa": {"enabled": False}}
        }
    },
]
CITYSCAPES_OVERRIDES = {
    # 强制指定 Cityscapes 数据配置 (覆盖 config.yaml)
    "data": {
        "type": "cityscapes",
        # ⚠️ 确保这个路径和你服务器上的真实路径一致 (参考了你提供的yaml)
        "dataset_path": "/data/chengfengwu/alrl/mtl_dataset/cityscape_preprocess",
        "img_size": [128, 256],
        "batch_size": 32,
        "num_workers": 4
    },
    # 强制指定 Cityscapes 的分类数 (参考你的yaml是7类)
    "model_common": {
        "encoder_name": "resnet50",
        "num_seg_classes": 7,  # Cityscapes (7类) vs NYUv2 (40类)
        "num_scene_classes": 0 # Cityscapes 没有 scene 分类
    }
}

ABLATION_EXPERIMENTS = [
    # -------------------------------------------------------------------------
    # 2. Cityscapes STL: Depth
    # -------------------------------------------------------------------------
    {
        "name": "02_CS_STL_Depth",
        "override": {
            "data": CITYSCAPES_OVERRIDES["data"],
            "model": {
                "type": "single_task",
                "active_task": "depth",
                **CITYSCAPES_OVERRIDES["model_common"]
            },
            "losses": {
                "tasks": ["depth"],
                "lambda_depth": 20.0
            }
        }
    }


]

# ==============================================================================
#  🛠️ [Hotfix] 修复 SingleTaskLoss 缺失 Normal 的问题
# ==============================================================================

class NormalLoss(nn.Module):
    """Cos Sim Loss: 1 - cos(theta)"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        pred = F.normalize(pred, p=2, dim=1)
        binary_mask = (torch.sum(torch.abs(gt), dim=1) > 0).float().unsqueeze(1)
        dot_prod = (pred * gt).sum(dim=1, keepdim=True)
        num_valid = torch.sum(binary_mask)
        if num_valid > 0:
            loss = 1 - torch.sum(dot_prod * binary_mask) / num_valid
        else:
            loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        return loss


class FixedSingleTaskLoss(nn.Module):
    """
    修复版 SingleTaskLoss：
    1. 补全 Normal
    2. 统一使用 ignore_index = -1 (配合 Wrapper)
    """

    def __init__(self, active_task, loss_weights):
        super().__init__()
        self.active_task = active_task
        self.weights = loss_weights

        # 🔥 关键：ignore_index 改为 -1，配合 Wrapper 的 255->-1
        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.depth_loss_fn = nn.L1Loss()
        self.normal_loss_fn = NormalLoss()
        self.scene_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=outputs['pred_seg'].device)

        if self.active_task == 'seg':
            l_seg = self.seg_loss_fn(outputs['pred_seg'], targets['segmentation'])
            total_loss += self.weights.get('lambda_seg', 1.0) * l_seg
            loss_dict['seg_loss'] = l_seg

        elif self.active_task == 'depth':
            pred_d = outputs['pred_depth']
            gt_d = targets['depth']
            mask = (gt_d > 0).float().detach()
            num_valid = mask.sum()
            if num_valid > 0:
                l_depth = (torch.abs(pred_d - gt_d) * mask).sum() / num_valid
            else:
                l_depth = torch.tensor(0.0, device=pred_d.device)
            total_loss += self.weights.get('lambda_depth', 1.0) * l_depth
            loss_dict['depth_loss'] = l_depth

        elif self.active_task == 'normal':
            l_normal = self.normal_loss_fn(outputs['pred_normal'], targets['normal'])
            total_loss += self.weights.get('lambda_normal', 1.0) * l_normal
            loss_dict['normal_loss'] = l_normal

        elif self.active_task == 'scene':
            if 'scene_type' in targets:
                l_scene = self.scene_loss_fn(outputs['pred_scene'], targets['scene_type'])
                total_loss += self.weights.get('lambda_scene', 1.0) * l_scene
                loss_dict['scene_loss'] = l_scene

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


# ==============================================================================
#  🛡️ 数据转换 Wrapper (255 -> -1)
# ==============================================================================

class SafeDatasetWrapper(Dataset):
    """
    因为 evaluator.py 写死了 ignore_index=-1，
    我们必须把数据里的 255 转成 -1，否则就会报 shape error。
    """

    def __init__(self, dataset):
        self.dataset = dataset
        if hasattr(dataset, 'scene_classes'):
            self.scene_classes = dataset.scene_classes

    def __getitem__(self, index):
        data = self.dataset[index]

        # 🔥【关键修复】 255 -> -1
        # 这就是解决 "shape invalid" 的终极方案
        if 'segmentation' in data:
            seg = data['segmentation']
            if isinstance(seg, torch.Tensor) and (seg == 255).any():
                seg[seg == 255] = -1
                data['segmentation'] = seg

        return data

    def __len__(self):
        return len(self.dataset)


# ==============================================================================
#  🛠️ 3. 数据加载器工厂
# ==============================================================================

def get_dataloaders(data_cfg):
    dataset_type = data_cfg.get('type', 'nyuv2').lower()
    img_size = tuple(data_cfg['img_size'])
    dataset_path = data_cfg.get('dataset_path', data_cfg.get('root_path'))
    batch_size = data_cfg['batch_size']
    num_workers = data_cfg['num_workers']
    pin_memory = data_cfg.get('pin_memory', True)

    logging.info(f"📋 Initializing Dataset: {dataset_type}")

    val_dataset_src = None

    if dataset_type == 'gta5_to_cityscapes':
        train_dataset = GTA5Dataset(root_dir=data_cfg['train_dataset_path'], img_size=img_size, augmentation=True)
        val_dataset_tgt = CityscapesDataset(root_dir=data_cfg['val_dataset_path'], split='val')
        if 'source_val_path' in data_cfg:
            val_dataset_src = GTA5Dataset(root_dir=data_cfg['source_val_path'], img_size=img_size, augmentation=False)

    elif dataset_type == 'cityscapes':
        train_dataset = CityscapesDataset(root_dir=dataset_path, split='train')
        val_dataset_tgt = CityscapesDataset(root_dir=dataset_path, split='val')

    elif dataset_type == 'nyuv2':
        train_dataset = NYUv2Dataset(root_dir=dataset_path, mode='train',
                                     augmentation=data_cfg.get('augmentation', False))
        val_dataset_tgt = NYUv2Dataset(root_dir=dataset_path, mode='val')

    elif dataset_type == 'cityscapes2cityscapes_c':
        train_dataset = CityscapesDataset(root_dir=data_cfg['train_dataset_path'], split='train')
        val_dataset_tgt = CityscapesDataset(root_dir=data_cfg['train_dataset_path'], split='val')

    else:
        raise ValueError(f"❌ Unsupported dataset type: '{dataset_type}'")

    # 🔥 套上 Wrapper (255 -> -1)
    train_dataset = SafeDatasetWrapper(train_dataset)
    val_dataset_tgt = SafeDatasetWrapper(val_dataset_tgt)
    if val_dataset_src:
        val_dataset_src = SafeDatasetWrapper(val_dataset_src)

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      pin_memory=pin_memory, drop_last=True)
    dataloaders['val'] = DataLoader(val_dataset_tgt, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    pin_memory=pin_memory)
    if val_dataset_src is not None:
        dataloaders['val_source'] = DataLoader(val_dataset_src, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=pin_memory)

    return dataloaders


# ==============================================================================
#  Helpers
# ==============================================================================

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            recursive_update(d[k], v)
        else:
            d[k] = v
    return d


def save_code_snapshot(log_dir):
    try:
        if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
        backup_dir = os.path.join(log_dir, 'code_snapshot')
        if os.path.exists(backup_dir): shutil.rmtree(backup_dir)
        ignore_func = shutil.ignore_patterns('__pycache__', 'experiments', 'checkpoints', 'data', '.git', '.idea',
                                             'wandb', '*.pth', '*.tar', 'output', 'runs')
        shutil.copytree('.', backup_dir, ignore=ignore_func)
    except Exception as e:
        print(f"⚠️ Failed to save code snapshot: {e}")


def build_optimizer(model, config):
    opt_cfg = config.get('training', {})
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(opt_cfg.get('learning_rate', opt_cfg.get('lr', 1e-4)))
    weight_decay = float(opt_cfg.get('weight_decay', 1e-4))
    opt_type = opt_cfg.get('optimizer', 'adam').lower()

    if opt_type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return optimizer


# ==============================================================================
#  Worker
# ==============================================================================

def run_single_experiment(config, device):
    log_dir = config['logging']['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    setup_logging(log_dir)
    logging.info(f"🚀 Experiment Started: {log_dir}")
    save_code_snapshot(log_dir)

    if 'backbone' in config['model'] and 'encoder_name' not in config['model']:
        config['model']['encoder_name'] = config['model']['backbone']

    # 1. Data Loading
    dataloaders = get_dataloaders(config['data'])
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    val_loader_source = dataloaders.get('val_source', None)

    # 2. Model Selection
    model_type = config['model'].get('type', 'causal')

    if model_type == 'single_task':
        active_task = config['model'].get('active_task', 'seg')
        logging.info(f"👉 Mode: Single Task (STL) - Task: [{active_task.upper()}]")
        model = SingleTaskModel(config['model'], config['data']).to(device)
        # 🔥 使用本地定义的 FixedSingleTaskLoss
        criterion = FixedSingleTaskLoss(active_task, config['losses']).to(device)

    elif model_type == 'raw_mtl':
        logging.info("👉 Mode: EW Baseline (RawMTLModel)")
        if 'num_scene_classes' not in config['model']:
            config['model']['num_scene_classes'] = 27
        model = RawMTLModel(config['model'], config['data']).to(device)
        criterion = MTLLoss(config['losses'], use_uncertainty=False).to(device)
        # 补丁：确保 MTLLoss 也忽略 -1
        if hasattr(criterion, 'seg_loss_fn'):
            criterion.seg_loss_fn.ignore_index = -1

    else:
        logging.info(f"👉 Mode: Causal Model")
        model = CausalMTLModel(config['model'], config['data']).to(device)
        dataset_type = config['data'].get('type', 'nyuv2')
        criterion = CompositeLoss(config['losses'], dataset_type).to(device)
        # 补丁：确保 CompositeLoss 也忽略 -1 (它本来就是-1，但以防万一)
        if hasattr(criterion, 'seg_loss'):
            criterion.seg_loss.ignore_index = -1

    # 3. Training
    optimizer = build_optimizer(model, config)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir, exist_ok=True)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=None,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        val_loader_source=val_loader_source
    )
    torch.cuda.empty_cache()


# ==============================================================================
#  Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, default=BASE_CONFIG_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.base_config):
        print(f"❌ 错误：找不到基础配置文件 {args.base_config}")
        return

    base_config = load_config(args.base_config)

    for i, exp in enumerate(ABLATION_EXPERIMENTS):
        exp_name = exp['name']
        print(f"\n{'=' * 60}\n▶️  Running [{i + 1}/{len(ABLATION_EXPERIMENTS)}]: {exp_name}\n{'=' * 60}")

        current_config = copy.deepcopy(base_config)
        current_config = recursive_update(current_config, exp['override'])

        orig_log = current_config.get('logging', {}).get('log_dir', 'output')
        current_config.setdefault('logging', {})['log_dir'] = os.path.join(orig_log, exp_name)

        try:
            run_single_experiment(current_config, device)
            print(f"✅ Finished: {exp_name}")
        except Exception as e:
            print(f"❌ Failed: {exp_name}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()