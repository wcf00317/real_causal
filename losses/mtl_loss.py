import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalLoss(nn.Module):
    """
    Cosine Similarity Loss for Surface Normals
    Loss = 1 - cos(theta)
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        # pred: [B, 3, H, W], gt: [B, 3, H, W]
        # 1. 预测值归一化
        pred = F.normalize(pred, p=2, dim=1)

        # 2. 制作 Mask：GT 为 0 的区域（无效区域）不计算 Loss
        # 通常 NYUv2 的 Normal GT 在无效区域是 0
        binary_mask = (torch.sum(torch.abs(gt), dim=1) > 0).float().unsqueeze(1)

        # 3. 计算 Cosine Similarity
        # dot_prod: [B, 1, H, W]
        dot_prod = (pred * gt).sum(dim=1, keepdim=True)

        # 4. 只计算有效区域的平均 Loss
        num_valid = torch.sum(binary_mask)
        if num_valid > 0:
            loss = 1 - torch.sum(dot_prod * binary_mask) / num_valid
        else:
            loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

        return loss


class MTLLoss(nn.Module):
    """
    General Multi-Task Loss (Fixed for Normal Task)
    """

    def __init__(self, loss_weights, use_uncertainty=False):
        super().__init__()
        self.weights = loss_weights
        self.use_uncertainty = use_uncertainty

        # Seg: 确保 ignore_index=255
        self.seg_loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        # Depth: L1 Loss
        self.depth_loss_fn = nn.L1Loss()

        # 🔥 Normal: 补全 Normal Loss
        self.normal_loss_fn = NormalLoss()

        # Scene: Cross Entropy
        self.scene_loss_fn = nn.CrossEntropyLoss()

        if self.use_uncertainty:
            self.log_vars = nn.ParameterDict({
                'seg': nn.Parameter(torch.zeros(1)),
                'depth': nn.Parameter(torch.zeros(1)),
                'normal': nn.Parameter(torch.zeros(1)),  # 补上 normal
                'scene': nn.Parameter(torch.zeros(1))
            })

    def forward(self, outputs, targets):
        loss_dict = {}

        # 1. Seg Loss
        l_seg = self.seg_loss_fn(outputs['pred_seg'], targets['segmentation'])

        # 2. Depth Loss (带 Mask)
        if 'depth' in targets:
            pred_d = outputs['pred_depth']
            gt_d = targets['depth']
            mask = (gt_d > 0).float().detach()
            num_valid = mask.sum()
            if num_valid > 0:
                l_depth = (torch.abs(pred_d - gt_d) * mask).sum() / num_valid
            else:
                l_depth = torch.tensor(0.0, device=pred_d.device, requires_grad=True)
        else:
            l_depth = torch.tensor(0.0, device=outputs['pred_seg'].device)

        # 3. 🔥 Normal Loss (补全逻辑)
        if 'normal' in targets and 'pred_normal' in outputs:
            l_normal = self.normal_loss_fn(outputs['pred_normal'], targets['normal'])
        else:
            l_normal = torch.tensor(0.0, device=outputs['pred_seg'].device)

        # 4. Scene Loss
        if 'scene_type' in targets and 'pred_scene' in outputs:
            l_scene = self.scene_loss_fn(outputs['pred_scene'], targets['scene_type'])
        else:
            l_scene = torch.tensor(0.0, device=outputs['pred_seg'].device)

        # 记录原始 Loss
        loss_dict['seg_loss_raw'] = l_seg.item()
        loss_dict['depth_loss_raw'] = l_depth.item()
        loss_dict['normal_loss_raw'] = l_normal.item()  # 记录
        loss_dict['scene_loss_raw'] = l_scene.item()

        # 5. 加权求和
        if self.use_uncertainty:
            # Seg
            precision_seg = 0.5 * torch.exp(-self.log_vars['seg'])
            loss_seg = precision_seg * l_seg + 0.5 * self.log_vars['seg']

            # Depth
            precision_depth = 0.5 * torch.exp(-self.log_vars['depth'])
            loss_depth = precision_depth * l_depth + 0.5 * self.log_vars['depth']

            # Normal
            precision_normal = 0.5 * torch.exp(-self.log_vars['normal'])
            loss_normal = precision_normal * l_normal + 0.5 * self.log_vars['normal']

            # Scene
            precision_scene = 0.5 * torch.exp(-self.log_vars['scene'])
            loss_scene = precision_scene * l_scene + 0.5 * self.log_vars['scene']
        else:
            w_seg = self.weights.get('lambda_seg', 1.0)
            w_depth = self.weights.get('lambda_depth', 1.0)
            w_normal = self.weights.get('lambda_normal', 1.0)  # 获取权重
            w_scene = self.weights.get('lambda_scene', 1.0)

            loss_seg = w_seg * l_seg
            loss_depth = w_depth * l_depth
            loss_normal = w_normal * l_normal  # 加权
            loss_scene = w_scene * l_scene

        total_loss = loss_seg + loss_depth + loss_normal + loss_scene

        loss_dict['total_loss'] = total_loss
        loss_dict['seg_loss'] = loss_seg
        loss_dict['depth_loss'] = loss_depth
        loss_dict['normal_loss'] = loss_normal
        loss_dict['scene_loss'] = loss_scene

        return total_loss, loss_dict