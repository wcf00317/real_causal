import torch
import torch.nn as nn
import torch.nn.functional as F
from .building_blocks import ViTEncoder, MLP, ResNetEncoder


# ==============================================================================
# Helper Classes (内置实现通用解码头，解决 Import Error)
# ==============================================================================

class SegDepthDecoder(nn.Module):
    """
    通用分割/深度/法线预测头
    结构: Conv-BN-ReLU -> Conv -> Upsample
    """

    def __init__(self, input_channels, output_channels, scale_factor=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, output_channels, kernel_size=1)
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        out = self.head(x)
        if self.scale_factor > 1:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return out


# ==============================================================================
# Models
# ==============================================================================

class SingleTaskModel(nn.Module):
    """
    Single-Task Learning (STL) Baseline.
    【已适配 ResNet & ViT，且补全 Normal 任务】
    """

    def __init__(self, model_config, data_config):
        super().__init__()
        self.config = model_config
        self.active_task = model_config.get('active_task', 'seg')
        self.img_size = tuple(data_config['img_size'])

        print(f"🔒 Initializing Single-Task Model for: [{self.active_task.upper()}]")

        # 1. Backbone (ResNet or ViT)
        encoder_name = model_config['encoder_name']
        if 'resnet' in encoder_name:
            self.encoder = ResNetEncoder(name=encoder_name, pretrained=model_config['pretrained'], dilated=True)
            target_dim = 1024
            self.resnet_adapters = nn.ModuleList([
                nn.Conv2d(in_c, target_dim, kernel_size=1) for in_c in self.encoder.feature_dims
            ])
            encoder_feature_dim = target_dim
        else:
            self.encoder = ViTEncoder(
                name=encoder_name,
                pretrained=model_config['pretrained'],
                img_size=self.img_size[0]
            )
            encoder_feature_dim = self.encoder.feature_dim
            self.resnet_adapters = None

        # 2. Projection
        combined_feature_dim = encoder_feature_dim * 4
        self.shared_dim = 512
        self.shared_proj = nn.Conv2d(combined_feature_dim, self.shared_dim, kernel_size=1)

        # 3. Task Configs
        self.num_seg_classes = model_config.get('num_seg_classes', 40)
        self.num_scene_classes = model_config.get('num_scene_classes', 27)

        # 4. Initialize Heads & Dummy Attributes

        # 统一缩放因子
        scale = 8 if 'resnet' in encoder_name else 16

        # --- Segmentation ---
        if self.active_task == 'seg':
            self.seg_head = SegDepthDecoder(input_channels=self.shared_dim, output_channels=self.num_seg_classes,
                                            scale_factor=scale)
            self.predictor_seg = self.seg_head
        else:
            self.seg_head = None
            self.predictor_seg = nn.Module()
            self.predictor_seg.output_channels = self.num_seg_classes

        # --- Depth ---
        if self.active_task == 'depth':
            self.depth_head = SegDepthDecoder(input_channels=self.shared_dim, output_channels=1, scale_factor=scale)
            self.predictor_depth = self.depth_head
        else:
            self.depth_head = None
            self.predictor_depth = nn.Module()

        # --- Normal (🔥 补全) ---
        if self.active_task == 'normal':
            self.normal_head = SegDepthDecoder(input_channels=self.shared_dim, output_channels=3, scale_factor=scale)
            self.predictor_normal = self.normal_head
        else:
            self.normal_head = None
            self.predictor_normal = nn.Module()

        # --- Scene ---
        if self.active_task == 'scene':
            self.scene_mlp = MLP(self.shared_dim, self.num_scene_classes, hidden_dim=256)
            self.predictor_scene = self.scene_mlp
        else:
            self.scene_mlp = None
            self.predictor_scene = nn.Module()
            self.predictor_scene.out_features = self.num_scene_classes

        # 5. Trainer Compatibility (Placeholders)
        self.projector_p_seg = None
        self.projector_p_depth = None
        self.proj_z_p_seg = None
        self.proj_z_p_depth = None
        self.zp_seg_refiner = None
        self.zp_depth_refiner = None
        self.decoder_zp_depth = None

    def forward(self, x, stage=None):
        B, _, H, W = x.shape

        # 1. Encoder
        raw_features = self.encoder(x)

        if self.resnet_adapters is not None:
            features = []
            target_h, target_w = raw_features[2].shape[-2:]
            for i, feat in enumerate(raw_features):
                feat = self.resnet_adapters[i](feat)
                if feat.shape[-2:] != (target_h, target_w):
                    feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                features.append(feat)
        else:
            features = raw_features

        # 2. Aggregation
        combined_feat = torch.cat(features, dim=1)
        shared_feat = self.shared_proj(combined_feat)

        outputs = {}

        # 3. Active Task Forward

        # Seg
        if self.active_task == 'seg':
            outputs['pred_seg'] = self.seg_head(shared_feat)
            if outputs['pred_seg'].shape[-2:] != (H, W):
                outputs['pred_seg'] = F.interpolate(outputs['pred_seg'], size=(H, W), mode='bilinear',
                                                    align_corners=False)
        else:
            outputs['pred_seg'] = torch.zeros(B, self.num_seg_classes, H, W, device=x.device)

        # Depth
        if self.active_task == 'depth':
            outputs['pred_depth'] = self.depth_head(shared_feat)
            if outputs['pred_depth'].shape[-2:] != (H, W):
                outputs['pred_depth'] = F.interpolate(outputs['pred_depth'], size=(H, W), mode='bilinear',
                                                      align_corners=False)
        else:
            outputs['pred_depth'] = torch.zeros(B, 1, H, W, device=x.device)

        # Normal (🔥 补全)
        if self.active_task == 'normal':
            pred_normal = self.normal_head(shared_feat)
            if pred_normal.shape[-2:] != (H, W):
                pred_normal = F.interpolate(pred_normal, size=(H, W), mode='bilinear', align_corners=False)
            outputs['pred_normal'] = F.normalize(pred_normal, p=2, dim=1)
        else:
            outputs['pred_normal'] = torch.zeros(B, 3, H, W, device=x.device)

        # Scene
        if self.active_task == 'scene':
            h = F.adaptive_avg_pool2d(shared_feat, (1, 1)).flatten(1)
            outputs['pred_scene'] = self.scene_mlp(h)
        else:
            outputs['pred_scene'] = torch.zeros(B, self.num_scene_classes, device=x.device)

        return outputs


class RawMTLModel(nn.Module):
    """
    Standard Hard Parameter Sharing Multi-Task Learning (Raw MTL).
    【已适配 ResNet & ViT，且补全 Normal 任务】
    """

    def __init__(self, model_config, data_config):
        super().__init__()
        self.config = model_config

        # 1. Shared Backbone
        encoder_name = model_config['encoder_name']
        if 'resnet' in encoder_name:
            self.encoder = ResNetEncoder(name=encoder_name, pretrained=model_config['pretrained'], dilated=True)
            target_dim = 1024
            self.resnet_adapters = nn.ModuleList([
                nn.Conv2d(in_c, target_dim, kernel_size=1) for in_c in self.encoder.feature_dims
            ])
            encoder_feature_dim = target_dim
            self.is_resnet = True
        else:
            self.encoder = ViTEncoder(
                name=encoder_name,
                pretrained=model_config['pretrained'],
                img_size=data_config['img_size'][0]
            )
            encoder_feature_dim = self.encoder.feature_dim
            self.resnet_adapters = None
            self.is_resnet = False

        # 2. Shared Projection (Bottleneck)
        combined_feature_dim = encoder_feature_dim * 4
        self.shared_dim = 512
        self.shared_proj = nn.Conv2d(combined_feature_dim, self.shared_dim, kernel_size=1)

        # 3. Task Heads
        scale = 8 if self.is_resnet else 16

        # --- Scene Head ---
        self.num_scene_classes = model_config.get('num_scene_classes', 27)
        self.scene_mlp = MLP(self.shared_dim, self.num_scene_classes, hidden_dim=256)

        # --- Seg Head ---
        self.num_seg_classes = model_config.get('num_seg_classes', 40)
        self.seg_head = SegDepthDecoder(input_channels=self.shared_dim, output_channels=self.num_seg_classes,
                                        scale_factor=scale)

        # --- Depth Head ---
        self.depth_head = SegDepthDecoder(input_channels=self.shared_dim, output_channels=1, scale_factor=scale)

        # --- Normal Head (🔥 补全) ---
        self.normal_head = SegDepthDecoder(input_channels=self.shared_dim, output_channels=3, scale_factor=scale)

        # =========================================================
        #   兼容性适配层 (Compatibility Layer)
        # =========================================================

        # 1. [For Evaluator] 提供别名
        self.predictor_seg = self.seg_head
        self.predictor_depth = self.depth_head
        self.predictor_normal = self.normal_head  # 补全
        self.predictor_scene = self.scene_mlp

        # 2. [For Trainer] 占位符
        self.projector_p_seg = None
        self.projector_p_depth = None
        self.proj_z_p_seg = None
        self.proj_z_p_depth = None
        self.zp_seg_refiner = None
        self.zp_depth_refiner = None
        self.decoder_zp_depth = None

        # 3. [For GradNorm]
        self.layer_to_norm = self.shared_proj

    def forward(self, x, stage=None):
        # 1. Encoder
        raw_features = self.encoder(x)

        if self.resnet_adapters is not None:
            features = []
            target_h, target_w = raw_features[2].shape[-2:]
            for i, feat in enumerate(raw_features):
                feat = self.resnet_adapters[i](feat)
                if feat.shape[-2:] != (target_h, target_w):
                    feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                features.append(feat)
        else:
            features = raw_features

        # 2. Aggregation
        combined_feat = torch.cat(features, dim=1)
        shared_feat = self.shared_proj(combined_feat)

        # 3. Task Predictions
        pred_seg = self.seg_head(shared_feat)
        pred_depth = self.depth_head(shared_feat)
        pred_normal = self.normal_head(shared_feat)  # 补全

        # 确保尺寸一致
        img_h, img_w = x.shape[-2:]
        if pred_seg.shape[-2:] != (img_h, img_w):
            pred_seg = F.interpolate(pred_seg, size=(img_h, img_w), mode='bilinear', align_corners=False)
        if pred_depth.shape[-2:] != (img_h, img_w):
            pred_depth = F.interpolate(pred_depth, size=(img_h, img_w), mode='bilinear', align_corners=False)
        if pred_normal.shape[-2:] != (img_h, img_w):
            pred_normal = F.interpolate(pred_normal, size=(img_h, img_w), mode='bilinear', align_corners=False)

        # Normal 归一化
        pred_normal = F.normalize(pred_normal, p=2, dim=1)

        # Scene
        h = F.adaptive_avg_pool2d(shared_feat, (1, 1)).flatten(1)
        pred_scene = self.scene_mlp(h)

        return {
            'pred_seg': pred_seg,
            'pred_depth': pred_depth,
            'pred_normal': pred_normal,  # 补全 key
            'pred_scene': pred_scene,
        }

    def get_last_shared_layer(self):
        return self.layer_to_norm