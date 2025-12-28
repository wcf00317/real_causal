import os
import argparse
import yaml
import torch
import numpy as np
import logging
import sys
from torch.utils.data import DataLoader

# --- 项目模块导入 ---
from data_utils.cityscapes_c_dataset import CityscapesCDataset
from models.causal_model import CausalMTLModel
from losses.composite_loss import CompositeLoss
from engine.evaluator import evaluate
from utils.general_utils import setup_logging, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Full Metric Testing for Cityscapes-C")

    # 核心参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重路径 (例如 runs/xxx/checkpoints/model_best.pth.tar)')
    parser.add_argument('--config', type=str, default=None,
                        help='[可选] 配置文件路径 (若 checkpoint 缺省配置)')

    # 路径覆盖
    parser.add_argument('--cc_dataset_path', type=str, default=None, help='覆盖 Cityscapes-C 路径')
    parser.add_argument('--original_gt_path', type=str, default=None, help='覆盖 GT 路径')

    # 运行参数
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--batch_size', type=int, default=None, help='覆盖 batch_size')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置日志格式
    logging.basicConfig(level=logging.INFO, format='%(message)s')  # 简化日志头，方便看表
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(f"🚀  Cityscapes-C Full Metric Evaluation")
    print(f"    Checkpoint: {args.checkpoint}")
    print("=" * 80)

    # 1. 加载 Checkpoint
    if not os.path.exists(args.checkpoint):
        logging.error(f"❌ Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except Exception as e:
        logging.error(f"❌ Failed to load checkpoint: {e}")
        sys.exit(1)

    # 2. 获取 Config
    config = None
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("✅  Loaded config from checkpoint.")
    else:
        print("⚠️  No config in checkpoint. Trying external file...")
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logging.error("❌ No config found. Please provide --config.")
            sys.exit(1)

    # 3. 覆盖路径参数
    if args.cc_dataset_path: config['data']['cc_dataset_path'] = args.cc_dataset_path
    if args.original_gt_path: config['data']['original_gt_path'] = args.original_gt_path
    if args.batch_size: config['data']['batch_size'] = args.batch_size

    set_seed(config['training'].get('seed', 42))

    # 4. 初始化模型
    try:
        model = CausalMTLModel(config['model'], config['data']).to(device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        logging.error(f"❌ Model load failed: {e}")
        sys.exit(1)

    # 5. 准备测试资源
    criterion = CompositeLoss(config['losses'], 'cityscapes2cityscapes_c').to(device)

    data_cfg = config['data']
    cc_root = data_cfg.get('cc_dataset_path')
    gt_root = data_cfg.get('original_gt_path')
    img_size = tuple(data_cfg['img_size'])
    batch_size = data_cfg['batch_size']

    # 6. 扫描 Corruption 目录
    if not cc_root or not os.path.exists(cc_root):
        logging.error(f"❌ Invalid CC Root: {cc_root}")
        sys.exit(1)

    all_corruptions = sorted([d for d in os.listdir(cc_root)
                              if os.path.isdir(os.path.join(cc_root, d))])

    if not all_corruptions:
        logging.error(f"❌ No corruption folders found in {cc_root}")
        sys.exit(1)

    # 存储最终结果 {corruption_name: {metric: avg_value}}
    final_results = {}

    print(f"\n📋  Found {len(all_corruptions)} corruptions. Starting evaluation...\n")

    # 7. 循环测试
    for corruption in all_corruptions:
        print(f">>> Testing Corruption: {corruption.upper()}")

        # 临时存储该 Corruption 下 5 个 Severity 的指标
        cor_metrics = {
            'seg_miou': [],
            'seg_pixel_acc': [],
            'depth_abs_err': [],
            'depth_rel_err': []
        }

        for severity in range(1, 6):
            test_ds = CityscapesCDataset(
                root_dir=cc_root, gt_dir=gt_root,
                corruption=corruption, severity=severity, img_size=img_size
            )

            if len(test_ds) == 0:
                print(f"    Sev {severity}: ⚠️ No images found.")
                continue

            test_loader = DataLoader(test_ds, batch_size=batch_size,
                                     shuffle=False, num_workers=4, pin_memory=True)

            # 运行评估
            # 这里的 evaluate 必须是你修改好(修复了score_fun)的版本
            metrics = evaluate(model, test_loader, criterion, device,
                               stage=2, data_type="cityscapes_c", mask_zeros=True)

            # 提取 4 个核心指标
            miou = metrics.get('seg_miou', 0.0)
            p_acc = metrics.get('seg_pixel_acc', 0.0)
            abs_err = metrics.get('depth_abs_err', 0.0)
            rel_err = metrics.get('depth_rel_err', 0.0)

            # 存入列表
            cor_metrics['seg_miou'].append(miou)
            cor_metrics['seg_pixel_acc'].append(p_acc)
            cor_metrics['depth_abs_err'].append(abs_err)
            cor_metrics['depth_rel_err'].append(rel_err)

            # 打印当前 Severity 结果
            print(
                f"    Sev {severity} | mIoU: {miou:.4f} | P.Acc: {p_acc:.4f} | Abs: {abs_err:.4f} | Rel: {rel_err:.4f}")

        # 计算该 Corruption 的平均值 (Avg over 5 severities)
        avg_results = {}
        for k, v_list in cor_metrics.items():
            if len(v_list) > 0:
                avg_results[k] = np.mean(v_list)
            else:
                avg_results[k] = 0.0

        final_results[corruption] = avg_results

        # 打印该 Corruption 的平均
        print(
            f"    👉 AVG   | mIoU: {avg_results['seg_miou']:.4f} | P.Acc: {avg_results['seg_pixel_acc']:.4f} | Abs: {avg_results['depth_abs_err']:.4f} | Rel: {avg_results['depth_rel_err']:.4f}\n")

    # 8. 最终汇总报表
    print("=" * 95)
    print("🏆  FINAL CITYSCAPES-C BENCHMARK REPORT (Average over Severity 1-5)")
    print("=" * 95)

    # 表头
    header = f"{'Corruption Type':<20} | {'mIoU':<10} | {'Pix Acc':<10} | {'Abs Err':<10} | {'Rel Err':<10}"
    print(header)
    print("-" * 95)

    # 累加器用于计算 Grand Average
    grand_totals = {'seg_miou': 0.0, 'seg_pixel_acc': 0.0, 'depth_abs_err': 0.0, 'depth_rel_err': 0.0}
    valid_count = 0

    for corr, res in final_results.items():
        # 打印行
        row = f"{corr:<20} | {res['seg_miou']:.4f}     | {res['seg_pixel_acc']:.4f}     | {res['depth_abs_err']:.4f}     | {res['depth_rel_err']:.4f}"
        print(row)

        # 累加
        if res['seg_miou'] > 0:  # 简单判断是否有效
            valid_count += 1
            for k in grand_totals:
                grand_totals[k] += res[k]

    print("-" * 95)

    # 计算 Grand Average
    if valid_count > 0:
        grand_avg = {k: v / valid_count for k, v in grand_totals.items()}
        footer = f"{'GRAND AVERAGE':<20} | {grand_avg['seg_miou']:.4f}     | {grand_avg['seg_pixel_acc']:.4f}     | {grand_avg['depth_abs_err']:.4f}     | {grand_avg['depth_rel_err']:.4f}"
        print(footer)
    else:
        print("No valid results.")

    print("=" * 95)
    print("Done.")


if __name__ == '__main__':
    main()