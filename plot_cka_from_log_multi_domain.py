#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从多个 GTA5 -> Cityscapes 训练日志中抽取 Validation Indep(CKA)，
分别在 Target (Cityscapes) 和 Source (GTA5) 上画平滑后的曲线。

用法示例：
    python plot_cka_from_log_multi_domain.py \
        --logs runs/lambda0/run.log \
               runs/lambda0.1/run.log \
               runs/lambda1/run.log \
               runs/lambda10/run.log \
               runs/lambda100/run.log \
        --lambdas 0 0.1 1.0 10.0 100.0 \
        --out_prefix cka_da
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


# ---------- 日志解析部分 ----------

def parse_cka_both_domains(
    log_path: str,
) -> Tuple[List[int], List[float], List[int], List[float]]:
    """
    从 DA 日志中解析每个 epoch 的 Validation Indep(CKA)，区分 target/source。

    日志结构假设：
        - 每个 epoch 先跑 target val (Cityscapes)，后跑 source val (GTA5)。
        - source 段前面有一行包含 "[Val - Source]".
        - 两个段里都会有 "Avg Loss: ... | Indep(CKA): vvv".

    返回:
        target_epochs, target_cka, source_epochs, source_cka
    """
    epoch_re = re.compile(r"-----\s*Epoch\s+(\d+)/\d+")
    cka_re = re.compile(r"Indep\(CKA\):\s*([0-9.]+)")

    target_epochs: List[int] = []
    target_cka: List[float] = []
    source_epochs: List[int] = []
    source_cka: List[float] = []

    current_epoch = None
    in_source_block = False

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_epoch = epoch_re.search(line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))
                in_source_block = False  # 每个 epoch 的第一段 val 是 target
                continue

            if current_epoch is None:
                continue

            if "[Val - Source]" in line:
                in_source_block = True
                continue

            if "Avg Loss:" in line and "Indep(CKA):" in line:
                m_cka = cka_re.search(line)
                if not m_cka:
                    continue
                val = float(m_cka.group(1))
                if in_source_block:
                    source_epochs.append(current_epoch)
                    source_cka.append(val)
                else:
                    target_epochs.append(current_epoch)
                    target_cka.append(val)

    return target_epochs, target_cka, source_epochs, source_cka


# ---------- 工具函数：滑动平均 ----------

def moving_average(values: List[float], window: int = 5) -> List[float]:
    """
    简单滑动平均，用于平滑 CKA 曲线。

    对第 i 个点，取 [i-window+1, i] 区间内的平均（不足时就用已有的）。

    window <= 1 时直接返回原序列拷贝。
    """
    if window <= 1:
        return values[:]

    smoothed: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        seg = values[start:i + 1]
        smoothed.append(sum(seg) / len(seg))
    return smoothed


# ---------- 画图部分 ----------

def plot_domain_cka(
    domain_name: str,
    out_path: Path,
    all_epochs: List[List[int]],
    all_ckas: List[List[float]],
    lambdas: List[float],
    smooth_window: int = 5,
):
    """
    在单个域上画多条 λ 的 CKA 曲线（使用滑动平均平滑后绘制）。
    """
    if not all_epochs:
        print(f"[WARN] {domain_name}: 没有任何曲线可以画，跳过。")
        return

    # 统一长度：截到所有 run 的最短 epoch 数，防止长度不一致
    lengths = [len(ep) for ep in all_epochs if ep]
    if not lengths:
        print(f"[WARN] {domain_name}: 所有序列长度为 0，跳过。")
        return
    min_len = min(lengths)

    # 配色：显式指定 λ -> 颜色，避免相近颜色
    lambda_color_map = {
        0.0: "black",       # baseline
        0.1: "tab:blue",
        1.0: "tab:red",     # 重点：你关心的 λ=1
        10.0: "tab:green",
        100.0: "tab:orange",
    }
    fallback_colors = ["tab:purple", "tab:cyan", "tab:pink", "tab:olive"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_all: List[float] = []

    for i, (epochs, ckas) in enumerate(zip(all_epochs, all_ckas)):
        if not epochs:
            continue

        # 截断到统一长度
        epochs = epochs[:min_len]
        ckas = ckas[:min_len]

        # 滑动平均平滑
        ckas_smooth = moving_average(ckas, window=smooth_window)
        y_all.extend(ckas_smooth)

        lam_val = lambdas[i]
        color = lambda_color_map.get(lam_val, None)
        if color is None:
            color = fallback_colors[i % len(fallback_colors)]
        marker = markers[i % len(markers)]

        label = rf"$\lambda_{{\mathrm{{CKA}}}}={lam_val}$"

        ax.plot(
            epochs,
            ckas_smooth,
            linestyle="-",
            marker=marker,
            linewidth=2.0,
            markersize=4,
            color=color,
            label=label,
        )

    if not y_all:
        print(f"[WARN] {domain_name}: 没有有效的 CKA 数值，跳过。")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Indep (CKA)")
    ax.grid(True, alpha=0.3)

    # 图例放在上方外侧一排
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        frameon=True,
        framealpha=0.9,
        fancybox=True,
    )

    # 给上方图例留空间
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])

    # y 轴稍微留点 margin
    ymin, ymax = min(y_all), max(y_all)
    margin = (ymax - ymin) * 0.05 if ymax > ymin else 0.02
    ax.set_ylim(ymin - margin, ymax + margin)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] {domain_name} 域的 CKA 曲线已保存到: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Validation Indep(CKA) curves on both target (Cityscapes) and source (GTA5)."
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="多个 run.log 路径（顺序要和 --lambdas 一一对应）",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        required=True,
        help="对应每个 log 的 lambda_CKA 数值，比如: --lambdas 0 0.1 1 10 100",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="cka_curve",
        help="输出文件名前缀（默认: cka_curve）",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="滑动平均窗口大小（默认 5），设为 1 则不平滑。",
    )
    args = parser.parse_args()

    if len(args.logs) != len(args.lambdas):
        raise ValueError("logs 数量必须等于 lambdas 数量。")

    target_epochs_all: List[List[int]] = []
    target_cka_all: List[List[float]] = []
    source_epochs_all: List[List[int]] = []
    source_cka_all: List[List[float]] = []

    for log_path in args.logs:
        t_ep, t_cka, s_ep, s_cka = parse_cka_both_domains(log_path)
        target_epochs_all.append(t_ep)
        target_cka_all.append(t_cka)
        source_epochs_all.append(s_ep)
        source_cka_all.append(s_cka)

    prefix = Path(args.out_prefix)

    # 1) Target: Cityscapes
    out_target = prefix.with_name(prefix.name + "_target_cityscapes.png")
    plot_domain_cka(
        domain_name="Target (Cityscapes)",
        out_path=out_target,
        all_epochs=target_epochs_all,
        all_ckas=target_cka_all,
        lambdas=args.lambdas,
        smooth_window=args.smooth_window,
    )

    # 2) Source: GTA5
    out_source = prefix.with_name(prefix.name + "_source_gta5.png")
    plot_domain_cka(
        domain_name="Source (GTA5)",
        out_path=out_source,
        all_epochs=source_epochs_all,
        all_ckas=source_cka_all,
        lambdas=args.lambdas,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()
