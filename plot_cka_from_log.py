#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从多个训练日志中抽取 Validation Indep(CKA) 并画在同一张图上。

用法：
    python plot_cka_from_log.py \
        --logs runs/2026-01-03_11-48/run.log \
               runs/2026-01-03_11-49/run.log \
               runs/2026-01-03_11-51/run.log \
               runs/2026-01-03_14-/run.log
               runs/2026-01-03_11-52/run.log \
        --lambdas 0 0.1 1.0 10.0 100.0
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def extract_val_cka_from_log(log_path: str) -> Tuple[List[int], List[float]]:
    """
    只从 log 中解析 Validation 阶段的 Indep(CKA) 序列。

    返回:
        epochs: List[int]  每个 epoch 的编号
        cka   : List[float] 对应的 Indep(CKA) 数值
    """
    cur_epoch = None
    epochs, cka_values = [], []
    seen_val_epoch = set()

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 匹配 epoch 行：----- Epoch 6/100 (Stage 2) | lr=...
            m_epoch = re.search(r"-----\s*Epoch\s+(\d+)\s*/\s*\d+", line)
            if m_epoch:
                cur_epoch = int(m_epoch.group(1))
                continue

            if cur_epoch is None:
                continue

            # 验证阶段 CKA： "Avg Loss: ... | Indep(CKA): 0.7491 | ..."
            if "Avg Loss:" in line and "Indep(CKA):" in line:
                if cur_epoch not in seen_val_epoch:
                    m_val = re.search(r"Indep\(CKA\):\s*([0-9.]+)", line)
                    if m_val:
                        epochs.append(cur_epoch)
                        cka_values.append(float(m_val.group(1)))
                        seen_val_epoch.add(cur_epoch)
                continue

    return epochs, cka_values


def plot_multi_cka(
    log_paths: List[str],
    lambdas: List[str] = None,
    out_name: str = "cka_curve.png",
):
    """
    从多个 log 中抽取 Validation CKA 并画图。

    参数:
        log_paths: 多个 run.log 路径
        lambdas  : 与 log_paths 对应的 lambda 值（字符串），用于图例。
                   如果为 None，则使用文件名作为图例。
        out_name : 输出文件名（保存到当前工作目录）
    """
    if lambdas is not None and len(lambdas) != len(log_paths):
        raise ValueError("len(lambdas) 必须等于 len(log_paths)。")

    # 论文友好的全局样式
    plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    fig, ax = plt.subplots()

    # 不同曲线用不同线型/marker，方便区分
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "v"]

    y_all = []

    for i, log_path in enumerate(log_paths):
        epochs, cka_vals = extract_val_cka_from_log(log_path)
        if not epochs:
            print(f"[WARN] {log_path} 中没有解析到 Validation CKA，跳过。")
            continue

        y_all.extend(cka_vals)

        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]

        if lambdas is not None:
            label = r"$\lambda_{\mathrm{CKA}}=" + str(lambdas[i]) + r"$"
        else:
            label = Path(log_path).name

        ax.plot(
            epochs,
            cka_vals,
            linestyle=ls,
            marker=mk,
            linewidth=2.0,
            markersize=4,
            label=label,
        )

    if not y_all:
        raise RuntimeError("所有日志中都没有解析到 Validation CKA，请检查 log 格式。")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Indep (CKA)")
    ax.grid(True, alpha=0.3)

    # 固定图例位置在右上角，白底稍透明，避免挡住曲线
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fancybox=True,
    )

    # y 轴稍微留一点上下边距
    ymin, ymax = min(y_all), max(y_all)
    margin = (ymax - ymin) * 0.05 if ymax > ymin else 0.02
    ax.set_ylim(ymin - margin, ymax + margin)

    fig.tight_layout()

    out_path = Path.cwd() / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] 多 run 的 Validation CKA 曲线已保存到: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot validation Indep(CKA) over epochs for multiple runs."
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="多个 run.log 的路径（顺序应与 lambdas 对应）",
    )
    parser.add_argument(
        "--lambdas",
        nargs="*",
        help="与每个 log 对应的 lambda_CKA 值，比如: --lambdas 0 0.1 1 10 100",
    )
    parser.add_argument(
        "--out",
        default="cka_curve.png",
        help="输出图片文件名（默认: cka_curve.png，保存在当前目录）",
    )

    args = parser.parse_args()
    plot_multi_cka(args.logs, args.lambdas, out_name=args.out)


if __name__ == "__main__":
    main()
