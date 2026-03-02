#!/usr/bin/env python3
# coding: utf-8

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import esm


# =====================================================
# 配置
# =====================================================
# 替换为 S-Carboxyethylation 数据路径
input_pos = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/pos_2.txt"
input_neg = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/neg_2.txt"
out_dir = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/"

os.makedirs(out_dir, exist_ok=True)
batch_size = 4
window = 41   # ✅ 改为 41
center_idx = window // 2   # = 20


# =====================================================
# 加载 ESM-2 模型
# =====================================================
print("🔹 Loading ESM-2 model ...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")


# =====================================================
# 功能函数
# =====================================================
def extract_cys_window(seq, window=41):
    seq = seq.strip().upper().replace(" ", "")
    if "C" not in seq:
        return None
    cys_idx = seq.index("C")
    half = window // 2
    start = max(0, cys_idx - half)
    end = min(len(seq), cys_idx + half + 1)
    subseq = seq[start:end]
    if len(subseq) < window:
        pad_left = max(0, half - cys_idx)
        pad_right = max(0, (cys_idx + half + 1) - len(seq))
        subseq = "A" * pad_left + subseq + "A" * pad_right
    return subseq


def compute_caw_from_attn(attn_all, center_idx=20):
    """
    从 ESM-2 注意力计算对称矩阵和中心注意力曲线
    attn_all: [num_layers, num_heads, L, L]
    """
    A = attn_all.mean(dim=(0, 1))  # 平均层和head
    A_sym = (A + A.T) / 2
    w = A_sym[center_idx] / A_sym[center_idx].sum()
    return A_sym.cpu().numpy(), w.cpu().numpy()


def average_attention(A_list, w_list):
    A_mean = np.mean(np.stack(A_list), axis=0)
    w_mean = np.mean(np.stack(w_list), axis=0)
    return A_mean, w_mean


import json
import re

def get_sequences_from_file(file_path, window=41):
    seqs = []
    with open(file_path) as f:
        try:
            data = json.load(f)
            seq_iter = data.values()
        except json.JSONDecodeError:
            # 若不是标准JSON格式则退回行读
            f.seek(0)
            seq_iter = [line.strip() for line in f if line.strip()]

    for seq in seq_iter:
        seq = seq.strip().upper()
        seq = re.sub(r"[^A-Z]", "", seq)
        subseq = extract_cys_window(seq, window)
        if subseq:
            seqs.append(subseq)

    return seqs


# =====================================================
# ESM-2 注意力提取
# =====================================================
def compute_mean_attention_esm2(seqs, label, save_prefix):
    A_list, w_list = [], []
    for i in tqdm(range(0, len(seqs), batch_size), desc=f"[{label}] batches"):
        batch = seqs[i:i + batch_size]
        batch_data = [(str(idx), seq) for idx, seq in enumerate(batch)]
        batch_labels, batch_strs, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, need_head_weights=True)
            attns = out["attentions"]  # [num_layers, B, num_heads, L, L]

        for j in range(len(batch)):
            attn_all = attns[:, j, :, :, :].detach().cpu()
            A_sym, w = compute_caw_from_attn(attn_all, center_idx)
            A_list.append(A_sym)
            w_list.append(w)

    A_mean, w_mean = average_attention(A_list, w_list)
    np.savez_compressed(f"{save_prefix}_{label}.npz", A_mean=A_mean, w_mean=w_mean)
    print(f"✅ Saved: {save_prefix}_{label}.npz")
    return A_mean, w_mean


# =====================================================
# 绘图函数：差异热图 + 曲线
# =====================================================
def plot_dual_results(A_pos, w_pos, A_neg, w_neg, save_prefix):
    n = A_pos.shape[0]
    rel_pos = np.arange(-n // 2 + 1, n // 2 + 1)
    diff = A_pos - A_neg

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 600
    })

    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 0.8], hspace=0.4)

    # --- 差异热图（S-巯基化特异性注意力模式）---
    ax1 = fig.add_subplot(gs[0])

    # 创建热图
    im = sns.heatmap(
        diff,
        cmap="bwr",
        center=0.0,
        cbar_kws={
            "label": "Attention Difference\n(Positive - Negative)",
            "extend": "both",
            # "fraction": 0.046,  # 控制 colorbar 的宽度（相对于主图）
            # "pad": 0.04,  # 控制 colorbar 与主图的间距
            "shrink": 0.9  # 控制 colorbar 的长度比例
        },
        ax=ax1,
        square=True,
    )

    # 添加垂直和水平参考线（中心Cys位置）
    ax1.axvline(n // 2, color='black', linestyle='--', lw=1, alpha=0.7)
    ax1.axhline(n // 2, color='black', linestyle='--', lw=1, alpha=0.7)

    # 添加颜色图例说明
    # ax1.text(0.02, 0.98,
    #          "ESM-2 Attention Analysis for S-sulfhydration Prediction",
    #          transform=ax1.transAxes,
    #          fontsize=12,
    #          fontweight='bold',
    #          verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # 增强颜色条说明
    cbar = im.collections[0].colorbar
    cbar.set_label("Attention Difference Map",
                    fontsize=11) # fontweight='bold',

    # 在颜色条上添加文字说明
    cbar.ax.text(1.65, 1.05, "Higher in\nPositive Class",
                 transform=cbar.ax.transAxes,
                 fontsize=9, fontweight='bold',
                 color='red',
                 verticalalignment='bottom',
                 horizontalalignment='center')
    cbar.ax.text(1.75, -0.06, "Higher in\nNegative Class",
                 transform=cbar.ax.transAxes,
                 fontsize=9, fontweight='bold',
                 color='darkblue',
                 verticalalignment='top',
                 horizontalalignment='center')

    ax1.set_title("ESM-2 Attention Analysis for S-Carboxyethylation Prediction",
                  pad=12, fontweight="bold", fontsize=14)  #pad=12
    ax1.set_xlabel("Residue Position")
    ax1.set_ylabel("Residue Position")

    # 标记中心Cys位置
    ax1.text(n // 2 + 0.5, n // 2 + 0.5, 'Cys',
             fontsize=10, fontweight='bold',
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))

    # --- 半胱氨酸中心注意力权重曲线 ---
    ax2 = fig.add_subplot(gs[1])

    # 绘制两条曲线
    ax2.plot(rel_pos, w_pos, '-o', color='crimson', lw=2, ms=5,
             label='Positive', alpha=0.8)
    ax2.plot(rel_pos, w_neg, '-o', color='steelblue', lw=2, ms=5,
             label='Negative', alpha=0.8)

    # 添加中心线（Cys位置）
    ax2.axvline(0, color='black', linestyle='--', lw=1.5, alpha=0.7,
                label='Central Cysteine')

    # 填充差异区域
    # ax2.fill_between(rel_pos, w_pos, w_neg, where=(w_pos > w_neg),
    #                  color='red', alpha=0.15, label='Higher in S-sulfhydrated')
    # ax2.fill_between(rel_pos, w_pos, w_neg, where=(w_neg > w_pos),
    #                  color='blue', alpha=0.15, label='Higher in Native')

    ax2.set_xlabel("Position Relative to Central Cysteine")
    ax2.set_ylabel("Normalized Attention Weight")
    ax2.legend(frameon=True, framealpha=0.9, loc="best", fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')

    ax2.set_title("Cys-centered Attention-derived Weight",
                  pad=12, fontweight="bold", fontsize=14)

    # 添加模型信息标注
    ax2.text(0.98, 0.02, "Model: ESM-2 | Task: S-Carboxyethylation Prediction",
             transform=ax2.transAxes,
             fontsize=9, fontstyle='italic',
             horizontalalignment='right',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_esm2_sulfhydration_analysis.png",
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Saved figure: {save_prefix}_esm2_sulfhydration_analysis.png")


# =====================================================
# 主流程
# =====================================================
def main():
    seqs_pos = get_sequences_from_file(input_pos, window)
    seqs_neg = get_sequences_from_file(input_neg, window)
    print(f"Positive: {len(seqs_pos)} sequences, Negative: {len(seqs_neg)} sequences")

    A_pos, w_pos = compute_mean_attention_esm2(seqs_pos, "pos", os.path.join(out_dir, "CAW_SCarboxyethylation"))
    A_neg, w_neg = compute_mean_attention_esm2(seqs_neg, "neg", os.path.join(out_dir, "CAW_SCarboxyethylation"))

    plot_dual_results(A_pos, w_pos, A_neg, w_neg, os.path.join(out_dir, "CAW_SCarboxyethylation"))
    print("✅ All done.")


if __name__ == "__main__":
    main()
