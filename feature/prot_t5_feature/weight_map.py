#!/usr/bin/env python3
# coding: utf-8

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

# ===============================
# 配置
# ===============================
PROTT5_DIR = "/mnt/share/xeh/Work/AMP/PLM/prot_t5_xl_half"
input_txt = "/home/lichangyong/documents/tangyi/SulMoNet/data/all_pos.txt"
output_npz = "/home/lichangyong/documents/tangyi/SulMoNet/data/caw_average_neg.npz"
output_fig = "/home/lichangyong/documents/tangyi/SulMoNet/data/CAW_average_neg.png"

batch_size = 4   # 可调大一点（建议4~8）
window = 31      # Cys中心窗口
center_idx = window // 2

# ===============================
# 加载 ProtT5 模型
# ===============================
print("Loading ProtT5-XL model...")
tokenizer = T5Tokenizer.from_pretrained(PROTT5_DIR, do_lower_case=False)
model = T5EncoderModel.from_pretrained(PROTT5_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"✅ Model loaded on {device}")

# ===============================
# 工具函数
# ===============================
def extract_cys_window(seq, window=31):
    seq = seq.strip().upper().replace(" ", "")
    if "C" not in seq:
        return None
    cys_idx = seq.index("C")
    half = window // 2
    start = max(0, cys_idx - half)
    end = min(len(seq), cys_idx + half + 1)
    subseq = seq[start:end]
    # pad
    if len(subseq) < window:
        pad_left = max(0, half - cys_idx)
        pad_right = max(0, (cys_idx + half + 1) - len(seq))
        subseq = "A"*pad_left + subseq + "A"*pad_right
    return subseq


def compute_caw_from_attn(attn_all, center_idx=15):
    """计算平均注意力矩阵与中心导向权重"""
    A = attn_all.mean(dim=(0,1))   # [L,L]
    A_sym = (A + A.T) / 2
    w = A_sym[center_idx] / A_sym[center_idx].sum()
    return A_sym.cpu().numpy(), w.cpu().numpy()


def average_attention(A_list, w_list):
    """平均注意力矩阵与权重"""
    A_mean = np.mean(np.stack(A_list), axis=0)
    w_mean = np.mean(np.stack(w_list), axis=0)
    return A_mean, w_mean


def plot_caw(A_mean, w_mean, save_path):
    n = A_mean.shape[0]
    rel_pos = np.arange(-n//2 + 1, n//2 + 1)

    fig = plt.figure(figsize=(7,8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3,1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(A_mean, cmap="coolwarm", center=0.5,
                cbar_kws={"label":"Mean symmetric attention"},
                ax=ax1, square=True)
    ax1.axvline(center_idx, color='red', linestyle='--', lw=1)
    ax1.axhline(center_idx, color='red', linestyle='--', lw=1)
    ax1.set_title("Mean Cys-centered Symmetric Attention Map")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(rel_pos, w_mean, '-o', color='steelblue', lw=2, markersize=4)
    ax2.axvline(0, color='red', linestyle='--', lw=1, label='Cys site')
    ax2.set_xlabel("Position relative to Cys")
    ax2.set_ylabel("Normalized weight")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved figure: {save_path}")

# ===============================
# 主处理逻辑
# ===============================
def main():
    # === 读入序列 ===
    seqs = []
    with open(input_txt) as f:
        for line in f:
            seq = line.strip()
            if not seq: continue
            window_seq = extract_cys_window(seq, window)
            if window_seq:
                seqs.append(window_seq)
    print(f"Total valid sequences with Cys: {len(seqs)}")

    A_list, w_list = [], []

    # === 批处理 ===
    for i in tqdm(range(0, len(seqs), batch_size), desc="Batched inference"):
        batch = seqs[i:i+batch_size]
        batch_spaced = [" ".join(list(s)) for s in batch]

        # tokenize
        tokens = tokenizer(batch_spaced, return_tensors="pt",
                           padding=True, truncation=True,
                           max_length=window).to(device)

        with torch.no_grad():
            outputs = model(**tokens, output_attentions=True)
            attns = torch.stack(outputs.attentions)  # [LAYER,B,HEAD,T,T]

        for j in range(len(batch)):
            attn_all = attns[:, j, :, :, :]  # [num_layers,num_heads,L,L]
            A_sym, w = compute_caw_from_attn(attn_all, center_idx)
            A_list.append(A_sym)
            w_list.append(w)

    # === 平均 ===
    A_mean, w_mean = average_attention(A_list, w_list)
    np.savez_compressed(output_npz, A_mean=A_mean, w_mean=w_mean)
    print(f"✅ Saved averaged attention: {output_npz}")

    # === 绘图 ===
    plot_caw(A_mean, w_mean, output_fig)


if __name__ == "__main__":
    main()
