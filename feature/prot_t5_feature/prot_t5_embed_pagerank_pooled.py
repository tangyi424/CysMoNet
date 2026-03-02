#!/usr/bin/env python3
# coding: utf-8

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
import networkx as nx

# ===============================
# 配置路径（按需修改）
# ===============================
PROTT5_DIR = "/mnt/share/xeh/Work/AMP/PLM/prot_t5_xl_half"

input_txt = "/home/lichangyong/documents/tangyi/SulMoNet/data/all_neg.txt"

output_full = "/home/lichangyong/documents/tangyi/SulMoNet/data/prott5_full_2.npz"
output_left = "/home/lichangyong/documents/tangyi/SulMoNet/data/prott5_left_2.npz"
output_right = "/home/lichangyong/documents/tangyi/SulMoNet/data/prott5_right_2.npz"

batch_size = 8


# ===============================
# 加载 ProtT5 模型
# ===============================
print("Loading ProtT5-XL model...")
tokenizer = T5Tokenizer.from_pretrained(PROTT5_DIR, do_lower_case=False)
model = T5EncoderModel.from_pretrained(PROTT5_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device} ✅")


# ===============================
# 读取序列（JSON 格式）
# ===============================
with open(input_txt, "r") as f:
    seq_dict = json.load(f)

items = list(seq_dict.items())  # [(id, seq)]


# ===============================
# 准备 3 类序列
# ===============================
full_examples = []
left_examples = []
right_examples = []

for sid, seq in items:
    seq = seq.strip().replace(" ", "")
    seq = " ".join(list(seq))  # ProtT5 要求 amino acids 用空格分开

    # 固定规则(序列长度=31)
    left_seq = " ".join(list(seq_dict[sid].strip()[0:15]))
    right_seq = " ".join(list(seq_dict[sid].strip()[16:31]))

    full_examples.append((sid, seq))
    left_examples.append((sid + "_L", left_seq))
    right_examples.append((sid + "_R", right_seq))


# ===============================
# 批量 embedding 函数
# ===============================
def embed_batch(examples):
    """
    使用 ProtT5 获取基于 pixelwise 最大池化 + PageRank 的序列级 embedding
    理论步骤:
      1. 输入序列 -> Transformer -> token embedding E_tok ∈ R^{n×d} & attention A_l ∈ R^{n×n}
      2. 对所有层 attention 取像素级最大池化 → M_att = max_l A_l
      3. 构建加权图 G(M_att)
      4. 执行 PageRank 得到 token 重要性权重 α_imp
      5. 归一化 α_imp 并计算加权平均 E_seq = Σ α_i * E_tok,i
    """
    result = {}

    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i+batch_size]
        sids = [x[0] for x in batch]
        seqs_raw = [x[1].replace(" ", "") for x in batch]
        seqs_spaced = [x[1] for x in batch]

        # === Tokenize ===
        tokens = tokenizer(
            seqs_spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        # === 前向传播，输出 embedding + attention ===
        with torch.no_grad():
            outputs = model(**tokens, output_attentions=True)
            embeddings = outputs.last_hidden_state        # [B, L, D]
            attentions = torch.stack(outputs.attentions)  # [num_layers, B, num_heads, L, L]
            # 注意：部分模型的attentions为tuple，这里stack使之成为tensor

        attention_mask = tokens.attention_mask  # [B, L]

        for j, sid in enumerate(sids):
            # --- 1️⃣ 提取当前样本的 embedding & attention ---
            E_tok = embeddings[j].detach().cpu()           # [L, D]
            attn_all = attentions[:, j, :, :, :].detach().cpu()  # [num_layers, num_heads, L, L]
            mask = attention_mask[j].bool().cpu()
            n_valid = int(mask.sum())

            # 截断至有效长度（去掉padding）
            E_tok = E_tok[:n_valid]
            attn_all = attn_all[:, :, :n_valid, :n_valid]

            # --- 2️⃣ Pixelwise 最大池化 across layers & heads ---
            # 取所有层和头的元素级最大值 -> [n, n]
            M_att = attn_all.max(dim=0)[0].max(dim=0)[0].numpy()

            # --- 3️⃣ 构建图并执行 PageRank ---
            G = nx.from_numpy_array(M_att, create_using=nx.DiGraph)
            pr = nx.pagerank(G, alpha=0.85, tol=1e-6, weight='weight')

            # --- 4️⃣ 删除 CLS / END token ---
            if 0 in pr:
                del pr[0]
            if len(pr) > 0:
                del pr[max(pr.keys(), default=0)]

            # --- 5️⃣ 提取并标准化权重 ---
            alpha_imp = np.array(list(pr.values()))
            alpha_imp = alpha_imp / alpha_imp.sum()

            # --- 6️⃣ 计算加权平均 E_seq ---
            E_tok_valid = E_tok[1:len(alpha_imp)+1]  # 去掉CLS
            E_seq = torch.tensor(np.average(E_tok_valid.numpy(), axis=0, weights=alpha_imp))

            result[seqs_raw[j]] = E_seq.numpy()

    return result



# ===============================
# 生成三个 embedding
# ===============================
print("Embedding FULL sequences ...")
emb_full = embed_batch(full_examples)

print("Embedding LEFT sequences ...")
emb_left = embed_batch(left_examples)

print("Embedding RIGHT sequences ...")
emb_right = embed_batch(right_examples)


# ===============================
# 保存结果
# ===============================
np.savez_compressed(output_full, **emb_full)
np.savez_compressed(output_left, **emb_left)
np.savez_compressed(output_right, **emb_right)

print("=====================================")
print("ProtT5 Embedding 生成完毕：")
print(f"✔ full  -> {output_full}")
print(f"✔ left  -> {output_left}")
print(f"✔ right -> {output_right}")
print("=====================================")
