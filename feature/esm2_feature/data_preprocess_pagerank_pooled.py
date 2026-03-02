#!/usr/bin/env python3
# coding: utf-8

import json
import numpy as np
import torch
import esm
from tqdm import tqdm
import networkx as nx
# ===============================
# 配置路径（按需修改）
# ===============================
input_txt = "/home/lichangyong/documents/tangyi/SulMoNet/data/all_neg.txt"

output_full = "/home/lichangyong/documents/tangyi/SulMoNet/data/emb_full_2.npz"
output_left = "/home/lichangyong/documents/tangyi/SulMoNet/data/emb_left_2.npz"
output_right = "/home/lichangyong/documents/tangyi/SulMoNet/data/emb_right_2.npz"

batch_size = 8


# ===============================
# 加载模型
# ===============================
print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


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
    seq = seq.strip()

    # 固定规则(序列长度=31)
    left = seq[0:15]     # 1--15
    right = seq[16:31]   # 17--31

    full_examples.append((sid, seq))
    left_examples.append((sid + "_L", left))
    right_examples.append((sid + "_R", right))


# ===============================
# 批量 embedding 函数
# ===============================
def embed_batch(examples):
    """
    使用 ESM-2 获取基于 pixelwise 最大池化 + PageRank 的序列级 embedding
    理论步骤:
      1. 输入序列 -> Transformer -> token embedding E_tok ∈ R^{n×d} & attention A_l ∈ R^{n×n}
      2. 对所有层 attention 取像素级最大池化 → M_att = max_l A_l
      3. 构建加权图 G(M_att)
      4. 执行 PageRank 得到 token 重要性权重 α_imp
      5. 归一化 α_imp 并计算加权平均 E_seq = Σ α_i * E_tok,i
    """
    result = {}

    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i + batch_size]
        sids = [x[0] for x in batch]
        seqs_raw = [x[1].replace(" ", "") for x in batch]

        # === ESM2 的输入批处理 ===
        batch_data = [(sid, seq) for sid, seq in zip(sids, seqs_raw)]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        # === 前向传播 ===
        with torch.no_grad():
            out = model(tokens, repr_layers=[33], return_contacts=False, need_head_weights=True)
            embeddings = out["representations"][33]        # [B, L, D]
            attentions = out["attentions"]       # [num_layers, B, num_heads, L, L]

        for j, sid in enumerate(sids):
            # --- 1️⃣ 提取当前样本的 embedding & attention ---
            E_tok = embeddings[j].detach().cpu()           # [L, D]
            attn_all = attentions[:, j, :, :, :].detach().cpu()  # [num_layers, num_heads, L, L]

            # 去除 padding token（ESM 中 0 是 padding）
            mask = (tokens[j] != alphabet.padding_idx).cpu()
            n_valid = int(mask.sum())

            E_tok = E_tok[:n_valid]
            attn_all = attn_all[:, :, :n_valid, :n_valid]

            # --- 2️⃣ Pixelwise 最大池化 across layers & heads ---
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
print("生成完毕：")
print(f"✔ full  -> {output_full}")
print(f"✔ left  -> {output_left}")
print(f"✔ right -> {output_right}")
print("=====================================")
