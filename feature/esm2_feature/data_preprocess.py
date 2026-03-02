#!/usr/bin/env python3
# coding: utf-8

import json
import numpy as np
import torch
import esm
from tqdm import tqdm

# ===============================
# 配置路径（按需修改）
# ===============================
input_txt = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/neg_2.txt"

output_full = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/emb_full_2.npz"
output_left = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/emb_left_2.npz"
output_right = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/emb_right_2.npz"

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
    result = {}
    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i+batch_size]

        labels, strs, tokens = batch_converter(batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, repr_layers=[33], return_contacts=False)
            reps = out["representations"][33]  # (B, L, D)

        for b_idx, (_, seq) in enumerate(batch):
            # token 长度：包含 BOS/EOS
            L = (tokens[b_idx] != alphabet.padding_idx).sum().item()

            # 去除 BOS/EOS 再均值
            vec = reps[b_idx, 1:L-1].mean(0).cpu().numpy()

            result[seq] = vec

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
