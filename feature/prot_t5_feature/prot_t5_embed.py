#!/usr/bin/env python3
# coding: utf-8

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel


# ===============================
# 配置路径（按需修改）
# ===============================
PROTT5_DIR = "/mnt/share/xeh/Work/AMP/PLM/prot_t5_xl_half"

input_txt = "/home/lichangyong/documents/tangyi/SulMoNet/data/all_pos.txt"

output_full = "/home/lichangyong/documents/tangyi/SulMoNet/data/prott5_full.npz"
output_left = "/home/lichangyong/documents/tangyi/SulMoNet/data/prott5_left.npz"
output_right = "/home/lichangyong/documents/tangyi/SulMoNet/data/prott5_right.npz"

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
    """使用 ProtT5 获取平均 embedding"""
    result = {}

    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i+batch_size]
        sids = [x[0] for x in batch]

        # seqs_spaced 用于 tokenizer，seqs_raw 作为 result 的键
        seqs_raw = [x[1].replace(" ", "") for x in batch]     # 去掉空格的原始序列
        seqs_spaced = [x[1] for x in batch]                   # 保留空格的用于 tokenization

        # Tokenize
        tokens = tokenizer(
            seqs_spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            embeddings = model(**tokens).last_hidden_state  # [B, L, 1024]

        # 平均池化（去掉 padding）
        attention_mask = tokens.attention_mask

        for j, sid in enumerate(sids):
            mask = attention_mask[j].unsqueeze(-1)
            length = mask.sum()
            vec = (embeddings[j] * mask).sum(0) / length

            # ✅ 用原始序列作为键

            result[seqs_raw[j]] = vec.cpu().numpy()

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
