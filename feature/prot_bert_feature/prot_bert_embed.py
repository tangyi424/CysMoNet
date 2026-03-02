
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# ===============================================================
# 配置路径（按需修改）
# ===============================================================
input_txt = "/home/lichangyong/documents/tangyi/SulMoNet/data/all_neg.txt"

output_full = "/home/lichangyong/documents/tangyi/SulMoNet/data/emb_full_2.npz"
output_left = "/home/lichangyong/documents/tangyi/SulMoNet/data/emb_left_2.npz"
output_right = "/home/lichangyong/documents/tangyi/SulMoNet/data/emb_right_2.npz"

PROTBERT_DIR = "/mnt/share/xeh/Work/AMP/PLM/prot_bert"
batch_size = 8

# ===============================================================
# 加载 ProtBERT 模型
# ===============================================================
print("Loading ProtBERT model from local directory ...")

tokenizer = BertTokenizer.from_pretrained(PROTBERT_DIR, do_lower_case=False)
model = BertModel.from_pretrained(PROTBERT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# ===============================================================
# 读取序列（JSON 格式）
# ===============================================================
with open(input_txt, "r") as f:
    seq_dict = json.load(f)

items = list(seq_dict.items())  # [(id, seq)]

# ===============================================================
# 准备 3 类序列
# ===============================================================
full_examples = []
left_examples = []
right_examples = []

for sid, seq in items:
    seq = seq.strip().replace(" ", "")
    seq = re.sub(r"[UZOB]", "X", seq)  # 替换非标准氨基酸
    left = seq[:15]
    right = seq[16:31]

    full_examples.append((sid, seq))
    left_examples.append((sid + "_L", left))
    right_examples.append((sid + "_R", right))

# ===============================================================
# 批量 embedding 函数
# ===============================================================
def embed_batch(examples):
    """使用 ProtBERT 获取平均 embedding，键为完整序列"""
    result = {}

    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i + batch_size]
        sids = [x[0] for x in batch]
        seqs = [x[1] for x in batch]

        # ProtBERT 输入需要用空格分隔氨基酸
        seqs_spaced = [" ".join(list(seq)) for seq in seqs]

        # Tokenize
        tokens = tokenizer(
            seqs_spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            out = model(**tokens)
            embeddings = out.last_hidden_state  # [B, L, D]
            attention_mask = tokens.attention_mask

        # 平均池化（去掉 padding）
        for j, seq in enumerate(seqs):
            mask = attention_mask[j].unsqueeze(-1)
            length = mask.sum()
            vec = (embeddings[j] * mask).sum(0) / length
            result[seq] = vec.cpu().numpy()  # ✅ 用完整序列作为键

    return result

# ===============================================================
# 生成三个 embedding
# ===============================================================
print("Embedding FULL sequences ...")
emb_full = embed_batch(full_examples)

print("Embedding LEFT sequences ...")
emb_left = embed_batch(left_examples)

print("Embedding RIGHT sequences ...")
emb_right = embed_batch(right_examples)

# ===============================================================
# 保存结果
# ===============================================================
np.savez_compressed(output_full, **emb_full)
np.savez_compressed(output_left, **emb_left)
np.savez_compressed(output_right, **emb_right)

print("=====================================")
print("生成完毕：")
print(f"✔ full -> {output_full}")
print(f"✔ left -> {output_left}")
print(f"✔ right -> {output_right}")
print("=====================================")
