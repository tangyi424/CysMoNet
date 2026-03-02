#!/usr/bin/env python3
# coding: utf-8

import json
import numpy as np
from tqdm import tqdm

# ===============================
# 配置路径（按需修改）
# ===============================
pos_txt = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/pos_2.txt"
neg_txt = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/neg_2.txt"

output_full = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/blosum_fea/blosum_full.npz"
output_left = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/blosum_fea/blosum_left.npz"
output_right = "/home/lichangyong/documents/tangyi/SulMoNet/data_2/blosum_fea/blosum_right.npz"

batch_size = 8


# ===============================
# 读取正负样本（JSON 格式）
# ===============================
def load_json_sequences(path):
    with open(path, "r") as f:
        return json.load(f)

pos_dict = load_json_sequences(pos_txt)
neg_dict = load_json_sequences(neg_txt)

print(f"✅ 正样本数量: {len(pos_dict)}")
print(f"✅ 负样本数量: {len(neg_dict)}")

# 合并正负样本（序列作为唯一键，重复自动覆盖）
seq_dict = {}
for seq in pos_dict.values():
    seq_dict[seq.strip()] = seq.strip()
for seq in neg_dict.values():
    seq_dict[seq.strip()] = seq.strip()

print(f"✅ 合并后唯一序列数: {len(seq_dict)}")


# ===============================
# 准备 3 类序列（full / left / right）
# ===============================
full_examples, left_examples, right_examples = [], [], []

for seq in seq_dict.values():
    seq = seq.strip()
    left = seq[0:20]
    right = seq[21:41]

    full_examples.append((seq, seq))
    left_examples.append((seq, left))
    right_examples.append((seq, right))


# ===============================
# BLOSUM62 编码字典
# ===============================
encoding_dict = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
    'X': [0] * 20,
}


# ===============================
# 定义 encode_sequence 函数
# ===============================
def encode_sequence(sequence):
    encoded = np.zeros((len(sequence), 20), dtype=float)
    for i, aa in enumerate(sequence):
        encoded[i] = encoding_dict.get(aa, encoding_dict['X'])
    # 不做均值池化，直接展开成一维向量
    return encoded.flatten()


# ===============================
# 批量编码函数
# ===============================
def embed_batch(examples):
    result = {}
    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i + batch_size]
        for seq_id, seq in batch:
            vec = encode_sequence(seq)
            result[seq] = vec
    return result


# ===============================
# 生成三个 embedding
# ===============================
print("Encoding FULL sequences (BLOSUM62)...")
emb_full = embed_batch(full_examples)

print("Encoding LEFT sequences (BLOSUM62)...")
emb_left = embed_batch(left_examples)

print("Encoding RIGHT sequences (BLOSUM62)...")
emb_right = embed_batch(right_examples)


# ===============================
# 保存结果
# ===============================
np.savez_compressed(output_full, **emb_full)
np.savez_compressed(output_left, **emb_left)
np.savez_compressed(output_right, **emb_right)

print("=====================================")
print("BLOSUM62 编码生成完毕（序列为索引，重复自动覆盖）：")
print(f"✔ full  -> {output_full}")
print(f"✔ left  -> {output_left}")
print(f"✔ right -> {output_right}")
print("=====================================")
