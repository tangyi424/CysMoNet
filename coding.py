#!/usr/bin/env python3
# coding: utf-8

import json

# ===============================
# 配置路径（自行修改）
# ===============================
input_file = "independent_set_test.fasta"  # 原始序列文件
output_pos = "./data_2/pos_2.txt"        # 正样本输出
output_neg = "./data_2/neg_2.txt"        # 负样本输出

# ===============================
# 读取 fasta-like 文件
# ===============================
pos_dict, neg_dict = {}, {}
with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

i_pos, i_neg = 1, 1
for idx in range(0, len(lines), 2):
    header = lines[idx]
    seq = lines[idx + 1]

    if header.startswith(">+"):  # 正样本
        pos_dict[str(i_pos)] = seq
        i_pos += 1
    elif header.startswith(">-"):  # 负样本
        neg_dict[str(i_neg)] = seq
        i_neg += 1

# ===============================
# 保存为 JSON 格式的 .txt 文件
# ===============================
with open(output_pos, "w") as f:
    json.dump(pos_dict, f, indent=2)

with open(output_neg, "w") as f:
    json.dump(neg_dict, f, indent=2)

# ===============================
# 输出信息
# ===============================
print(f"✅ 正样本保存至: {output_pos} ({len(pos_dict)} 条)")
print(f"✅ 负样本保存至: {output_neg} ({len(neg_dict)} 条)")
