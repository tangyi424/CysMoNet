import numpy as np

def inspect_npz(path):
    data = np.load(path)
    keys = list(data.keys())
    count = len(keys)
    shapes = {k: data[k].shape for k in keys}

    # 假设所有 embedding shape 一样，取第一个打印
    example_shape = shapes[keys[0]]

    print(f"\nFile: {path}")
    print(f"  → 数量: {count}")
    print(f"  → embedding shape: {example_shape}")

    return data


# -------------------------
# 正样本（1）
# -------------------------
emb_full_1  = inspect_npz("/home/lichangyong/documents/tangyi/SulMoNet/data/emb_full.npz")
emb_left_1  = inspect_npz("/home/lichangyong/documents/tangyi/SulMoNet/data/emb_left.npz")
emb_right_1 = inspect_npz("/home/lichangyong/documents/tangyi/SulMoNet/data/emb_right.npz")

# -------------------------
# 负样本（2）
# -------------------------
emb_full_2  = inspect_npz("/home/lichangyong/documents/tangyi/SulMoNet/data/emb_full_2.npz")
emb_left_2  = inspect_npz("/home/lichangyong/documents/tangyi/SulMoNet/data/emb_left_2.npz")
emb_right_2 = inspect_npz("/home/lichangyong/documents/tangyi/SulMoNet/data/emb_right_2.npz")


# =========================
# 合并正负样本：full / left / right
# =========================
def merge_npz_union(dict1, dict2):
    merged = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())  # 并集

    print("总 key 数（并集）:", len(all_keys))

    for k in all_keys:
        emb1 = dict1.get(k, None)
        emb2 = dict2.get(k, None)

        if emb1 is None:
            merged[k] = emb2
        elif emb2 is None:
            merged[k] = emb1
        else:
            merged[k] = np.concatenate([emb1,], axis=0)

    return merged

# --- 保存 dict 为 npz ---
def save_npz(data_dict, output_path):
    np.savez(output_path, **data_dict)
    print(f"已保存: {output_path}, 共 {len(data_dict)} 条")

merged_full  = merge_npz_union(emb_full_1, emb_full_2)
merged_left  = merge_npz_union(emb_left_1, emb_left_2)
merged_right = merge_npz_union(emb_right_1, emb_right_2)

print("\n=== 合并后结果 ===")
print("full merged shape:", merged_full[list(merged_full.keys())[0]].shape)
print("left merged shape:", merged_left[list(merged_left.keys())[0]].shape)
print("right merged shape:", merged_right[list(merged_right.keys())[0]].shape)

save_npz(merged_full,  "../data/protbert_fea/merged_full.npz")
save_npz(merged_left,  "../data/protbert_fea/merged_left.npz")
save_npz(merged_right, "../data/protbert_fea/merged_right.npz")