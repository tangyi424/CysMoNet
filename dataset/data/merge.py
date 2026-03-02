import os
import numpy as np
from tqdm import tqdm

def merge_npz(file_a, file_b, out_path):
    """将两个 .npz 文件按 key 合并，embedding 向量拼接"""
    data_a = np.load(file_a)
    data_b = np.load(file_b)

    keys_a = set(data_a.keys())
    keys_b = set(data_b.keys())
    common_keys = keys_a & keys_b

    print(f"Merging {os.path.basename(file_a)} + {os.path.basename(file_b)}")
    print(f"共有 {len(common_keys)} 个共同 key")

    merged = {}
    for k in tqdm(sorted(common_keys)):
        vec_a = data_a[k]
        vec_b = data_b[k]
        merged[k] = np.concatenate([vec_a.flatten(), vec_b.flatten()], axis=0)

    np.savez_compressed(out_path, **merged)
    print(f"[OK] Saved merged file → {out_path}\n")


def find_npz_by_suffix(folder, suffix):
    """在文件夹中查找指定后缀的 npz 文件"""
    for f in os.listdir(folder):
        if f.endswith(f"_{suffix}.npz"):
            return os.path.join(folder, f)
    return None


def merge_all(folder_a, folder_b, out_folder):
    """根据 suffix (full, left, right) 匹配合并"""
    os.makedirs(out_folder, exist_ok=True)
    suffixes = ["full", "left", "right"]

    for sfx in suffixes:
        file_a = find_npz_by_suffix(folder_a, sfx)
        file_b = find_npz_by_suffix(folder_b, sfx)
        out_path = os.path.join(out_folder, f"merged_{sfx}.npz")

        if not file_a or not file_b:
            print(f"[WARN] 未找到 {sfx} 对应的文件，跳过。")
            continue

        merge_npz(file_a, file_b, out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按后缀匹配合并两个文件夹中的对应 npz 文件")
    parser.add_argument("--folder_a", type=str, default="./pretrain_fea",  help="第一个文件夹路径")
    parser.add_argument("--folder_b", type=str, default="./blosum_fea",  help="第二个文件夹路径")
    parser.add_argument("--out", type=str, default="./atom+blosum62", help="输出文件夹")
    args = parser.parse_args()

    merge_all(args.folder_a, args.folder_b, args.out)
