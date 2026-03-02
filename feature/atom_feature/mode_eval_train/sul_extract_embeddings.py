# ---------------- sul_extract_embeddings.py ----------------
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sul_dataset import SulDataset, collate_graph
from sul_model import SulGraphEncoder


def get_embeddings_for_graphs(graph_cache, device, batch_size, save_path):
    """
    graph_cache : {key: Data}
    """
    keys = list(graph_cache.keys())
    dataset = SulDataset(keys)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_graph(b, graph_cache)
    )

    model = SulGraphEncoder().to(device)
    model.eval()

    embed_dict = {}
    start_idx = 0  # <- 维护当前 batch 在 keys 中的起始索引

    with torch.no_grad():
        for batch_graph, map_local in tqdm(loader, desc=f"Extracting {save_path}"):
            batch_graph = batch_graph.to(device)

            # [B, hidden]
            emb = model(batch_graph)

            # 分配到字典
            # 假设 map_local 是可迭代的，长度等于本 batch 的图数量（即实际 batch size）
            for i, local_idx in enumerate(map_local):
                key = keys[start_idx + i]
                # local_idx 可能是 int 或 tensor，需要确保能索引 emb
                if isinstance(local_idx, torch.Tensor):
                    local_idx = int(local_idx.item())
                embed_dict[key] = emb[local_idx].cpu().numpy()

            start_idx += len(map_local)

    # 保存 npz
    np.savez_compressed(save_path, **embed_dict)
    print(f"[OK] Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./data/graph_cache_sul")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load caches
    left_cache  = torch.load(os.path.join(args.cache_dir, "left_graphs.pt"))
    right_cache = torch.load(os.path.join(args.cache_dir, "right_graphs.pt"))
    full_cache  = torch.load(os.path.join(args.cache_dir, "full_graphs.pt"))

    # Extract
    get_embeddings_for_graphs(left_cache,  device, args.batch_size, "left_atom.npz")
    get_embeddings_for_graphs(right_cache, device, args.batch_size, "right_atom.npz")
    get_embeddings_for_graphs(full_cache,  device, args.batch_size, "full_atom.npz")


if __name__ == "__main__":
    main()
