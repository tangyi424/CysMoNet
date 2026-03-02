# ---------------- sul_graphcl_pretrain.py ----------------
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph

from sul_dataset import SulDataset, collate_graph
from sul_model import SulGraphEncoder


# ============================================================
# 1) Graph Augmentation (GraphCL Standard)
# ============================================================

def drop_nodes_safe(data, drop_ratio=0.2):
    """安全版节点丢弃：支持单图和Batch，确保索引有效"""
    import copy
    if isinstance(data, Batch):
        data_list = data.to_data_list()
        new_list = []
        for g in data_list:
            new_list.append(drop_nodes_safe(g, drop_ratio))
        return Batch.from_data_list(new_list)

    # -------- 单图情况 --------
    N = data.num_nodes
    device = data.x.device
    keep_num = max(1, int(N * (1 - drop_ratio)))
    perm = torch.randperm(N, device=device)[:keep_num]
    perm = perm.sort().values

    edge_index, edge_attr = subgraph(
        perm, data.edge_index, relabel_nodes=True,
        num_nodes=N, edge_attr=getattr(data, 'edge_attr', None)
    )

    new_data = Data(
        x=data.x[perm],
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    # 保留其他属性
    for k, v in data.items():
        if k not in ['x', 'edge_index', 'edge_attr']:
            new_data[k] = v
    return new_data


def drop_edges(data, drop_ratio=0.2):
    """随机丢弃一定比例的边（设备一致性安全版）"""
    device = data.edge_index.device
    E = data.edge_index.size(1)
    keep_num = max(1, int(E * (1 - drop_ratio)))

    idx = torch.randperm(E, device=device)[:keep_num]
    data.edge_index = data.edge_index[:, idx]

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr[idx]

    return data


def mask_atom_features(data, mask_ratio=0.15):
    """遮蔽一部分节点特征"""
    x = data.x
    N = x.size(0)
    mask = (torch.rand(N, device=x.device) < mask_ratio)
    x = x.clone()
    x[mask] = 0.0
    data.x = x
    return data


def graphcl_augment(data):
    """
    GraphCL 双视角增强：节点丢弃、边丢弃、特征遮蔽
    ✅ 支持 batch 模式（对每个子图分别处理）
    ✅ 无节点越界风险
    """
    import copy
    from torch_geometric.data import Batch

    # 深拷贝以免影响原图
    view = copy.deepcopy(data)

    # 如果是Batch，则对每个子图分别增强
    if isinstance(view, Batch):
        data_list = view.to_data_list()
        new_list = []
        for g in data_list:
            if torch.rand(1).item() < 0.5:
                g = drop_nodes_safe(g, drop_ratio=0.2)
            if torch.rand(1).item() < 0.5:
                g = drop_edges(g, drop_ratio=0.2)
            if torch.rand(1).item() < 0.5:
                g = mask_atom_features(g, mask_ratio=0.15)
            new_list.append(g)
        view = Batch.from_data_list(new_list)
    else:
        # 单图模式
        if torch.rand(1).item() < 0.5:
            view = drop_nodes_safe(view, drop_ratio=0.2)
        if torch.rand(1).item() < 0.5:
            view = drop_edges(view, drop_ratio=0.2)
        if torch.rand(1).item() < 0.5:
            view = mask_atom_features(view, mask_ratio=0.15)

    return view


# ============================================================
# 2) Contrastive Loss (NT-Xent)
# ============================================================

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -9e15)

        pos = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B)
        ], dim=0).to(z.device)

        loss = -torch.log_softmax(sim, dim=1)[torch.arange(2 * B), pos]
        return loss.mean()


# ============================================================
# 3) GraphCL 预训练
# ============================================================

def pretrain_graphcl(graph_cache, device, batch_size, epochs, save_path):
    # 修改：增加 batch_size=128 并启用节点丢弃
    dataset = SulDataset(list(graph_cache.keys()))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=lambda b: collate_graph(b, graph_cache)
    )

    model = SulGraphEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = NTXentLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for graph, _ in tqdm(loader, desc=f"GraphCL Epoch {ep}/{epochs}"):
            graph = graph.to(device)
            g1 = graphcl_augment(graph.clone()).to(device)
            g2 = graphcl_augment(graph.clone()).to(device)

            z1 = model(g1)
            z2 = model(g2)

            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {ep}] Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[OK] Pretrained encoder saved → {save_path}")


# ============================================================
# 4) Embedding Extraction (uses pretrained encoder)
# ============================================================

def get_embeddings_for_graphs(graph_cache, device, batch_size, save_path, encoder_path):
    keys = list(graph_cache.keys())
    dataset = SulDataset(keys)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda b: collate_graph(b, graph_cache)
    )

    model = SulGraphEncoder().to(device)
    model.load_state_dict(torch.load(encoder_path, map_location=device))
    model.eval()

    embed_dict = {}
    start_idx = 0

    with torch.no_grad():
        for batch_graph, map_local in tqdm(loader, desc=f"Extract {save_path}"):
            batch_graph = batch_graph.to(device)
            emb = model(batch_graph)

            for i, local_idx in enumerate(map_local):
                if isinstance(local_idx, torch.Tensor):
                    local_idx = int(local_idx.item())
                key = keys[start_idx + i]
                embed_dict[key] = emb[local_idx].cpu().numpy()
            start_idx += len(map_local)

    np.savez_compressed(save_path, **embed_dict)
    print(f"[OK] Saved → {save_path}")

# ============================================================
# 5) Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="../data_2/graph_cache_sul")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pretrain", action="store_true", help="run GraphCL unsupervised pretraining")
    parser.add_argument("--extract", action="store_true", help="extract embeddings using pretrained encoder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--encoder_out", type=str, default="encoder_graphcl.pt")
    args = parser.parse_args()

    device = torch.device(args.device)

    left_cache = torch.load(os.path.join(args.cache_dir, "left_graphs.pt"))
    right_cache = torch.load(os.path.join(args.cache_dir, "right_graphs.pt"))
    full_cache = torch.load(os.path.join(args.cache_dir, "full_graphs.pt"))

    if args.pretrain:
        print("=== GraphCL Pretraining ===")
        pretrain_graphcl(full_cache, device, args.batch_size,
                         epochs=args.epochs, save_path=args.encoder_out)

    if args.extract:
        print("=== Extract embeddings ===")
        get_embeddings_for_graphs(left_cache, device, args.batch_size,
                                  "../data_2/pretrain_fea/left_atom.npz", args.encoder_out)
        get_embeddings_for_graphs(right_cache, device, args.batch_size,
                                  "../data_2/pretrain_fea/right_atom.npz", args.encoder_out)
        get_embeddings_for_graphs(full_cache, device, args.batch_size,
                                  "../data_2/pretrain_fea/full_atom.npz", args.encoder_out)


if __name__ == "__main__":
    main()


# python sul_graphcl_pretrain.py --pretrain --batch_size 128 --epochs 50

# python sul_graphcl_pretrain.py --extract --batch_size 64
