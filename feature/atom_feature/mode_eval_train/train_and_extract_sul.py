#!/usr/bin/env python3
"""
Usage examples:

# 训练并抽取 (默认用 cuda:0)
python train_and_extract_sul.py --cache_dir ./data/graph_cache_sul --all_pos all_pos.txt --all_neg all_neg.txt --device cuda --epochs 20

"""
import json
import os
import argparse
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 下面两个模块按你项目结构导入
# collate_graph(keys, graph_cache) -> (batch_graph, map_local)
# SulGraphEncoder (as defined in your message)
from sul_dataset import collate_graph  # must exist
from sul_model import SulGraphEncoder    # must exist and be the class you posted
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# ------------------------------
# small helper: read pos/neg files
# ------------------------------
def read_id_file(path: str) -> List[str]:
    """每行一个 id（或 json 格式行也可，但这里假设每行是 id）"""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


# ---------------------------------------
# Supervised dataset wrapper (返回 key, label)
# ---------------------------------------
class SulSupervisedDataset(Dataset):
    def __init__(self, keys: List[str], labels_map: Dict[str, int]):
        self.keys = keys
        self.labels_map = labels_map

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        label = self.labels_map[key]
        return key, label


# ---------------------------------------
# collate wrapper: 将 list[(key,label)] 转回 collate_graph 所需的 keys list
# 并返回 (batch_graph, map_local, labels_tensor, batch_keys)
# ---------------------------------------
def collate_with_labels(batch, graph_cache):
    """
    batch: list of (key, label)
    graph_cache: dict of prebuilt Data objects
    returns: batch_graph (pyg Batch), map_local, labels_tensor, batch_keys(list)
    """
    keys = [kv[0] for kv in batch]
    labels = [kv[1] for kv in batch]
    batch_graph, map_local = collate_graph(keys, graph_cache)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return batch_graph, map_local, labels_tensor, keys


# ---------------------------------------
# Wrap encoder + linear head for supervised training
# ---------------------------------------
class GraphWithHead(nn.Module):
    def __init__(self, encoder: SulGraphEncoder, emb_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(emb_dim, 2)

    def forward(self, batch_graph):
        # encoder returns [B, emb_dim]
        emb = self.encoder(batch_graph)
        logits = self.head(emb)
        return logits, emb


# ---------------------------------------
# training loop
# ---------------------------------------
def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_graph, map_local, labels, keys in tqdm(loader, desc="Train", leave=False):
        batch_graph = batch_graph.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(batch_graph)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def eval_epoch(model: nn.Module, loader: DataLoader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probs = []  # prob of class-1
    all_preds = []

    with torch.no_grad():
        for batch_graph, map_local, labels, keys in tqdm(loader, desc="Val", leave=False):
            batch_graph = batch_graph.to(device)
            labels = labels.to(device)

            logits, _ = model(batch_graph)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # predictions
            probs = torch.softmax(logits, dim=1)[:, 1]   # prob of class 1
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # collect for AUC / MCC
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0.0

    # ---- Compute AUC ----
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # e.g., only one class in validation set

    # ---- Compute MCC ----
    try:
        mcc = matthews_corrcoef(all_labels, all_preds)
    except ValueError:
        mcc = 0.0

    return avg_loss, acc, auc, mcc


# ---------------------------------------
# Use trained encoder to extract embeddings (uses start_idx accumulation)
# model_encoder: SulGraphEncoder (not including head)
# graph_cache: dict {key: Data}
# ---------------------------------------
def get_embeddings_for_graphs_with_model(model_encoder, graph_cache, device, batch_size, save_path):
    keys = list(graph_cache.keys())
    dataset = SulSupervisedDataset(keys, {k: 0 for k in keys})  # labels dummy, we won't use them in collate
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_with_labels(b, graph_cache)
    )

    model_encoder.eval()
    embed_dict = {}
    start_idx = 0

    with torch.no_grad():
        for batch_graph, map_local, _, batch_keys in tqdm(loader, desc=f"Extracting {save_path}"):
            batch_graph = batch_graph.to(device)
            emb = model_encoder(batch_graph)  # [B, emb_dim]

            # map each key in batch_keys to corresponding emb row
            # assume emb rows correspond to order of batch_keys (collate_graph must preserve order)
            for i, key in enumerate(batch_keys):
                # local_idx may not be needed if emb is already aligned; to be safe use i
                embed_dict[key] = emb[i].cpu().numpy()
            start_idx += len(batch_keys)

    np.savez_compressed(save_path, **embed_dict)
    print(f"[OK] Saved → {save_path}")


# ---------------------------------------
# main
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./data/graph_cache_sul")
    parser.add_argument("--all_pos", type=str, default="./data/all_pos.txt")
    parser.add_argument("--all_neg", type=str, default="./data/all_neg.txt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    def load_pos_neg_data(pos_path, neg_path):
        # --- load positives ---
        with open(pos_path, "r", encoding="utf8") as f:
            pos_dict = json.load(f)  # {"1": "...", "2": "...", ...}
        positives = list(pos_dict.values())  # values 是序列

        # --- load negatives ---
        with open(neg_path, "r", encoding="utf8") as f:
            neg_dict = json.load(f)
        negatives = list(neg_dict.values())  # values 是序列

        # --- 生成标签映射 ---
        labels_map = {}
        for seq in positives:
            labels_map[seq] = 1  # 正样本标签
        for seq in negatives:
            labels_map[seq] = 0  # 负样本标签

        return positives, negatives, labels_map

    positives, negatives, labels_map = load_pos_neg_data(args.all_pos, args.all_neg)
    print(f"Loaded positives: {len(positives)}, negatives: {len(negatives)}")
    all_keys = list(labels_map.keys())

    # -- load graph caches --
    left_cache_path = os.path.join(args.cache_dir, "left_graphs.pt")
    right_cache_path = os.path.join(args.cache_dir, "right_graphs.pt")
    full_cache_path = os.path.join(args.cache_dir, "full_graphs.pt")

    left_cache  = torch.load(left_cache_path)
    right_cache = torch.load(right_cache_path)
    full_cache  = torch.load(full_cache_path)
    print("Loaded graph caches:", left_cache_path, right_cache_path, full_cache_path)



    # -- split train/val (stratified simple split) --
    random.shuffle(all_keys)
    n_val = max(1, int(len(all_keys) * args.val_ratio))
    val_keys = all_keys[:n_val]
    train_keys = all_keys[n_val:]
    print(f"Train size: {len(train_keys)}, Val size: {len(val_keys)}")

    # datasets & loaders (we'll train on 'full' graph view —可改为left/right/full单独训练)
    train_dataset = SulSupervisedDataset(train_keys, labels_map)
    val_dataset   = SulSupervisedDataset(val_keys, labels_map)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_with_labels(b, full_cache)  # 这里用 full_cache 做训练特征（你可以换 left/right）
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_with_labels(b, full_cache)
    )

    # model init
    backbone = SulGraphEncoder().to(device)  # 这里默认 hidden_dim 与模型内部一致
    # we need to know embedding dim — infer by a forward with a tiny fake batch if necessary,
    # but assume encoder.out_dim == hidden_dim you used e.g., 128. If different adjust emb_dim.
    EMB_DIM = 128  # 如果你的 SulGraphEncoder 返回 384，修改为 384
    model = GraphWithHead(backbone, emb_dim=EMB_DIM).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_ckpt = os.path.join(args.save_dir, "sul_encoder_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, auc, mcc = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={auc:.4f} mcc={mcc:.4f}")

        # 保存最优 encoder（只保存 encoder 的 state_dict）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.encoder.state_dict(), best_ckpt)
            print(f"[Saved best encoder] val_acc={val_acc:.4f} -> {best_ckpt}")

    print("Training finished. Best val acc:", best_val_acc)

    # 加载最优 encoder 权重到新的 encoder 实例（以确保只用 encoder 提取）
    trained_encoder = SulGraphEncoder().to(device)
    trained_encoder.load_state_dict(torch.load(best_ckpt, map_location=device))
    trained_encoder.eval()

    # 使用训练好的 encoder 分别抽取 left/right/full 的 atom embedding 并保存 npz
    get_embeddings_for_graphs_with_model(trained_encoder, left_cache, device, args.batch_size, os.path.join(args.save_dir, "left_atom.npz"))
    get_embeddings_for_graphs_with_model(trained_encoder, right_cache, device, args.batch_size, os.path.join(args.save_dir, "right_atom.npz"))
    get_embeddings_for_graphs_with_model(trained_encoder, full_cache, device, args.batch_size, os.path.join(args.save_dir, "full_atom.npz"))

    print("All done.")


if __name__ == "__main__":
    main()
