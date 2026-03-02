# ---------------- sul_dataset.py ----------------
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

class SulDataset(Dataset):
    """
    Dataset 仅包含所有序列（key）
    """
    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.keys[idx]


def collate_graph(batch_keys, graph_cache):
    """
    batch 中是多个序列 key，
    graph_cache 是 {key: Data}
    返回：
        - Batch 图
        - map_local: batch 中每条样本对应的 local_idx
    """
    data_list = []
    map_local = []

    for key in batch_keys:
        data = graph_cache[key]
        map_local.append(len(data_list))
        data_list.append(data)

    bottom_batch = Batch.from_data_list(data_list)
    map_local = torch.tensor(map_local, dtype=torch.long)

    return bottom_batch, map_local
