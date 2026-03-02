import json, os, torch
from tqdm import tqdm
from featurizer import MolGraphConvFeaturizer
from rdkit import Chem
from torch_geometric import data as DATA

def load_sul_txt(path):
    with open(path) as f:
        data = json.load(f)
    # 返回 [(id, seq)]
    return list(data.items())

def split_seq(seq):
    seq = seq.strip()
    left = seq[0:20]
    right = seq[21:41]
    return left, right, seq  # left, right, full

class SulGraphBuilder:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def build_and_merge(self, pos_items, neg_items):
        """
        直接构建所有 pos 和 neg 的 left / right / full 图，
        并一次性合并，最终只保存 3 个 pt 文件。
        """

        merged_left = {}
        merged_right = {}
        merged_full = {}

        def generateGraph(seq):
            seq = seq.replace('X', 'G')
            featurizer = MolGraphConvFeaturizer(use_edges=True)
            seq_chem = Chem.MolFromSequence(seq)
            seq_feature = featurizer._featurize(seq_chem)
            feature, edge_index, edge_feature = seq_feature.node_features, seq_feature.edge_index, seq_feature.edge_features
            graph = DATA.Data(x=torch.Tensor(feature), edge_index=torch.LongTensor(edge_index),
                              edge_attr=torch.Tensor(edge_feature))
            return graph

        # ------------------------------------------------------
        # Helper：处理 pos + neg
        # ------------------------------------------------------
        def process_items(items, prefix):
            for sid, seq in tqdm(items, desc=f"Building {prefix}"):
                left, right, full = split_seq(seq)
                try:
                    if left not in merged_left:
                        merged_left[left] = generateGraph(left)
                    if right not in merged_right:
                        merged_right[right] = generateGraph(right)
                    if full not in merged_full:
                        merged_full[full] = generateGraph(full)
                except Exception as e:
                    print("Graph failed:", sid, seq, e)
                    continue

        # ------------------------------------------------------
        # 先构建 POS
        # ------------------------------------------------------
        print("Building POS graphs...")
        process_items(pos_items, "POS")

        # ------------------------------------------------------
        # 再构建 NEG
        # ------------------------------------------------------
        print("Building NEG graphs...")
        process_items(neg_items, "NEG")

        # ------------------------------------------------------
        # 最后一次性保存 3 个合并后的文件
        # ------------------------------------------------------
        torch.save(merged_left,  os.path.join(self.cache_dir, "left_graphs.pt"))
        torch.save(merged_right, os.path.join(self.cache_dir, "right_graphs.pt"))
        torch.save(merged_full,  os.path.join(self.cache_dir, "full_graphs.pt"))

        print("[DONE] merged results saved:")
        print(" left =", len(merged_left))
        print(" right =", len(merged_right))
        print(" full =", len(merged_full))



builder = SulGraphBuilder('../data_2/graph_cache_sul')
pos_items = load_sul_txt('../data_2/pos_2.txt')
neg_items = load_sul_txt('../data_2/neg_2.txt')
builder.build_and_merge(pos_items, neg_items)