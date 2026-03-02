# ---------------- sul_atom_explain.py ----------------
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from sul_model import SulGraphEncoder
from Bio.PDB import PDBParser

# ============================================================
# 1. 定义模型
# ============================================================
class SulPredictor(nn.Module):
    def __init__(self, encoder_path, out_dim=1):
        super().__init__()
        self.encoder = SulGraphEncoder()
        self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        self.fc = nn.Linear(128, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        node_emb = self.encoder(data, return_node=True)
        graph_emb = node_emb.mean(dim=0, keepdim=True)
        out = self.fc(graph_emb)
        out = self.sigmoid(out)
        return out, node_emb


# ============================================================
# 2. Grad-CAM 原子级解释
# ============================================================
def atom_level_gradcam(model, data):
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    data.x.requires_grad_(True)

    pred, node_emb = model(data)
    score = pred.squeeze()
    score.backward()

    # 原始 Grad-CAM 重要性
    importance = (data.x.grad * data.x).sum(dim=1).abs()

    # 数值标准化（min-max normalization）
    imp_min = importance.min()
    imp_max = importance.max()
    if imp_max - imp_min > 1e-8:
        importance = (importance - imp_min) / (imp_max - imp_min)
    else:
        importance = torch.zeros_like(importance)

    # 限制范围 [0, 1]（确保安全）
    importance = torch.clamp(importance, 0.0, 1.0)

    return importance.detach().cpu().numpy()


# ============================================================
# 3. 从PDB获取坐标
# ============================================================
def get_coords_from_pdb(pdb_path, chain_id=None, limit=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coords = []
    for model in structure:
        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            for residue in chain:
                for atom in residue:
                    coords.append(atom.coord.tolist())

    if limit:
        coords = coords[:limit]
    print(f"[INFO] 从PDB加载 {len(coords)} 个原子坐标")
    return np.array(coords, dtype=np.float32)


# ============================================================
# 4. 在PDB中标注三段序列颜色
# ============================================================
def find_residue_indices(pdb_path, segments):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    chain = next(structure.get_chains())

    # 三字母到一字母映射表
    AA_THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
        "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
        "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
        "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y"
    }

    residues = [res for res in chain.get_residues() if res.id[0] == " "]

    seq = ""
    residue_ids = []
    for res in residues:
        resname = res.resname.strip()
        if resname in AA_THREE_TO_ONE:
            seq += AA_THREE_TO_ONE[resname]
            residue_ids.append(res.id[1])
        else:
            # 跳过无法识别的残基
            continue

    print(f"[INFO] 提取到链 {chain.id} 序列长度: {len(seq)}")

    found = []
    for seg in segments:
        idx = seq.find(seg)
        if idx == -1:
            print(f"[WARN] 未找到序列段: {seg}")
            continue
        start = idx + 1
        end = start + len(seg) - 1
        found.append((start, end))
        print(f"[INFO] 找到序列 {seg[:6]}... → 残基 {start}-{end}")

    return found


# ============================================================
# 5. 保存带颜色和重要性信息的PDB
# ============================================================
def save_colored_pdb(data, importance, pdb_path, out_path, regions):
    with open(out_path, "w") as f:
        for i, (x, y, z) in enumerate(data.pos.tolist()):
            b = float(importance[i])
            # 默认B-factor为Grad-CAM重要性
            f.write(f"ATOM  {i+1:5d}  C   MOL A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 {b:6.2f}\n")
        f.write("END\n")
    print(f"[OK] 写入基础重要性PDB → {out_path}")

    # 三段颜色（红、绿、蓝）
    color_map = {0: 1.0, 1: 0.5, 2: 0.0}

    # 根据残基编号重写B因子颜色
    new_lines = []
    with open(out_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith("ATOM"):
                new_lines.append(line)
                continue
            resid = int(line[22:26].strip())
            color_value = None
            for i, (start, end) in enumerate(regions):
                if start <= resid <= end:
                    color_value = color_map[i]
                    break
            if color_value is not None:
                line = line[:60] + f"{color_value:6.2f}" + line[66:]
            new_lines.append(line)

    colored_path = out_path.replace(".pdb", "_colored.pdb")
    with open(colored_path, "w") as f:
        f.writelines(new_lines)
    print(f"[OK] 三段颜色版PDB写入 → {colored_path}")
    return colored_path


# ============================================================
# 6. 可视化重要性（直方图）
# ============================================================
def plot_atom_importance(importance, save_path="atom_heatmap.png"):
    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(importance)), importance, color='tomato')
    plt.xlabel("Atom index")
    plt.ylabel("Grad-CAM Importance")
    plt.title("Atom-level importance (GraphCL)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Heatmap saved → {save_path}")


# ============================================================
# 7. 主程序
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, default="../data/graph_cache_sul/full_graphs.pt")
    parser.add_argument("--encoder_path", type=str, default="encoder_graphcl.pt")
    parser.add_argument("--pdb_path", type=str, default="AF-A0A075B759-F1-model_v6.pdb")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="./explain_out")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--chain", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 三段序列
    SEQ_SEGMENTS = [
        "RALSTGEKGFRYKGSCFHRIIPGFMCQGGDF",
        "RYKGSCFHRIIPGFMCQGGDFTRPNGTGDKS",
        "MANAGPNTNGSQFFICAAKTEWLDGKHVAFG"
    ]

    # 载入图数据
    print(f"[INFO] Loading graphs from {args.graph_path}")
    graph_cache = torch.load(args.graph_path)
    keys = list(graph_cache.keys())
    key = keys[args.index]
    graph = graph_cache[key]
    print(f"[INFO] Loaded molecule: {key}  (#atoms={graph.num_nodes})")

    # 从PDB加载坐标
    coords = get_coords_from_pdb(args.pdb_path, chain_id=args.chain, limit=graph.num_nodes)
    graph.pos = torch.tensor(coords, dtype=torch.float32)

    # 模型与Grad-CAM
    model = SulPredictor(args.encoder_path).to(device)
    importance = atom_level_gradcam(model, graph)

    # 生成文件
    pdb_out = os.path.join(args.out_dir, f"{key}_explain.pdb")
    png_out = os.path.join(args.out_dir, f"{key}_heatmap.png")

    regions = find_residue_indices(args.pdb_path, SEQ_SEGMENTS)
    colored_pdb = save_colored_pdb(graph, importance, args.pdb_path, pdb_out, regions)
    plot_atom_importance(importance, png_out)

    print("\n✅ 完成解释生成！")
    print(f"输出文件：\n  {colored_pdb}\n  {png_out}\n")
    print("在 VMD 中运行以下命令进行可视化：\n"
          f"  mol new {colored_pdb}\n"
          "  mol modstyle 0 0 CPK 1.0 0.3 12.0 12.0\n"
          "  mol modcolor 0 0 Beta\n"
          "  color scale method BGR\n"
          "  mol scaleminmax 0 0 0.0 1.0\n"
          "  display projection Orthographic\n"
          "  display depthcue off\n"
          "  axes location off\n")


if __name__ == "__main__":
    main()
