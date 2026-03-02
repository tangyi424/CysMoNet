import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch_geometric.nn.dense.linear as pyg_linear
from torch_geometric.nn.norm import BatchNorm


# ===========================================================
#      —— 保持完全一致 —— 你的底层 EdgeGCNLayer + BottomGCN
# ===========================================================
class EdgeGCNLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, aggr='add'):
        super().__init__(aggr=aggr)
        self.msg_fc = pyg_linear.Linear(hidden_dim + edge_dim, hidden_dim, weight_initializer='kaiming_uniform')
        self.up_fc = pyg_linear.Linear(2 * hidden_dim, hidden_dim, weight_initializer='kaiming_uniform')
        self.bn = BatchNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.bn(F.relu(out))

    def message(self, x_j, edge_attr):
        msg = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.msg_fc(msg)
        return F.leaky_relu(msg, 0.1)

    def update(self, aggr_out, x):
        out = self.up_fc(torch.cat([aggr_out, x], dim=-1))
        return F.leaky_relu(out, 0.1)


class BottomGCN(nn.Module):
    """
    底层原子图编码器
    """
    def __init__(self, in_dim=25, edge_dim=11, hidden_dim=128, depth=4, dropout=0.3):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.input_proj = pyg_linear.Linear(in_dim, hidden_dim, weight_initializer='kaiming_uniform')
        self.layers = nn.ModuleList([EdgeGCNLayer(hidden_dim, edge_dim) for _ in range(depth)])
        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(depth)])

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.leaky_relu(self.input_proj(x), 0.1)
        for i in range(self.depth):
            x = self.layers[i](x, edge_index, edge_attr)
            # x = self.bns[i](x)  # 保持原注释
            x = self.dropout(x)
        entity_emb = global_mean_pool(x, batch)
        return entity_emb


# ===========================================================
#             —— 最重要 —— SulGraphEncoder（供 embedding 提取）
# ===========================================================
# class SulGraphEncoder(nn.Module):
#     """
#     专门用于 get_embeddings_for_graphs() 的封装：
#     输入 batch_graph（pyg Data），输出每个子图的 embedding
#     """
#     def __init__(
#         self,
#         atom_in_dim=25,
#         atom_edge_dim=11,
#         hidden_dim=128,
#         depth=4,
#         dropout=0.3
#     ):
#         super().__init__()
#
#         # 这里只需要 BottomGCN（你用于单图 embedding）
#         self.encoder = BottomGCN(
#             in_dim=atom_in_dim,
#             edge_dim=atom_edge_dim,
#             hidden_dim=hidden_dim,
#             depth=depth,
#             dropout=dropout
#         )
#
#     def forward(self, batch_graph):
#         """
#         输入:
#             batch_graph.x           [N, in_dim]
#             batch_graph.edge_index  [2, E]
#             batch_graph.edge_attr   [E, edge_dim]
#             batch_graph.batch       [N]  — 每个节点属于哪个子图
#         输出:
#             graph_emb [B, hidden_dim] — 每个子图的 embedding
#         """
#         graph_emb = self.encoder(
#             batch_graph.x,
#             batch_graph.edge_index,
#             batch_graph.edge_attr,
#             batch_graph.batch
#         )
#         return graph_emb
class SulGraphEncoder(nn.Module):
    """
    输入 batch_graph（pyg Data），输出：
      - 默认：图级 embedding
      - return_node=True 时：节点级 embedding（不池化）
    """
    def __init__(
        self,
        atom_in_dim=25,
        atom_edge_dim=11,
        hidden_dim=128,
        depth=4,
        dropout=0.3
    ):
        super().__init__()

        self.bottom_gcn = BottomGCN(
            in_dim=atom_in_dim,
            edge_dim=atom_edge_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout
        )

    def forward(self, batch_graph, return_node=False):
        x = F.leaky_relu(self.bottom_gcn.input_proj(batch_graph.x), 0.1)
        for layer in self.bottom_gcn.layers:
            x = layer(x, batch_graph.edge_index, batch_graph.edge_attr)
            x = self.bottom_gcn.dropout(x)

        if return_node:
            # 返回每个节点 embedding（不做 pooling）
            return x
        else:
            # 返回每个子图 embedding（默认模式）
            return global_mean_pool(x, batch_graph.batch)
