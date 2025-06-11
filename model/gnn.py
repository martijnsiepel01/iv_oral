import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool  

class GNNModel(nn.Module):
    def __init__(self, in_dim, node_type_vocab_size, model_type="GAT", emb_dim=32, 
                 num_layers=1, pooling="max", dropout=0.0, num_classes=4):
        super().__init__()
        self.model_type = model_type
        self.node_type_emb = nn.Embedding(node_type_vocab_size, emb_dim)
        self.proj = nn.Linear(in_dim + emb_dim, emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Create GNN layers
        if model_type == "GAT":
            self.gnn_layers = nn.ModuleList([
                GATConv(emb_dim if i == 0 else emb_dim, emb_dim, heads=2, concat=False)
                for i in range(num_layers)
            ])
        elif model_type == "GCN":
            self.gnn_layers = nn.ModuleList([
                GCNConv(emb_dim if i == 0 else emb_dim, emb_dim)
                for i in range(num_layers)
            ])
        elif model_type == "GIN":
            self.gnn_layers = nn.ModuleList([
                GINConv(nn.Sequential(
                    nn.Linear(emb_dim if i == 0 else emb_dim, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim)
                ))
                for i in range(num_layers)
            ])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Pooling configuration
        self.pooling = pooling
        if pooling == "concat":
            self.pool_proj = nn.Linear(emb_dim * 2, emb_dim)
        
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, data):
        x = data.x
        nt = self.node_type_emb(data.node_type_ids)
        
        if torch.any(data.node_type_ids >= self.node_type_emb.num_embeddings):
            print("[FATAL] node_type_ids exceed embedding size!")
        
        if torch.all(nt == 0):
            print("[WARN] All node type embeddings are zero")

        if torch.isnan(x).any():
            print("[ERROR] NaNs in input x")

        if torch.isnan(nt).any():
            print("[ERROR] NaNs in node type embeddings")
            
        x = self.proj(torch.cat([x, nt], dim=1))
        x = self.dropout(x)

        if torch.isnan(x).any():
            print("[ERROR] NaNs after projection")

        # Apply GNN layers
        for gnn in self.gnn_layers:
            x = torch.relu(gnn(x, data.edge_index))
            x = self.dropout(x)

        if torch.isnan(x).any():
            print("[ERROR] NaNs after GNN")

        # Apply pooling
        if self.pooling == "max":
            graph_repr = global_max_pool(x, data.batch)
        elif self.pooling == "mean":
            graph_repr = global_mean_pool(x, data.batch)
        elif self.pooling == "concat":
            max_pool = global_max_pool(x, data.batch)
            mean_pool = global_mean_pool(x, data.batch)
            graph_repr = self.pool_proj(torch.cat([max_pool, mean_pool], dim=1))
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        if torch.isnan(graph_repr).any():
            print("[ERROR] NaNs in graph_repr before classification")

        out = self.cls(graph_repr)
        
        return out