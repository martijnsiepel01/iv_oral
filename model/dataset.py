from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import torch

class IVSwitchGraphDataset(Dataset):
    def __init__(self, graph_label_tuples: list[tuple[Data, torch.Tensor]]):
        self.samples = graph_label_tuples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    graphs, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.tensor(labels, dtype=torch.long)