"""Implementation of Graph Attention Network (GAT) using PyTorch and PyTorch Geometric."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False) # concat=False for final layer

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def load_data(name: str='PubMed'):
    dataset = Planetoid(root=f'/tmp/{name}', name=name, transform=NormalizeFeatures())
    data = dataset[0]
    return data, dataset

def train_model(data, n_epochs:int=200):
    model = GAT(in_channels=dataset.num_features,
                hidden_channels=8,
                out_channels=dataset.num_classes,
                heads=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    model.train()
    # make tqdm instance that we can feed into
    tqdm_epoch = tqdm(range(n_epochs), desc="Training Epochs")
    for epoch in tqdm_epoch:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # update tqdm description
        tqdm_epoch.set_description(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
    return model

def eval_model(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')


if __name__ == '__main__':
    data, dataset = load_data()
    print(f'Loaded data: {data}, num_classes: {dataset.num_classes}, num_features: {dataset.num_features}, num_nodes: {data.num_nodes}')
    model = train_model(data)
    print(f'Trained model')
    eval_model(model, data)
