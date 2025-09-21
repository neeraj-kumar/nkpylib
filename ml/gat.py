"""Implementation of Graph Attention Network (GAT) using PyTorch and PyTorch Geometric.

TODO:

Yes, the current GAT implementation is designed around a node classification task, as evidenced by:

1. The output layer (conv2) produces out_channels=dataset.num_classes
2. Training uses cross-entropy loss against node labels (data.y)
3. Evaluation measures classification accuracy

There are several alternatives for training without/with different tasks:

1. Self-supervised approaches:

 • Link prediction
 • Node reconstruction
 • Random walk objectives
 • Contrastive learning between node pairs

2. Multi-task training:

 • Multiple classification heads
 • Mix of classification and regression tasks
 • Auxiliary tasks like structure prediction

3. Unsupervised approaches:

 • Graph autoencoders
 • Structure-preserving objectives
 • Neighborhood reconstruction

Would you like me to modify the GAT implementation to support one of these alternative training approaches?


Tokens: 31k sent, 167 received. Cost: $0.09 message, $1.53 session.
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ml/evaluator.py ml/gat.py
> /ask describe each of these

Here's a breakdown of each alternative GAT training approach:

Self-supervised approaches:

1. Link Prediction

 • Mask some edges during training
 • Train model to predict if edges exist between node pairs
 • Uses node embeddings to compute edge likelihood
 • Natural task for graphs, preserves structure

2. Node Reconstruction

 • Corrupt node features randomly
 • Train model to reconstruct original features
 • Forces embeddings to preserve node information
 • Similar to denoising autoencoders

3. Random Walk Objectives

 • Generate random walks through graph
 • Train model to predict nodes that appear together in walks
 • Similar to word2vec but for graphs
 • Captures both local and broader graph structure

4. Contrastive Learning

 • Create positive pairs (connected/similar nodes)
 • Create negative pairs (distant/dissimilar nodes)
 • Train embeddings to be close for positives, far for negatives
 • Very flexible in how pairs are chosen

Multi-task approaches:

1. Multiple Classification Heads

 • Different classifiers for different node labels
 • Share embedding layers
 • Each task provides different supervision signal
 • Can weight tasks differently

2. Mixed Classification/Regression

 • Some heads predict categories
 • Others predict continuous values
 • Richer supervision signal
 • More complex loss function

3. Structure Prediction

 • Main task plus structural auxiliary tasks
 • e.g., predict node degree, clustering coefficient
 • Helps embeddings capture graph properties
 • Can improve main task performance

Unsupervised approaches:

1. Graph Autoencoders

 • Encode nodes into embeddings
 • Decode to reconstruct graph structure
 • No labels needed
 • Can preserve both features and structure

2. Structure-preserving

 • Optimize embeddings to preserve graph distances
 • Similar to traditional dimensionality reduction
 • Can use various graph metrics
 • Pure structural approach

3. Neighborhood Reconstruction

 • Encode node's features
 • Decode to predict neighbors' features
 • Preserves local graph structure
 • Natural for attention mechanisms


Let me explain how each parameter would likely affect the GAT model's accuracy:

1. n_epochs (currently 200):
 • Too few epochs: Model won't converge, leading to poor accuracy
 • Too many epochs: Risk of overfitting, especially on smaller datasets
 • Optimal range typically 100-500 depending on dataset size and complexity
 • Best practice: Use early stopping based on validation accuracy

2. hidden_channels (currently 8):
 • Larger values (e.g., 16, 32, 64) allow capturing more complex patterns
 • But also increase model parameters and risk of overfitting
 • Too small values limit model capacity
 • Rule of thumb: Start with sqrt(input_dim * output_dim)

3. heads (currently 8):
 • More attention heads allow capturing different types of node relationships
 • Common values range from 4-16
 • Increasing heads increases parameters linearly
 • Diminishing returns after certain point (dataset dependent)

4. Number of GATConv layers (currently 2):
 • Adding layers allows capturing higher-order node relationships
 • But faces vanishing gradient problems beyond 2-3 layers
 • Each additional layer adds risk of over-smoothing
 • Common architectures use 2-3 layers
 • Could add skip/residual connections to help with deeper architectures

"""

from __future__ import annotations

import psutil
import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn.models import GAT
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, **kw):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        #self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False) # concat=False for final layer
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels * heads, heads=1)
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x

    def get_embeddings(self, x, edge_index):
        """Extract node embeddings from the first GAT layer.

        Returns tensor of shape [num_nodes, hidden_channels * heads]
        """
        self.eval()
        with torch.no_grad():
            emb1 = F.elu(self.conv1(x, edge_index))
            emb2 = self.conv2(emb1, edge_index)
            # concat these to get embeddings
            embeddings = torch.cat([emb1, emb2], dim=1)
            print(f'embs:', emb1.shape, emb2.shape, embeddings.shape)
            return embeddings

def get_embeddings(model, x, edge_index):
    """Returns embeddings from a gat model"""
    model.eval()
    with torch.no_grad():
        emb1 = F.elu(self.conv1(x, edge_index))
        emb2 = self.conv2(emb1, edge_index)
        return embeddings

def load_data(name: str='PubMed'):
    dataset = Planetoid(root=f'/tmp/{name}', name=name, transform=NormalizeFeatures())
    data = dataset[0]
    data = data.to(device)
    return data, dataset

def train_model(data, n_epochs:int=200, hidden_channels:int=8, heads:int=8):
    model = MyGAT(in_channels=-1,#dataset.num_features,
                hidden_channels=hidden_channels,
                out_channels=dataset.num_classes,
                heads=heads,
                num_layers=2,
                v2=False,
                dropout=0.6,
                )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    model.train()
    # make tqdm instance that we can feed into
    tqdm_epoch = tqdm(range(n_epochs), desc="Training Epochs")
    for epoch in tqdm_epoch:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # Get current memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = current_memory - initial_memory
        # update tqdm description with memory info
        tqdm_epoch.set_description(
            f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
            f"Memory: {current_memory:.1f}MB (+{memory_diff:+.1f}MB)"
        )
    return model

def eval_model(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')

# Test feature vs connectivity importance
def test_feature_importance(model, data):
    """Test how embeddings change with different node features."""
    # Get original embeddings and accuracy
    orig_embeddings = model.get_embeddings(data.x, data.edge_index)
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    orig_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

    # Test with randomized features
    random_x = torch.randn_like(data.x)
    random_embeddings = model.get_embeddings(random_x, data.edge_index)
    pred = model(random_x, data.edge_index).argmax(dim=1)
    random_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

    # Test with zeroed features
    zero_x = torch.zeros_like(data.x)
    zero_embeddings = model.get_embeddings(zero_x, data.edge_index)
    pred = model(zero_x, data.edge_index).argmax(dim=1)
    zero_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    return (orig_embeddings, orig_acc), (random_embeddings, random_acc), (zero_embeddings, zero_acc)

def test_connectivity_importance(model, data):
    """Test how embeddings change with different graph connectivity."""
    # Get original embeddings and accuracy
    orig_embeddings = model.get_embeddings(data.x, data.edge_index)
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    orig_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

    # Test with random edges (same number of edges)
    n_edges = data.edge_index.shape[1]
    random_edges = torch.randint(0, data.num_nodes, (2, n_edges))
    random_edge_embeddings = model.get_embeddings(data.x, random_edges)
    pred = model(data.x, random_edges).argmax(dim=1)
    random_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

    # Test with no edges
    no_edges = torch.zeros((2, 0), dtype=torch.long)
    isolated_embeddings = model.get_embeddings(data.x, no_edges)
    pred = model(data.x, no_edges).argmax(dim=1)
    no_edge_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    return (orig_embeddings, orig_acc), (random_edge_embeddings, random_acc), (isolated_embeddings, no_edge_acc)

def compare_embeddings(emb1, emb2):
    """Compare two sets of embeddings using cosine similarity."""
    # Compute cosine similarity between corresponding nodes
    sim = F.cosine_similarity(emb1, emb2)
    return {
        'mean_sim': sim.mean().item(),
        'std_sim': sim.std().item(),
        'min_sim': sim.min().item(),
        'max_sim': sim.max().item()
    }


if __name__ == '__main__':
    data, dataset = load_data()
    print(f'Device {device}, Loaded data: {data}, num_classes: {dataset.num_classes}, {data.y.shape}, {data.y}, {data.train_mask.shape}, {data.train_mask}')
    model = train_model(data)
    print(f'Trained model')
    eval_model(model, data)

    # Get and print node embeddings
    embeddings = model.get_embeddings(data.x, data.edge_index)
    print(f'\nNode embeddings shape: {embeddings.shape}')
    print(f'First node embedding:\n{embeddings[0]}')
    print(f'Embedding stats: mean={embeddings.mean():.3f}, std={embeddings.std():.3f}')

    # Run importance tests
    print("\nTesting feature importance:")
    (orig, orig_acc), (random, random_acc), (zero, zero_acc) = test_feature_importance(model, data)
    print(f"Original accuracy: {orig_acc:.4f}")
    print("Random features:", compare_embeddings(orig, random), f"accuracy: {random_acc:.4f}")
    print("Zero features:", compare_embeddings(orig, zero), f"accuracy: {zero_acc:.4f}")

    print("\nTesting connectivity importance:")
    (orig, orig_acc), (random_edge, random_acc), (no_edge, no_edge_acc) = test_connectivity_importance(model, data)
    print(f"Original accuracy: {orig_acc:.4f}")
    print("Random edges:", compare_embeddings(orig, random_edge), f"accuracy: {random_acc:.4f}")
    print("No edges:", compare_embeddings(orig, no_edge), f"accuracy: {no_edge_acc:.4f}")
