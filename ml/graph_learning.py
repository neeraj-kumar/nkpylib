"""Graph learning tools.

Implementation of Graph Attention Network (GAT) using PyTorch and PyTorch Geometric.

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

import time

from collections import Counter, defaultdict
from typing import Callable, Sequence, Any

import numpy as np
import numpy.random as npr
import psutil
import torch
import torch.nn.functional as F

from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC, SVC
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn.models import GAT
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tqdm import tqdm

from nkpylib.ml.feature_set import (
    array1d,
    array2d,
    nparray1d,
    nparray2d,
)

INVALID_NODE = -1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GATBase(torch.nn.Module):
    """An expanded version of the Graph ATtention Network (GAT) with various options.

    The GAT model was originally designed to be trained via a single node classification task.
    This implementation allows for training with many different kinds of objectives.

    The base class does node classification, as in the original paper.

    The original GAT paper: https://arxiv.org/abs/1710.10903
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 heads: int,
                 dropout: float = 0.6,
                 **kw):
        """Initialize this GAT model.

        - in_channels: Number of input node features
        - hidden_channels: Size of hidden layer embeddings
        - heads: Number of attention heads per layer

        The model has two GATConv layers, each followed by ELU and dropout (during training).

        Embeddings are the concatenation of the outputs of the two GAT layers, so have size
        `hidden_channels * heads * 2`.

        Note that this base class doesn't actually do any classification - it just outputs the
        final node embeddings. You can add a classification layer on top of this, or use the
        embeddings for other tasks.
        """
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        #self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False) # concat=False for final layer
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels * heads, heads=1)
        self.dropout = dropout

    def embedding_forward(self, x, edge_index):
        """Get raw embeddings from both layers before activation.

        Returns concatenated embeddings [e1, e2] suitable for embedding-based tasks.
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        e1 = self.conv1(x, edge_index)
        x = F.elu(e1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        e2 = self.conv2(x, edge_index)
        return torch.cat([e1, e2], dim=1)

    def forward(self, x, edge_index):
        """Regular forward pass with ELU activations for stable training."""
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return x

    def get_embeddings(self, x, edge_index):
        """Extract node embeddings from both GAT layers.

        Returns tensor of shape [num_nodes, hidden_channels * heads * 2]
        """
        self.eval()
        with torch.no_grad():
            return self.embedding_forward(x, edge_index)


class NodeClassificationGAT(GATBase):
    """GAT model for node classification tasks.

    """
    def __init__(self, out_channels: int, **kw):
        """
        Initialize this NodeClassificationGAT model.

        This adds a linear layer to the base model, which generates the final output.

        - out_channels: Size of output layer (e.g. number of classes)

        All other `kw` are passed to the base GAT model.
        """
        super().__init__(**kw)
        self.lin = torch.nn.Linear(self.conv1.out_channels, out_channels)

    def forward(self, x, edge_index):
        """Runs the base model and then the final linear layer."""
        x = super().forward(x, edge_index)
        x = self.lin(x)
        return x


class RandomWalkGAT(GATBase):
    """GAT model trained using random walk objectives.

    Instead of node classification, this learns node embeddings such that nodes
    appearing close together in random walks have similar embeddings.
    """
    def __init__(self,
                 walk_window: int = 5,
                 negative_samples: int = 10,
                 temperature: float = 0.07,
                 **kw):
        """initialize this RandomWalkGAT model.

        - walk_window: Context window size for walks (how many nodes before/after to consider)
        - negative_samples: Number of negative samples per positive pair
        - temperature: Temperature for similarity scaling (higher = softer attention)

        All other `kw` are passed to the base GAT model.
        """
        super().__init__(**kw)
        self.walk_window = walk_window
        self.negative_samples = negative_samples
        self.temperature = temperature

    def forward(self, x, edge_index, walks: Sequence[Sequence[int]]):
        """Forward pass using random walks for training.

        - x: Node features
        - edge_index: Graph connectivity
        - walks: List of random walks, each walk is list of node indices

        Returns the loss value computed from walk-based contrastive learning
        """
        # Get raw embeddings for contrastive loss
        embeddings = super().embedding_forward(x, edge_index)
        # But also run regular forward pass to maintain training stability
        _ = super().forward(x, edge_index)
        walks_tensor = torch.tensor(walks, device=x.device)
        valid_mask = walks_tensor != INVALID_NODE
        # get all anchors (all valid nodes in walks)
        anchors = walks_tensor[valid_mask]
        walk_length = walks_tensor.shape[1]
        all_pos_nodes = []
        all_anchor_idxs = []
        # generate positive pairs efficiently
        for i in range(walk_length):
            # get valid anchors at this position
            pos_mask = valid_mask[:, i]
            if not pos_mask.any():
                continue
            # get context window for these anchors
            start = max(0, i - self.walk_window)
            end = min(walk_length, i + self.walk_window + 1)
            # get all context nodes, excluding anchor position and invalid nodes
            context = walks_tensor[:, start:end]
            context_mask = valid_mask[:, start:end]
            context_mask[:, i-start] = False  # exclude anchor position
            # add valid context nodes for each anchor
            pos_walks = pos_mask.nonzero().squeeze(1)
            for walk_idx in pos_walks:
                valid_context = context[walk_idx][context_mask[walk_idx]]
                if len(valid_context) > 0:
                    all_pos_nodes.append(valid_context)
                    all_anchor_idxs.append(torch.full_like(valid_context, walks_tensor[walk_idx, i]))
        if not all_pos_nodes:  # No valid positive pairs found
            return torch.tensor(0.0, device=x.device)
        # concatenate all positive pairs
        pos_nodes = torch.cat(all_pos_nodes)
        anchors = torch.cat(all_anchor_idxs)
        # generate negative samples for all anchors at once
        neg_nodes = torch.randint(0, x.shape[0],
                                (len(anchors), self.negative_samples),
                                device=x.device)
        # compute similarities for all pairs at once
        anchor_embeds = embeddings[anchors]
        pos_embeds = embeddings[pos_nodes]
        neg_embeds = embeddings[neg_nodes.view(-1)].view(len(anchors), self.negative_samples, -1)
        # compute positive and negative similarities
        pos_sims = F.cosine_similarity(anchor_embeds, pos_embeds, dim=1) / self.temperature
        neg_sims = torch.bmm(
            anchor_embeds.unsqueeze(1),
            neg_embeds.transpose(1, 2)
        ).squeeze(1) / self.temperature
        # compute InfoNCE loss for all pairs at once
        pos_sum = torch.exp(pos_sims)
        neg_sum = torch.exp(neg_sims).sum(dim=1)
        loss = -torch.log(pos_sum / (pos_sum + neg_sum)).mean()
        return loss


class GraphLearner:
    """A class to do graph-based learning.

    Based on various input parameters, this creates and trains the appropriate kind of graph
    learning model. These can have different types of training objectives, some of which can be
    joined together:
    - Node classification: Define one or more classification tasks on nodes. The model will be
      trained to minimize cross-entropy loss on these tasks, with some weight on each task.
    - Random walk objectives: Generate random walks through the graph, and train the model to
      minimize a contrastive loss that brings together nodes appearing in the same walk. This is
      somewhat similar to node2vec/DeepWalk.
    """
    def __init__(self,
                 data,
                 hidden_channels:int=64,
                 heads:int=8,
                 dropout:float=0.6,
                 **kw):
        """Initialize this learning with the given `data` and parameters.

        The `data` should be a PyG data object containing the graph structure and node features.

        Learning parameters:
        - hidden_channels: Number of hidden embeddings per layer
        - heads: Number of attention heads per layer
        - dropout: Dropout rate during training

        The graph is a 2-layer graph convolutional net (specifically a GAT variant). The node
        embeddings are a concatenation of the outputs of the two layers, so have size
        `hidden_channels * heads * 2`.
        """
        self.data = data
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.kw = kw
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rng = npr.default_rng()

    def train_model(self,
                    model: torch.nn.Module,
                    loss_fn: Callable[[Any], torch.Tensor],
                    n_epochs:int=200) -> torch.Tensor:
        """Does actual model training in a loop.

        This does some bookkeeping around memory usage, etc.

        Args:
        - model: The graph model to train
        - loss_fn: A function that takes `(data, model)` and returns a loss tensor
        - n_epochs: Number of training epochs to run

        Returns the list of loss values per epoch.
        """
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        model = model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        pbar = tqdm(range(n_epochs), desc="Training Epochs")
        losses = []
        for epoch in pbar:
            optimizer.zero_grad()
            loss = loss_fn(self.data, model)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            # Get current memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = current_memory - initial_memory
            # update tqdm description with memory info
            pbar.set_description(
                f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
                f"Memory: {current_memory:.1f}MB (+{memory_diff:+.1f}MB)"
            )
        return torch.tensor(losses)

    def train_node_classification(self, n_epochs:int=200):
        model = NodeClassificationGAT(
                    in_channels=-1,#dataset.num_features,
                    hidden_channels=self.hidden_channels,
                    out_channels=dataset.num_classes,
                    heads=self.heads,
                    v2=False,
                    dropout=0.6,
                )

        def loss_fn(data, model):
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            return loss

        losses = self.train_model(model, loss_fn, n_epochs=n_epochs)
        return model

    def train_random_walks(self, walks: list[list[int]], n_epochs=20):
        """Train a graph model using random walk objectives.

        Pass in a list of `walks`, each of which is a list of node ids.
        """
        model = RandomWalkGAT(
            in_channels=self.data.num_features,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
        )
        def loss_fn(data, model):
            return model(data.x, data.edge_index, walks)

        losses = self.train_model(model, loss_fn, n_epochs=n_epochs)
        return model

    def gen_walks(self, n_walks_per_node:int=10, walk_length:int=12) -> nparray2d:
        """Generate random walks through the graph.

        The total number of walks is `n_walks_per_node * num_nodes`, each of length `walk_length`.

        Returns a list of walks, each of which is a list of node indices.
        """
        N = self.data.num_nodes
        # build adjacency matrix in CSR format for fast neighbor lookup
        edges = self.data.edge_index.cpu().numpy()
        adj = csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(N, N))
        # pre-allocate walks array and set starting nodes
        walks = np.zeros((n_walks_per_node * N, walk_length), dtype=np.int32)
        walks[:, 0] = np.tile(np.arange(N), n_walks_per_node)
        # generate walks in parallel
        for step in tqdm(range(1, walk_length), desc=f'Generating {walks.shape} walks'):
            # get neighbors for all current nodes
            current_nodes = walks[:, step-1]
            neighbors = [np.array(adj[node].indices) for node in current_nodes]
            has_neighbors = [len(n) > 0 for n in neighbors]
            # choose random neighbors (where possible)
            for i, (node_neighbors, valid) in enumerate(zip(neighbors, has_neighbors)):
                if valid: # randomly choose neighbors
                    walks[i, step] = self.rng.choice(node_neighbors)
                else: # no neighbors, add padding
                    walks[i, step] = INVALID_NODE
        return walks

    def test_gen_walks(self):
        """Tests out walk generation"""
        n, l = 20, 12
        t0 = time.time()
        walks = self.gen_walks(n_walks_per_node=n, walk_length=l)
        t1 = time.time()
        assert walks.shape == (self.data.num_nodes * n, l)
        edges = self.data.edge_index.cpu().numpy()
        print(f'Generated {walks.shape} walks in {t1-t0}s: {walks}, {edges}')
        counts = Counter()
        # check each walk
        for walk in walks:
            counts[walk[0]] += 1
            for i in range(1, len(walk)):
                if walk[i] == INVALID_NODE:
                    continue
                neighbors = edges[1][edges[0] == walk[i-1]]
                assert walk[i] in neighbors, f'Walk step {walk[i]} not a neighbor of {walk[i-1]}'
        # check that we have exactly n walks starting from each node
        for node in range(self.data.num_nodes):
            assert counts[node] == n, f'Node {node} has {counts[node]} walks, expected {n}'

    def train_and_eval_cls(self, model):
        """Train and evaluate a node classification model on our data.

        This uses the embeddings from the model and trains various classification models on them.
        """
        embs = model.get_embeddings(self.data.x, self.data.edge_index).cpu().numpy()
        y = self.data.y.cpu().numpy()
        train_mask = self.data.train_mask.cpu().numpy()
        test_mask = self.data.test_mask.cpu().numpy()
        X_train, y_train = embs[train_mask], y[train_mask]
        X_test, y_test = embs[test_mask], y[test_mask]
        print(f'Embeddings shape: {embs.shape}, train: {X_train.shape}, test: {X_test.shape}')
        models = dict(
            LinearSVC=LinearSVC(),
            SVC=SVC(kernel='rbf', gamma='scale'),
        )
        for name, model in models.items():
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f'Classifier {name} accuracy: {acc:.4f}')


def load_data(name: str='Cora'):
    dataset = Planetoid(root=f'/tmp/{name}', name=name, transform=NormalizeFeatures())
    data = dataset[0]
    data = data.to(device)
    return data, dataset

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


def quick_test(data):
    """Run a quick test to see how GAT works"""
    return
    gl = GraphLearner(data)
    model = gl.train_node_classification()
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


if __name__ == '__main__':
    data, dataset = load_data()
    print(f'Device {device}, Loaded data: {data}, num_classes: {dataset.num_classes}, {data.y.shape}, {data.y}')
    #quick_test(data)
    gl = GraphLearner(data)
    walks = gl.gen_walks(10, 10)
    model = gl.train_random_walks(walks)
    gl.train_and_eval_cls(model)
