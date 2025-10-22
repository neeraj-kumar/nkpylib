"""Graph learning tools.

Implementation of Graph Attention Network (GAT) using PyTorch and PyTorch Geometric.

TODO:
- Develop multi-task classification heads
- Fix memory issues
- contrastive loss is good for link prediction?
  - maybe instead of concatenating node vectors, could do elementwise product/diff/etc?


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

import logging
import time

from argparse import ArgumentParser
from collections import Counter, defaultdict
from typing import Callable, Sequence, Any

import numpy as np
import numpy.random as npr
import psutil
import torch
import torch.nn.functional as F

from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn.models import GAT
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from tqdm import tqdm

from nkpylib.ml.feature_set import (
    array1d,
    array2d,
    nparray1d,
    nparray2d,
    FeatureSet,
    NumpyLmdb,
)

torch.manual_seed(0)

logger = logging.getLogger(__name__)

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
        self.process = psutil.Process()

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
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        """Extract node embeddings from both GAT layers.

        Returns tensor of shape [num_nodes, hidden_channels * heads * 2]
        """
        self.eval()
        with torch.no_grad():
            return self.embedding_forward(x, edge_index)

    def log_memory(self, msg):
        mem = self.process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
        #print(f"{msg}: {mem:.2f}GB")


class NodeClassificationGAT(GATBase):
    """GAT model for node classification tasks.

    """
    def __init__(self, hidden_channels: int, heads: int, out_channels: int, **kw):
        """
        Initialize this NodeClassificationGAT model.

        This adds a linear layer to the base model, which generates the final output.

        - out_channels: Size of output layer (e.g. number of classes)

        All other `kw` are passed to the base GAT model.
        """
        super().__init__(hidden_channels=hidden_channels, heads=heads, **kw)
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index):
        """Runs the base model and then the final linear layer."""
        x = super().forward(x, edge_index)
        x = self.lin(F.elu(x))
        return x


class ContrastiveGAT(GATBase):
    """GAT model trained using contrastive learning.

    This class implements the core contrastive learning logic, processing pairs
    of positive and negative examples to learn node embeddings.
    """
    def __init__(self,
                 negative_samples: int = 10,
                 temperature: float = 0.07,
                 **kw):
        """Initialize this ContrastiveGAT model.

        Args:
            negative_samples: Number of negative samples per positive pair
            temperature: Temperature for similarity scaling (higher = softer attention)
        """
        super().__init__(**kw)
        self.negative_samples = negative_samples
        self.temperature = temperature

    def batch_loss(self,
                      embeddings: torch.Tensor,
                      anchors: torch.Tensor,
                      pos_nodes: torch.Tensor,
                      neg_nodes: torch.Tensor,
                      cur_batch_size: int) -> torch.Tensor:
        """Process a batch of nodes and compute the loss.

        Args:
            embeddings: Node embeddings tensor
            anchors: Anchor node indices
            pos_nodes: Positive sample node indices
            neg_nodes: Negative sample node indices
            cur_batch_size: Size of current batch

        Returns:
            Batch loss value
        """
        # Get embeddings for this batch
        anchor_embeds = embeddings[anchors]
        pos_embeds = embeddings[pos_nodes]
        neg_embeds = embeddings[neg_nodes.view(-1)].view(
            cur_batch_size, self.negative_samples, -1
        )

        # Compute similarities
        cos = torch.nn.CosineSimilarity(dim=1)
        pos_sims = cos(anchor_embeds, pos_embeds) / self.temperature

        anchor_embeds_reshaped = anchor_embeds.unsqueeze(1)
        neg_embeds_reshaped = neg_embeds.transpose(1, 2)
        neg_sims = torch.bmm(anchor_embeds_reshaped, neg_embeds_reshaped).squeeze(1) / self.temperature

        # Compute loss
        all_sims = torch.cat([pos_sims.unsqueeze(1), neg_sims], dim=1)
        targets = torch.zeros(cur_batch_size, dtype=torch.long, device=embeddings.device)
        batch_loss = F.cross_entropy(all_sims, targets)

        return batch_loss

    def pair_generator(self, batch_size: int) -> iterable[tuple[array1d, array1d]]:
        """Generates positive pairs of nodes for contrastive learning.

        This should yield tuples of `(anchors, positive_nodes)`
        """
        raise NotImplementedError("Subclasses must implement pair_generator()")

    def compute_loss(self, x, edge_index, batch_size: int = 1024):
        """Compute contrastive loss using pairs of nodes.

        Args:
            x: Node features
            edge_index: Graph connectivity
            batch_size: Number of pairs to process at once

        Returns:
            Average loss across all pairs
        """
        # Get embeddings
        embeddings = self.embedding_forward(x, edge_index).cpu()

        total_loss = 0
        total_pairs = 0

        for anchors, pos_nodes in self.pair_generator(batch_size=batch_size):
            cur_batch_size = len(anchors)

            # Generate negative samples
            with torch.no_grad():
                neg_nodes = torch.randint(0, x.shape[0], (cur_batch_size, self.negative_samples))

            # Process batch
            batch_loss = self.batch_loss(
                embeddings=embeddings,
                anchors=anchors,
                pos_nodes=pos_nodes,
                neg_nodes=neg_nodes,
                cur_batch_size=cur_batch_size,
            )

            total_loss += batch_loss
            total_pairs += cur_batch_size

        if total_pairs == 0:
            raise ValueError("No valid pairs found!")

        return total_loss / total_pairs


class RandomWalkGAT(ContrastiveGAT):
    """GAT model trained using random walk objectives.

    Instead of node classification, this learns node embeddings such that nodes
    appearing close together in random walks have similar embeddings.
    """
    def __init__(self, walks: Sequence[Sequence[int]], walk_window: int = 5, **kw):
        """Initialize this RandomWalkGAT model.

        Args:
            walk_window: Context window size for walks (how many nodes before/after to consider)
        """
        super().__init__(**kw)
        self.walk_window = walk_window
        self.walks = walks

    def pair_generator(self, batch_size: int) -> iterable[tuple[array1d, array1d]]:
        """Generate positive pairs from random walks."""
        walks_tensor = torch.tensor(self.walks)
        valid_mask = walks_tensor != INVALID_NODE
        walk_length = walks_tensor.shape[1]

        for i in range(walk_length):
            pos_mask = valid_mask[:, i].clone()
            if not pos_mask.any():
                continue

            # Get valid walks for this position
            pos_walks = pos_mask.nonzero().squeeze(1)
            if len(pos_walks.shape) == 0:
                pos_walks = pos_walks.unsqueeze(0)
            n_pos = len(pos_walks)

            # Process walks in batches
            for batch_start in range(0, n_pos, batch_size):
                batch_end = min(batch_start + batch_size, n_pos)
                batch_walks = pos_walks[batch_start:batch_end]

                # Get context window
                start = max(0, i - self.walk_window)
                end = min(walk_length, i + self.walk_window + 1)
                context = walks_tensor[batch_walks][:, start:end]
                context_mask = valid_mask[batch_walks][:, start:end].clone()
                context_mask[:, i-start] = False

                # Collect positive pairs
                batch_pos_nodes = []
                batch_anchor_idxs = []
                for idx, walk_idx in enumerate(batch_walks):
                    valid_context = context[idx][context_mask[idx]]
                    if len(valid_context) > 0:
                        batch_pos_nodes.append(valid_context)
                        batch_anchor_idxs.append(
                            torch.full_like(valid_context, walks_tensor[walk_idx, i])
                        )

                if batch_pos_nodes:
                    yield (
                        torch.cat(batch_anchor_idxs),
                        torch.cat(batch_pos_nodes)
                    )


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
        self.rng = npr.default_rng(0)

    def train_model(self,
                    model: torch.nn.Module,
                    loss_fn: Callable[[Any], torch.Tensor],
                    n_epochs:int=200) -> torch.Tensor:
        """Does actual model training in a loop.

        This does some bookkeeping around memory usage, etc.

        Args:
        - model: The graph model to train
        - loss_fn: A function that takes `(model)` and returns a loss tensor
        - n_epochs: Number of training epochs to run

        Returns the list of loss values per epoch.
        """
        process = psutil.Process()
        memory = Counter()
        memory['initial'] = process.memory_info().rss / 1024 / 1024  # MB
        model = model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        pbar = tqdm(range(n_epochs), desc="Training Epochs")
        losses = []
        for epoch in pbar:
            optimizer.zero_grad()
            loss = loss_fn(model)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            # Get current memory usage
            memory['current'] = process.memory_info().rss / 1024 / 1024  # MB
            memory['diff'] = memory['current'] - memory['initial']
            memory['peak'] = max(memory['peak'], memory['current'])
            memory['peak diff'] = memory['peak'] - memory['initial']
            # update tqdm description with memory info
            mem_s = ', '.join(f'{k}:{int(v)}' for k, v in memory.items())
            pbar.set_description(
                f'Epoch {epoch:03d}, Loss: {loss:.4f}, Memory (MB): {mem_s}'
            )
        return torch.tensor(losses)

    def train_node_classification(self, dataset, n_epochs:int=100):
        model = NodeClassificationGAT(
                    in_channels=-1,#dataset.num_features,
                    hidden_channels=self.hidden_channels,
                    out_channels=dataset.num_classes,
                    heads=self.heads,
                    v2=False,
                    dropout=0.6,
                )

        def loss_fn(model):
            out = model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[self.data.train_mask], data.y[self.data.train_mask])
            return loss

        losses = self.train_model(model, loss_fn, n_epochs=n_epochs)
        return model

    def train_random_walks(self, walks: list[list[int]], n_epochs=5):
        """Train a graph model using random walk objectives.

        Pass in a list of `walks`, each of which is a list of node ids.
        """
        model = RandomWalkGAT(
            in_channels=self.data.num_features,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
            walks=walks,
        )
        def loss_fn(model):
            return model.compute_loss(self.data.x, self.data.edge_index)

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

    def train_and_eval_cls(self, embs):
        """Train and evaluate a node classification model on given `embs`.

        This trains various classification models on them.
        """
        y = self.data.y.cpu().numpy()
        train_mask = self.data.train_mask.cpu().numpy()
        test_mask = self.data.test_mask.cpu().numpy()
        X_train, y_train = embs[train_mask], y[train_mask]
        X_test, y_test = embs[test_mask], y[test_mask]
        logger.debug(f'Embeddings shape: {embs.shape}, train: {X_train.shape}, test: {X_test.shape}')
        models = dict(
            LinearSVC=LinearSVC(),
            SVC=SVC(kernel='rbf', gamma='scale'),
        )
        for name, model in models.items():
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            print(f'Classifier {name} accuracy: {acc:.4f}')


def load_data(name):
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

def baic_test():
    # load not Cora and not PubMed, Citeseer
    data, dataset = load_data('Citeseer')
    print(f'Device {device}, Loaded data: {data}, num_classes: {dataset.num_classes}, {data.y.shape}, {data.y}')
    #quick_test(data)
    mode = 'walk'
    gl = GraphLearner(data)
    gl.train_and_eval_cls(data.x.cpu().numpy())
    if mode == 'cls':
        model = gl.train_node_classification(dataset)
        eval_model(model, data)
    elif mode == 'walk':
        walks = gl.gen_walks(n_walks_per_node=1, walk_length=6)
        model = gl.train_random_walks(walks, n_epochs=5)
    embs = model.get_embeddings(data.x, data.edge_index).cpu().numpy()
    gl.train_and_eval_cls(embs)

def build_graph_from_features(feature_set: FeatureSet, 
                             n_nodes: int = None,
                             similarity_threshold: float = 0.5) -> Data:
    """Convert FeatureSet into PyG Data object.
    
    Args:
        feature_set: FeatureSet containing node features
        n_nodes: Number of nodes to sample (None for all)
        similarity_threshold: Minimum cosine similarity to create edge
        
    Returns:
        PyG Data object with node features and edge connectivity
    """
    # Get keys and embeddings
    keys, embeddings = feature_set.get_keys_embeddings()
    
    # Sample subset if requested
    if n_nodes is not None and n_nodes < len(keys):
        indices = np.random.choice(len(keys), n_nodes, replace=False)
        keys = [keys[i] for i in indices]
        embeddings = embeddings[indices]
    
    # Compute similarity matrix and create edges
    sim_matrix = cosine_similarity(embeddings)
    edge_indices = np.where(sim_matrix > similarity_threshold)
    
    # Remove self-loops
    mask = edge_indices[0] != edge_indices[1]
    edge_index = torch.tensor([edge_indices[0][mask], edge_indices[1][mask]], dtype=torch.long)
    
    # Convert to PyG Data object
    x = torch.tensor(embeddings, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.keys = keys  # Store original keys for later mapping
    
    logger.info(f"Built graph with {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def create_learner(learner_type: str, data: Data, **kwargs) -> GraphLearner:
    """Factory function to create the appropriate GAT learner.
    
    Args:
        learner_type: Type of learner ('node_classification', 'random_walk', 'contrastive')
        data: PyG Data object
        **kwargs: Additional parameters for the learner
        
    Returns:
        Configured GraphLearner instance
    """
    gl = GraphLearner(data, **kwargs)
    
    if learner_type == 'node_classification':
        # For node classification, we need to create synthetic labels
        # This is a placeholder - in practice you'd have real labels
        data.y = torch.randint(0, 3, (data.num_nodes,))  # 3 classes
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[:data.num_nodes//2] = True
        data.test_mask[data.num_nodes//2:] = True
        
    return gl


def save_embeddings(model: torch.nn.Module, 
                   data: Data, 
                   output_path: str, 
                   output_flag: str = 'c'):
    """Extract learned embeddings and save to NumpyLmdb.
    
    Args:
        model: Trained GAT model
        data: PyG Data object with original keys
        output_path: Path to output NumpyLmdb
        output_flag: LMDB flag for opening
    """
    # Extract embeddings
    embeddings = model.get_embeddings(data.x, data.edge_index).cpu().numpy()
    
    # Save to NumpyLmdb
    with NumpyLmdb.open(output_path, flag=output_flag) as db:
        for key, embedding in zip(data.keys, embeddings):
            db[key] = embedding
        
        # Save metadata
        db.md_set(db.global_key, 
                 model_type=model.__class__.__name__,
                 embedding_dim=embeddings.shape[1],
                 num_nodes=len(embeddings))
    
    logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")


def main():
    parser = ArgumentParser(description='Graph Learning Driver')
    
    # Input/Output
    parser.add_argument('-i', '--inputs', nargs='+', required=True,
                       help='Paths to FeatureSet inputs (NumpyLmdb files)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output NumpyLmdb path for learned embeddings')
    parser.add_argument('-f', '--output-flag', default='c', choices=['c', 'w', 'n'],
                       help='LMDB flag for output (default: c)')
    
    # Model configuration
    parser.add_argument('-t', '--learner-type', default='random_walk',
                       choices=['node_classification', 'random_walk', 'contrastive'],
                       help='Type of GAT learner (default: random_walk)')
    parser.add_argument('-n', '--n-nodes', type=int,
                       help='Number of nodes to sample from feature set')
    parser.add_argument('-w', '--walk-length', type=int, default=12,
                       help='Length of random walks (default: 12)')
    parser.add_argument('--n-walks-per-node', type=int, default=10,
                       help='Number of walks per node (default: 10)')
    parser.add_argument('--walk-window', type=int, default=5,
                       help='Context window for walks (default: 5)')
    
    # Architecture parameters
    parser.add_argument('-h', '--hidden-channels', type=int, default=64,
                       help='Hidden channels in GAT layers (default: 64)')
    parser.add_argument('--heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('-d', '--dropout', type=float, default=0.6,
                       help='Dropout rate (default: 0.6)')
    
    # Training parameters
    parser.add_argument('-e', '--n-epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('-b', '--batch-size', type=int, default=1024,
                       help='Batch size for training (default: 1024)')
    parser.add_argument('-s', '--similarity-threshold', type=float, default=0.5,
                       help='Similarity threshold for edge creation (default: 0.5)')
    
    args = parser.parse_args()
    
    # Load FeatureSet
    logger.info(f"Loading FeatureSet from {len(args.inputs)} inputs")
    feature_set = FeatureSet(args.inputs)
    logger.info(f"Loaded FeatureSet with {len(feature_set)} keys, {feature_set.n_dims} dims")
    
    # Build graph from features
    data = build_graph_from_features(
        feature_set, 
        n_nodes=args.n_nodes,
        similarity_threshold=args.similarity_threshold
    )
    
    # Create learner
    gl = create_learner(
        args.learner_type,
        data,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        dropout=args.dropout
    )
    
    # Train model
    logger.info(f"Training {args.learner_type} model for {args.n_epochs} epochs")
    
    if args.learner_type == 'node_classification':
        # Create synthetic dataset for node classification
        class SyntheticDataset:
            def __init__(self, num_classes=3):
                self.num_classes = num_classes
        
        dataset = SyntheticDataset()
        model = gl.train_node_classification(dataset, n_epochs=args.n_epochs)
        
    elif args.learner_type == 'random_walk':
        # Generate walks and train
        walks = gl.gen_walks(
            n_walks_per_node=args.n_walks_per_node,
            walk_length=args.walk_length
        )
        logger.info(f"Generated {walks.shape[0]} walks of length {walks.shape[1]}")
        
        model = gl.train_random_walks(walks, n_epochs=args.n_epochs)
        
    else:
        raise NotImplementedError(f"Learner type {args.learner_type} not implemented")
    
    # Save embeddings
    save_embeddings(model, data, args.output, args.output_flag)
    
    # Evaluate embeddings with simple classifier
    embeddings = model.get_embeddings(data.x, data.edge_index).cpu().numpy()
    logger.info(f"Final embeddings shape: {embeddings.shape}")
    logger.info(f"Embedding stats: mean={embeddings.mean():.3f}, std={embeddings.std():.3f}")


if __name__ == '__main__':
    main()
