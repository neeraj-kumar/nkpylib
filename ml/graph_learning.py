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

import functools
import gc
import logging
import time

from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass
from queue import Queue, Full, Empty
from threading import Thread, Event
from typing import Callable, Sequence, Any, Iterator

import joblib # type: ignore
import numpy as np
import numpy.random as npr
import psutil
import torch
import torch.nn.functional as F
import yaml

from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC, SVC # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GATv2Conv # type: ignore
from torch_geometric.nn.models import GAT # type: ignore
from torch_geometric.datasets import Planetoid # type: ignore
from torch_geometric.transforms import NormalizeFeatures # type: ignore
from torch_geometric.data import Data # type: ignore
from tqdm import tqdm

from nkpylib.ml.feature_set import (
    array1d,
    array2d,
    nparray1d,
    nparray2d,
    FeatureSet,
    NumpyLmdb,
)
from nkpylib.ml.ml_utils import trace, list_gpu_tensors
from nkpylib.ml.graph_worker import initialize_worker, worker_one_step, WorkItem

CFG: Namespace|None = None
RNG = npr.default_rng(0)
torch.manual_seed(0)

logger = logging.getLogger(__name__)

# default batch size
BATCH_SIZE = 128

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
                 dropout: float,
                 **kw):
        """Initialize this GAT model.

        - in_channels: Number of input node features
        - hidden_channels: Size of hidden layer embeddings
        - heads: Number of attention heads per layer
        - dropout: Dropout rate during training

        The model has two GATConv layers, each followed by ELU and dropout (during training).

        Embeddings are the concatenation of the outputs of the two GAT layers, so have size
        `hidden_channels * heads * 2`.

        Note that this base class doesn't actually do any classification - it just outputs the
        final node embeddings. You can add a classification layer on top of this, or use the
        embeddings for other tasks.
        """
        super().__init__()
        ModelCls = GATv2Conv if kw.get('v2', False) else GATConv
        logger.info(f'Initializing model {ModelCls}')
        self.conv1 = ModelCls(in_channels, hidden_channels, heads=heads)
        #self.conv2 = ModelCls(hidden_channels * heads, out_channels, heads=1, concat=False) # concat=False for final layer
        self.conv2 = ModelCls(hidden_channels * heads, hidden_channels * heads, heads=1)
        self.dropout = dropout
        self.process = psutil.Process()

    @trace
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


    @classmethod
    @abstractmethod
    def worker_one_step(cls, batch_size: int) -> WorkItem:
        """Worker function to generate a single work item for training."""
        pass

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

    @classmethod
    def worker_one_step(cls, batch_size: int) -> WorkItem:
        """Worker function to generate a single work item for training."""
        from nkpylib.ml.graph_worker import node_classification_one_step
        return node_classification_one_step(batch_size)


class ContrastiveGAT(GATBase):
    """GAT model trained using contrastive learning.

    This class implements the core contrastive learning logic, processing pairs
    of positive and negative examples to learn node embeddings.
    """
    def __init__(self,
                 temperature: float = 0.07,
                 **kw):
        """Initialize this ContrastiveGAT model.

        Args:
            - temperature: Temperature for similarity scaling (higher = softer attention)
        """
        super().__init__(**kw)
        self.temperature = temperature

    @classmethod
    def worker_one_step(cls, batch_size: int) -> WorkItem:
        """Worker function to generate a single work item for training."""
        from nkpylib.ml.graph_worker import random_walk_worker_one_step
        return random_walk_worker_one_step(batch_size)

    @trace
    def batch_loss(self,
                   embeddings: Tensor,
                   anchors: Tensor,
                   pos_nodes: Tensor,
                   neg_nodes: Tensor) -> Tensor:
        """Process a batch of nodes and compute the loss.

        Args:
        - embeddings: Node embeddings tensor
        - anchors: Anchor node indices
        - pos_nodes: Positive sample node indices
        - neg_nodes: Negative sample node indices

        Returns: Batch loss value
        """
        # Get embeddings for this batch
        anchor_embeds = embeddings[anchors]
        pos_embeds = embeddings[pos_nodes]
        n = anchor_embeds.size(0)
        #neg_embeds = embeddings[neg_nodes.view(-1)].view(n, self.neg_samples_factor, -1)
        # get embeds in the right shape regardless of their actual size (i.e., if we didn't gen
        # neg_samples_factor amount of them)
        neg_embeds = embeddings[neg_nodes]

        # Compute similarities
        cos = torch.nn.CosineSimilarity(dim=1)
        pos_sims = cos(anchor_embeds, pos_embeds) / self.temperature

        anchor_embeds_reshaped = anchor_embeds.unsqueeze(1)
        neg_embeds_reshaped = neg_embeds.transpose(1, 2)
        neg_sims = torch.bmm(anchor_embeds_reshaped, neg_embeds_reshaped).squeeze(1) / self.temperature

        # Compute loss
        all_sims = torch.cat([pos_sims.unsqueeze(1), neg_sims], dim=1)
        targets = torch.zeros(n, dtype=torch.long, device=embeddings.device)
        batch_loss = F.cross_entropy(all_sims, targets)
        #print(f'devices: {embeddings.device}, {anchor_embeds.device}, {pos_embeds.device}, {neg_embeds.device}')
        #print(f'{cos}, {pos_sims.device}, {neg_sims.device}, {all_sims.device}, {targets.device}, {batch_loss.device}')
        return batch_loss

    @trace
    def compute_loss(self,
                     item: WorkItem,
                     x,
                     gpu_batch_size: int = BATCH_SIZE,
                     use_checkpoint: bool = False):
        """Compute contrastive loss using pairs of nodes.

        Args:
        - x: Node features
        - edge_index: Graph connectivity
        - gpu_batch_size: Number of pairs to process at once

        Returns:
        - Average loss across all pairs
        """
        logger.info(f'Computing contrastive loss with batch size {gpu_batch_size}, use_checkpoint {use_checkpoint}')
        # Get embeddings
        cur_edges = torch.tensor(item.cur_edges).to(x.device)
        logger.info(f'Got edges of shape {cur_edges.shape} [{cur_edges.dtype}]')
        if use_checkpoint:
            embeddings = checkpoint(self.embedding_forward, x, cur_edges, use_reentrant=False)
        else:
            embeddings = self.embedding_forward(x, cur_edges).cpu()
        # accumulate loss in batches using our precomputed work items
        N = len(item.anchors)
        total_loss = 0
        total_pairs = 0
        for start in range(0, N, gpu_batch_size):
            end = min(start + gpu_batch_size, N)
            logger.debug(f'  Got {start}->{end}, {embeddings.shape}, {item.anchors[start:end].shape}, {item.pos_nodes[start:end].shape}, {item.neg_nodes[start:end].shape}: {item.anchors[start:end]}, {item.pos_nodes[start:end]}, {item.neg_nodes[start:end]}')
            # Process batch
            batch_loss = self.batch_loss(
                embeddings=embeddings,
                anchors=item.anchors[start:end],
                pos_nodes=item.pos_nodes[start:end],
                neg_nodes=item.neg_nodes[start:end],
            )
            total_loss += batch_loss
            total_pairs += end - start
        if total_pairs == 0:
            raise ValueError("No valid pairs found!")
        return total_loss / total_pairs


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
                 v2:bool=False,
                 n_jobs:int=4,
                 queue_size:int=5,
                 cpu_batch_size: int=BATCH_SIZE,
                 walk_length: int=12,
                 sample_edges: int=5,
                 walk_window: int=5,
                 neg_samples_factor: int=10,
                 do_async:bool=True,
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
        self.device = device
        self.data = data.to(self.device)
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.v2 = v2
        self.n_jobs = n_jobs
        self.kw = kw
        self.queue: Queue[WorkItem] = Queue(maxsize=queue_size)
        self.cpu_batch_size = cpu_batch_size
        self.sample_edges = sample_edges
        self.cpu_thread: Thread|None = None
        self.event = Event()
        self.do_async = do_async
        if do_async:
            # n_nodes, edge_index, walk length, max_edges_per_node
            initargs = (data.num_nodes, data.edge_index.to('cpu').numpy(), walk_length, sample_edges, walk_window, neg_samples_factor)
            self.pool = ProcessPoolExecutor(max_workers=n_jobs, initializer=initialize_worker, initargs=initargs)
j
    def start_cpu_thread(self, model: torch.nn.Module) -> None:
        """The main CPU thread, start this in a separate thread.

        This calls the model's `worker_one_step` function which should return a single
        work item. Note that we call this in parallel across multiple processes.

        It puts filled-in `WorkItem`s into the queue for the main (gpu) thread to consume.
        """
        print("Starting CPU thread")
        x = self.data.x
        # submit a bunch of jobs
        futures = [self.pool.submit(model.worker_one_step, self.cpu_batch_size) for _ in range(5)]
        while not self.event.is_set():
            # pop a single finished job and add it to the queue
            for f in as_completed(futures):
                item = f.result()
                futures.remove(f)
                break
            logger.info(f'Work item: {[(x.shape, x.dtype) for x in (item.anchors, item.pos_nodes, item.neg_nodes)]}')
            self.queue.put(item)
            # submit another job
            futures.append(self.pool.submit(model.worker_one_step, self.cpu_batch_size))

    def train_model(self,
                    model: torch.nn.Module,
                    loss_fn: Callable[[Any], Tensor],
                    n_epochs:int=200,
                    gpu_batch_size:int=BATCH_SIZE) -> Tensor:
        """Does actual model training in a loop.

        This does some bookkeeping around memory usage, etc.

        Args:
        - model: The graph model to train
        - loss_fn: A function that takes `(model)` and returns a loss tensor
        - n_epochs: Number of training epochs to run

        Returns the list of loss values per epoch.
        """
        process = psutil.Process()
        memory: Counter[str] = Counter()
        memory['initial'] = process.memory_info().rss / 1024 / 1024  # MB
        model = model.to(self.device)
        model.train()
        if self.do_async:
            self.cpu_thread = Thread(target=self.start_cpu_thread, daemon=True, args=(model,) )
            self.cpu_thread.start()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        pbar = tqdm(range(n_epochs), desc="Training Epochs")
        losses = []
        self.data.edge_index = self.data.edge_index.to('cpu')
        try:
            for epoch in pbar:
                optimizer.zero_grad()
                if self.do_async:
                    item = self.queue.get()
                    loss = model.compute_loss(item=item, x=self.data.x, gpu_batch_size=gpu_batch_size)
                else:
                    loss = loss_fn(model)
                if 0:
                    for i, tensor in enumerate(list_gpu_tensors()):
                        mem = int(tensor.nbytes/1024/1024)
                        if mem > 10:
                            print(f'  {i}: {tensor.size()} - {mem}MB - {tensor.dtype}')
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
        except KeyboardInterrupt:
            if epoch < 2:
                logger.warning("Training interrupted at epoch {epoch}. Quitting.")
                raise
            logger.warning(f"\nTraining interrupted at epoch {epoch}. Returning model with {len(losses)} epochs of training.")
            pbar.close()
        if self.do_async and self.cpu_thread is not None:
            self.event.set()
            self.cpu_thread.join(timeout=1.0)
            self.pool.shutdown(wait=False)
            #TODO the pool doesn't seem to be shutting down properly?
        losses = torch.tensor(losses)
        logger.info(f'Got final losses: {losses}')
        return losses

    def train_node_classification(self, dataset, n_epochs:int=100) -> tuple[Model, Tensor]:
        """Trains a model using node classification as the criteria, returning `(model, losses)`."""
        model = NodeClassificationGAT(
                    in_channels=-1,#dataset.num_features,
                    hidden_channels=self.hidden_channels,
                    out_channels=dataset.num_classes,
                    heads=self.heads,
                    v2=self.v2,
                    dropout=0.6,
                )

        def loss_fn(model):
            #FIXME this is not right
            out = model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
            return loss

        losses = self.train_model(model, loss_fn, n_epochs=n_epochs)
        return model, losses

    def train_random_walks(self, walk_length: int, n_epochs=5, gpu_batch_size:int=BATCH_SIZE) -> tuple[Model, Tensor]:
        """Train a graph model using random walk objectives, returning `(model, losses)`.

        Pass in the `walk_length` to use for generating walks. This creates a `WalkGenerator` that
        the model uses to generate walks on-demand during training. These positive pairs are sampled
        from these walks, and then randomly generated (and filtered) negative pairs are used for the
        contrastive learning setup.
        """
        batch_size = gpu_batch_size
        model = ContrastiveGAT(
            in_channels=self.data.num_features,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
            dropout=self.dropout,
            v2=self.v2,
        )
        def loss_fn(model):
            #FIXME this is not right
            return model.compute_loss(self.data.x, self.data.edge_index, batch_size=batch_size)

        losses = self.train_model(model, loss_fn, n_epochs=n_epochs, gpu_batch_size=batch_size)
        return model, losses

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

LEARNERS = dict(
    node_classification=NodeClassificationGAT,
    contrastive=ContrastiveGAT,
    random_walk=ContrastiveGAT,
)

def create_learner(learner_type: str, data: Data, **kwargs) -> GraphLearner:
    """Factory function to create the appropriate GAT learner.

    Args:
    - learner_type: Type of learner ('node_classification', 'random_walk', 'contrastive')
    - data: PyG Data object
    - **kwargs: Additional parameters for the learner

    Returns:
    - Configured GraphLearner instance
    """
    assert learner_type in LEARNERS, f"Unknown learner type: {learner_type}"
    gl = GraphLearner(data, **kwargs)
    match learner_type:
        case 'node_classification':
            raise NotImplementedError('Node classification not implemented in this example')
        case '_':
            pass  # No special setup needed
    return gl


def save_embeddings(model: torch.nn.Module,
                    data: Data,
                    output_path: str,
                    output_flag: str = 'c',
                    **kwargs,
                    ):
    """Extract learned embeddings and save to NumpyLmdb.

    Args:
    - model: Trained GAT model
    - data: PyG Data object with original keys
    - output_path: Path to output NumpyLmdb
    - output_flag: LMDB flag for opening
    - kwargs: Additional metadata to save in the database
    """
    # Extract embeddings (using cpu mode, since we're using the full data, which might overflow gpu)
    cur_device = 'cpu'
    model = model.to(cur_device)
    embeddings = model.get_embeddings(data.x.to(cur_device), data.edge_index.to(cur_device)).cpu().numpy()
    logger.info(f'Got embeddings of shape {embeddings.shape}: {embeddings}')

    # Save to NumpyLmdb
    with NumpyLmdb.open(output_path, flag=output_flag) as db:
        to_update = {}
        for key, embedding in zip(data['keys'], embeddings):
            #db[key] = embedding
            to_update[key] = embedding
        db.update(to_update)

        # Save metadata
        db.md_set(db.global_key,
                  created_ts=time.time(),
                  model_type=model.__class__.__name__,
                  n_nodes=len(embeddings),
                  n_edges=data.num_edges,
                  n_orig_dims=data.num_features,
                  n_embedding_dims=embeddings.shape[1],
                  **kwargs)

    logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")

def add_yaml_config_parsing(parser: ArgumentParser) -> Namespace:
    """This adds YAML config file parsing to a `parser`.

    This does it by first creating a dummy parser that only takes -c/--configs arguments,
    parsing it to get the config files, loading them, and then setting the defaults.

    Finally, it runs the actual parser with the remaining args and returns the final args.
    Note that you should NOT call parser.parse_args(), since this does that.
    """
    # First pass: get config files only
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument('-c', '--configs', nargs='*', help='YAML config files')
    config_args, remaining_args = config_parser.parse_known_args()
    # Load configs and merge them
    config_dict = {}
    if config_args.configs:
        for config_file in config_args.configs:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                config_dict.update(file_config)
    # Now set defaults on the actual parser and then run it
    parser.set_defaults(**config_dict)
    args = parser.parse_args(remaining_args)
    return args

def main():
    global CFG
    # create and setup arg parser
    parser = ArgumentParser(description='Graph Learning Driver')
    A = lambda *s, **kw: parser.add_argument(*s, **kw)
    # Input/Output
    A('input_path', help='Input PyG.Data path (as .pt)')
    A('output_path', help='Output NumpyLmdb path for learned embeddings')
    A('-f', '--output-flag', default='c', choices=['c', 'w', 'n'], help='LMDB flag for output [c]')
    # Model configuration
    A('-t', '--learner-type', default='random_walk', choices=LEARNERS, help='GAT learner [random_walk]')
    A('-n', '--n-nodes', type=int, default=50000, help='Number of nodes to sample from feature set')
    A('-w', '--walk-length', type=int, default=12, help='Length of random walks [12]')
    A('--walk-window', type=int, default=5, help='Context window for walks [5]')
    # Architecture parameters
    A('--hidden-channels', type=int, default=48, help='Hidden channels in GAT layers [64]')
    A('-H', '--heads', type=int, default=4, help='Number of attention heads [8]')
    A('-d', '--dropout', type=float, default=0.6, help='Training dropout rate [0.6]')
    # Training parameters
    A('-e', '--n-epochs', type=int, default=200, help='Number of training epochs [200]')
    A('--cpu-batch-size', type=int, default=128, help=f'Batch size for CPU [{BATCH_SIZE}]')
    A('--gpu-batch-size', type=int, default=128, help=f'Batch size for GPU [{BATCH_SIZE}]')
    A('-s', '--similarity-threshold', type=float, default=0.5, help='Similarity threshold for edge creation [0.5]')
    A('-j', '--n_jobs', type=int, default=2, help='Number of parallel jobs [6]')
    # first load configs if any (they override defaults, but not command-line args)
    #args = parser.parse_args()
    CFG = args = add_yaml_config_parsing(parser)
    # load input graph
    data = torch.load(args.input_path, weights_only=False)
    assert data.num_nodes < 2**31, "Number of nodes exceeds int32 range"
    data.edge_index = data.edge_index.to(torch.int32)
    logger.info(f'Loaded PyG from {args.input_path} with {data.num_nodes}x{data.num_features} nodes, {data.num_edges} edges, {data.x.dtype}, {data.edge_index.dtype}')
    if args.n_nodes:
        if 0: # proper sampling
            data.x = data.x[:args.n_nodes]
            data.edge_index = data.edge_index[:, (data.edge_index[0] < args.n_nodes) & (data.edge_index[1] < args.n_nodes)]
        else: # cutting off edges
            #data.edge_index = data.edge_index[:, :args.n_nodes*100]
            pass
        logger.info(f'Sampled to {data.num_nodes} nodes, now {data.num_edges} edges')
    if 0:
        # print all pairs in edge_index where one of them is a given idx
        print(data.edge_index)
        for idx in [1, 9739, 764, 55542]:
            indices = torch.where(data.edge_index == idx)
            pairs = data.edge_index[:, indices[1]].T
            print(f'{len(pairs)} Edges involving node {idx}: {pairs.T}')
        #return

    # Create learner
    gl = create_learner(
        args.learner_type,
        data,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        dropout=args.dropout,
        n_jobs=args.n_jobs,
        cpu_batch_size=args.cpu_batch_size,
        v2=False,
    )

    # Train model
    logger.info(f"Training {args.learner_type} model for {args.n_epochs} epochs")

    match args.learner_type:
        case 'random_walk': # Generate walks and train
            model, losses = gl.train_random_walks(walk_length=args.walk_length, n_epochs=args.n_epochs, gpu_batch_size=args.gpu_batch_size)
        case '_':
            raise NotImplementedError(f"Learner type {args.learner_type} not implemented")

    # Save embeddings
    kwargs = {f'kw_{name}': value for name, value in vars(args).items()}
    save_embeddings(model, data, args.output_path, args.output_flag, **kwargs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s', level=logging.INFO)
    logger.info(f'Got device {device}')
    main()
