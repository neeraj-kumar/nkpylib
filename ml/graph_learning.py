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
import logging
import time

from argparse import ArgumentParser
from collections import Counter, defaultdict
from typing import Callable, Sequence, Any, Iterator

import joblib
import numpy as np
import numpy.random as npr
import psutil
import torch
import torch.nn.functional as F

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

RNG = npr.default_rng(0)
torch.manual_seed(0)

logger = logging.getLogger(__name__)

INVALID_NODE = -1

# default batch size
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def explain_args(*args, **kwargs) -> dict[str|int, Any]:
    """This takes an arbitrary set of input `args` and `kwargs` and "explains" them.

    That means for each one:
    - if it's a scalar, then just return it as-is
    - if it's a list/array, then return its type and shape
    - if it's numpy array or torch tensor, then return its type and shape

    The output is a dict mapping argument index (for `args`) or name (for `kwargs`) to the explanation.
    """
    result = {}
    def process(arg):
        if hasattr(arg, 'shape'):  # numpy arrays, torch tensors
            ret = f'{type(arg).__name__}{arg.shape}'
            if hasattr(arg, 'dtype'):
                ret += f',{arg.dtype}'
        elif isinstance(arg, (list, tuple)):
            return f'{type(arg).__name__}[{len(arg)}]'
        elif isinstance(arg, dict):
            return f'dict[{len(arg)}]'
        else:  # scalars
            return arg

    # Process positional arguments
    for i, arg in enumerate(args):
        result[i] = process(arg)
    # Process keyword arguments
    for name, arg in kwargs.items():
        result[name] = process(arg)
    return result

def trace(func):
    """Decorator that tracks total time and memory delta for a function.

    Handles both regular functions and generator functions.
    For generators, traces the entire iteration lifecycle.
    """
    import inspect

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB

        def finish(suffix=''):
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            time_delta = end_time - start_time
            memory_delta = end_memory - start_memory
            # Temporarily override findCaller to return the traced function
            original_findCaller = logger.findCaller
            def fake_findCaller(*args, **kwargs):
                return ('', 0, func.__name__, None)

            logger.findCaller = fake_findCaller
            try:
                logger.info(f"{time_delta:.3f}s, {start_memory:.2f}GB -> {end_memory:.2f}GB (Δ{memory_delta:+.2f}GB) {suffix}: {explain_args(*args, **kwargs)}")
            finally:
                logger.findCaller = original_findCaller

        try:
            result = func(*args, **kwargs)

            # Check if result is a generator
            if inspect.isgenerator(result):
                def traced_generator():
                    try:
                        yield_count = 0
                        while True:
                            value = next(result)
                            yield_count += 1
                            yield value
                    except StopIteration:
                        finish(f' [generator, {yield_count} yields]')
                        return

                return traced_generator()
            else:
                # Regular function
                finish()
                return result
        except Exception as e:
            finish(' [FAILED]')
            raise

    return wrapper


class WalkGenerator:
    """Generates random walks through a graph with stateful batch generation."""
    def __init__(self,
                 n_nodes: int,
                 edge_index: torch.Tensor,
                 walk_length: int = 12,
                 n_jobs: int = 6):
        """Initialize walk generator with graph data and parameters.

        Args:
        - n_nodes: Total number of nodes
        - edge_index: Edge index tensor of shape [2, num_edges]
        - walk_length: Length of each random walk
        - n_jobs: Number of parallel jobs for walk generation
        """
        self.N = n_nodes
        self.edge_index = edge_index
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be of shape [2, num_edges]"
        self.walk_length = walk_length
        self.n_jobs = n_jobs
        self.current_index = 0

        # Build adjacency matrix for fast neighbor lookup
        edges = self.edge_index.cpu().numpy()
        self.adj = csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(self.N, self.N))

    def _gen_batch(self, batch_start: int, batch_size: int) -> nparray2d:
        """Generate a single batch of walks starting at `batch_start`"""
        walks = np.ones((batch_size, self.walk_length), dtype=np.int32) * INVALID_NODE
        walks[:, 0] = np.arange(batch_start, batch_start + batch_size) % self.N
        # Generate walks (vectorized)
        for step in range(1, self.walk_length):
            current_nodes = walks[:, step-1]
            for i, node in enumerate(current_nodes):
                neighbors = self.adj[node].indices
                if len(neighbors) > 0:
                    walks[i, step] = RNG.choice(neighbors)
                else:
                    walks[i, step] = INVALID_NODE
        return walks

    @trace
    def gen_walks(self, n_walks: int) -> nparray2d:
        """Generate `n_walks` random walks, continuing from current index.

        This is a wrapper around self._gen_batch that handles parallelization.

        Args:
        - n_walks: Number of walks to generate

        Returns:
        - Array of walks with shape (n_walks, walk_length)
        """
        if self.n_jobs > 1:
            # Parallel generation
            batch_size = n_walks // self.n_jobs
            results = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._gen_batch)(
                    self.current_index + i * batch_size,
                    min(batch_size, n_walks - i * batch_size)
                )
                for i in range(self.n_jobs)
            )
            walks = np.vstack(results)
        else:
            # Single-threaded generation
            walks = self._gen_batch(self.current_index, n_walks)
        # Update current index for next batch
        self.current_index = (self.current_index + n_walks) % self.N
        return walks

    def reset_index(self):
        """Reset the current index to 0."""
        self.current_index = 0


class EdgeSampler:
    """Efficient edge sampling with caching for memory optimization."""
    def __init__(self,
                 edge_index: torch.Tensor,
                 max_edges_per_node: int,
                 proportional: bool = True):
        """Initialize edge sampler with caching.

        Args:
        - edge_index: Graph edge index tensor of shape [2, num_edges]
        - max_edges_per_node: Maximum number of edges to sample per node
        - proportional: If True, sample edges proportionally to their degree (capping to max_edges_per_node)
        """
        self.max_edges_per_node = max_edges_per_node
        self.edge_index = edge_index
        self.proportional = proportional
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be of shape [2, num_edges]"
        # Cache the edge grouping (do once, reuse many times)
        self._build_edge_groups()

    @trace
    def _build_edge_groups(self):
        """Build edge groups efficiently and cache them."""
        src_nodes = self.edge_index[0]
        unique_nodes, inverse_indices, counts = torch.unique(
            src_nodes, return_inverse=True, return_counts=True
        )
        # Pre-allocate storage
        max_node = src_nodes.max().item()
        self.node_edge_starts = torch.zeros(max_node + 1, dtype=torch.long)
        self.node_edge_counts = torch.zeros(max_node + 1, dtype=torch.long)
        # Sort edges by source node for efficient slicing
        sorted_indices = torch.argsort(src_nodes)
        self.sorted_edge_indices = sorted_indices
        # Store start positions and counts for each node
        start_pos = 0
        for node, count in zip(unique_nodes, counts):
            self.node_edge_starts[node] = start_pos
            self.node_edge_counts[node] = count
            start_pos += count

    @trace
    def sample(self, seed: int = None) -> torch.Tensor:
        """Sample edges efficiently using cached structure.

        Args:
            seed: Random seed for reproducible sampling

        Returns:
            Sampled edge_index tensor of shape [2, num_sampled_edges]
        """
        if seed is not None:
            torch.manual_seed(seed)
        sampled_indices = []
        for node in range(len(self.node_edge_starts)):
            count = self.node_edge_counts[node]
            if count == 0:
                continue
            start = self.node_edge_starts[node]
            end = start + count
            n_sample = self.max_edges_per_node
            if self.proportional:
                ratio = min(1.0, self.max_edges_per_node / count)
                n_sample = max(1, max(int(count * ratio), self.max_edges_per_node))
            if n_sample >= count: # take all edges
                node_edges = self.sorted_edge_indices[start:end]
            else:
                node_edges = self.sorted_edge_indices[start:end]
                perm = torch.randperm(count)[:n_sample]
                node_edges = node_edges[perm]
            sampled_indices.append(node_edges)
        if sampled_indices:
            sampled_indices = torch.cat(sampled_indices)
            ret = self.edge_index[:, sampled_indices]
        else:
            ret = torch.empty((2, 0), dtype=self.edge_index.dtype)
        return ret


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
                 negative_samples: int = 2, # should be 10?
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
        self.edge_sampler = None

    @trace
    def batch_loss(self,
                      embeddings: torch.Tensor,
                      anchors: torch.Tensor,
                      pos_nodes: torch.Tensor,
                      neg_nodes: torch.Tensor,
                      cur_batch_size: int) -> torch.Tensor:
        """Process a batch of nodes and compute the loss.

        Args:
        - embeddings: Node embeddings tensor
        - anchors: Anchor node indices
        - pos_nodes: Positive sample node indices
        - neg_nodes: Negative sample node indices
        - cur_batch_size: Size of current batch

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

    def pos_pair_generator(self, batch_size: int) -> Iterator[tuple[Tensor, Tensor]]:
        """Generates positive pairs of nodes for contrastive learning.

        This should yield tuples of `(anchors, positive_nodes)`
        """
        raise NotImplementedError("Subclasses must implement pos_pair_generator()")

    @trace
    def compute_loss(self,
                     x,
                     edge_index,
                     batch_size: int = BATCH_SIZE,
                     sample_edges: int=10,
                     use_checkpoint: bool = False):
        """Compute contrastive loss using pairs of nodes.

        Args:
        - x: Node features
        - edge_index: Graph connectivity
        - batch_size: Number of pairs to process at once

        Returns:
        - Average loss across all pairs
        """
        logger.info(f'Computing contrastive loss with batch size {batch_size}, sample_edges {sample_edges}, use_checkpoint {use_checkpoint}')
        # Get embeddings
        if sample_edges > 0:
            if not self.edge_sampler or self.edge_sampler.max_edges_per_node != sample_edges:
                self.edge_sampler = EdgeSampler(edge_index, sample_edges)
            cur_edges = self.edge_sampler.sample()
        else:
            cur_edges = edge_index
        logger.info(f'Got edges of shape {cur_edges.shape} vs {edge_index.shape}, {cur_edges.dtype} for loss computation')
        if use_checkpoint:
            embeddings = checkpoint(self.embedding_forward, x, cur_edges, use_reentrant=False)
        else:
            embeddings = self.embedding_forward(x, cur_edges).cpu()

        # accumulate loss in batches using pairs
        total_loss = 0
        total_pairs = 0
        for anchors, pos_nodes in self.pos_pair_generator(batch_size=batch_size):
            logger.debug(f'  Processing batch with {len(anchors)} pairs')
            cur_batch_size = len(anchors)

            # Generate negative samples
            with torch.no_grad():
                neg_nodes = torch.randint(0, x.shape[0], (cur_batch_size, self.negative_samples))
                #TODO filter out anchors, actual neighbors, and nodes from walks (if possible)

            # Process batch
            batch_loss = self.batch_loss(
                embeddings=embeddings,
                anchors=anchors,
                pos_nodes=pos_nodes,
                neg_nodes=neg_nodes,
                cur_batch_size=cur_batch_size,
            )

            total_loss += batch_loss #TODO what type is this? tensor? int?
            total_pairs += cur_batch_size

        if total_pairs == 0:
            raise ValueError("No valid pairs found!")

        return total_loss / total_pairs


class RandomWalkGAT(ContrastiveGAT):
    """GAT model trained using random walk objectives.

    Instead of node classification, this learns node embeddings such that nodes
    appearing close together in random walks have similar embeddings.

    Hence, this is a subclass of ContrastiveGAT, as we generate positive and negative node pairs
    based on closeness within walks.
    """
    def __init__(self, walk_gen: WalkGenerator, walk_window: int = 5, **kw):
        """Initialize this RandomWalkGAT model.

        Args:
        - walk_gen: A walk generator
        - walk_window: Context window size for walks (how many nodes before/after to consider)
        """
        super().__init__(**kw)
        self.walk_gen = walk_gen
        self.walk_window = walk_window

    @trace
    def pos_pair_generator(self, batch_size: int) -> Iterator[tuple[Tensor, Tensor]]:
        """Generate walks on-demand and yield positive pairs.

        This generates only as many walks as needed to produce a batch of positive pairs,
        reducing memory usage compared to pre-generating all walks.

        Args:
        - batch_size: Target number of positive pairs per batch

        Yields:
        - Tuples of (anchor_nodes, positive_nodes) tensors
        """
        logger.info(f'Starting pos_pair_generator with batch_size={batch_size}, walk_window={self.walk_window}')
        pairs_generated = 0
        iteration = 0
        
        while pairs_generated < batch_size:
            iteration += 1
            # Estimate how many walks we need for remaining pairs
            # Because we filter out various pairs, generate a bunch more than we need
            remaining_pairs = batch_size - pairs_generated
            walks_needed = max(1, remaining_pairs)

            # Generate a small batch of walks on-demand
            logger.info(f'Iteration {iteration}: Generating {walks_needed} walks to get {remaining_pairs} more pairs ({batch_size}, {pairs_generated}), based on window {self.walk_window}')
            batch_walks = self.walk_gen.gen_walks(walks_needed)
            logger.info(f'Generated batch_walks shape: {batch_walks.shape}, dtype: {batch_walks.dtype}')
            logger.info(f'First few walks: {batch_walks[:3] if len(batch_walks) > 0 else "EMPTY"}')

            # Convert to tensor and extract pairs
            walks_tensor = torch.tensor(batch_walks)
            valid_mask = walks_tensor != INVALID_NODE
            walk_length = walks_tensor.shape[1]
            logger.info(f'walks_tensor shape: {walks_tensor.shape}, valid_mask shape: {valid_mask.shape}')
            logger.info(f'Total valid nodes in walks: {valid_mask.sum().item()}/{valid_mask.numel()}')

            batch_anchors = []
            batch_positives = []

            # Extract pairs from these walks
            for i in range(walk_length):
                pos_mask = valid_mask[:, i].clone()
                logger.debug(f'Position {i}: pos_mask has {pos_mask.sum().item()} valid nodes out of {len(pos_mask)}')
                if not pos_mask.any():
                    logger.debug(f'Position {i}: No valid nodes, skipping')
                    continue

                pos_walks = pos_mask.nonzero().squeeze(1)
                if len(pos_walks.shape) == 0:
                    pos_walks = pos_walks.unsqueeze(0)
                logger.debug(f'Position {i}: Processing {len(pos_walks)} walks')

                for walk_idx in pos_walks:
                    # Get context window for this walk position
                    start = max(0, i - self.walk_window)
                    end = min(walk_length, i + self.walk_window + 1)
                    context = walks_tensor[walk_idx, start:end]
                    context_mask = valid_mask[walk_idx, start:end].clone()
                    context_mask[i-start] = False  # Exclude anchor position
                    
                    logger.debug(f'Walk {walk_idx.item()}, pos {i}: context window [{start}:{end}], context={context.tolist()}, mask={context_mask.tolist()}')

                    valid_context = context[context_mask]
                    logger.debug(f'Walk {walk_idx.item()}, pos {i}: valid_context={valid_context.tolist() if len(valid_context) > 0 else "EMPTY"}')
                    
                    if len(valid_context) > 0:
                        anchor_node = walks_tensor[walk_idx, i]
                        logger.debug(f'Adding {len(valid_context)} pairs with anchor {anchor_node.item()}')
                        batch_anchors.extend([anchor_node] * len(valid_context))
                        batch_positives.extend(valid_context.tolist())

                        # Check if we have enough pairs for this batch
                        if len(batch_anchors) >= remaining_pairs:
                            logger.debug(f'Reached target pairs ({len(batch_anchors)} >= {remaining_pairs}), breaking')
                            break
                    else:
                        logger.debug(f'Walk {walk_idx.item()}, pos {i}: No valid context nodes')
                        
                if len(batch_anchors) >= remaining_pairs:
                    logger.debug(f'Breaking from position loop, have {len(batch_anchors)} pairs')
                    break
                    
            logger.info(f'Iteration {iteration}: Generated {len(batch_anchors)} positive pairs from {walks_tensor.shape} walks, needed {remaining_pairs}')
            
            # Yield pairs if we have any
            if batch_anchors:
                # Limit to exactly the number of pairs we need
                n_pairs = min(len(batch_anchors), remaining_pairs)
                logger.info(f'Yielding {n_pairs} pairs (anchors: {batch_anchors[:3]}..., positives: {batch_positives[:3]}...)')
                yield (
                    torch.tensor(batch_anchors[:n_pairs]),
                    torch.tensor(batch_positives[:n_pairs])
                )
                pairs_generated += n_pairs
                logger.info(f'Total pairs generated so far: {pairs_generated}/{batch_size}')
            else:
                logger.warning(f'Iteration {iteration}: No pairs generated from {walks_tensor.shape[0]} walks!')
                # Prevent infinite loop if no pairs can be generated
                if iteration > 10:
                    logger.error(f'Breaking after {iteration} iterations with no pairs generated')
                    break


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
                 n_jobs:int=6,
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = data.to(self.device)
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.n_jobs = n_jobs
        self.kw = kw

    def train_model(self,
                    model: torch.nn.Module,
                    loss_fn: Callable[[Any], torch.Tensor],
                    n_epochs:int=200,
                    batch_size:int=BATCH_SIZE) -> torch.Tensor:
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        pbar = tqdm(range(n_epochs), desc="Training Epochs")
        losses = []
        try:
            for epoch in pbar:
                logger.info('setting 0 grad')
                optimizer.zero_grad()
                logger.info(f'computing loss for epoch {epoch}')
                loss = loss_fn(model)
                losses.append(loss.item())
                logger.info(f'running backward')
                loss.backward()
                logger.info(f'incrementing optimizer')
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
            logger.warning(f"\nTraining interrupted at epoch {epoch}. Returning model with {len(losses)} epochs of training.")
            pbar.close()
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
            loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
            return loss

        losses = self.train_model(model, loss_fn, n_epochs=n_epochs)
        return model

    def train_random_walks(self, walk_length: int, n_epochs=5, batch_size:int=BATCH_SIZE):
        """Train a graph model using random walk objectives.

        Pass in the `walk_length` to use for generating walks. This creates a `WalkGenerator` that
        the model uses to generate walks on-demand during training. These positive pairs are sampled
        from these walks, and then randomly generated (and filtered) negative pairs are used for the
        contrastive learning setup.
        """
        model = RandomWalkGAT(
            in_channels=self.data.num_features,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
            dropout=self.dropout,
            walk_gen=WalkGenerator(
                n_nodes=self.data.num_nodes,
                edge_index=self.data.edge_index,
                walk_length=walk_length,
                n_jobs=self.n_jobs,
            ),
        )
        def loss_fn(model):
            return model.compute_loss(self.data.x, self.data.edge_index, batch_size=batch_size)

        losses = self.train_model(model, loss_fn, n_epochs=n_epochs, batch_size=batch_size)
        return model

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
    random_walk=RandomWalkGAT,
    contrastive=ContrastiveGAT,
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
    # Extract embeddings
    embeddings = model.get_embeddings(data.x, data.edge_index).cpu().numpy()
    logger.info(f'Got embeddings of shape {embeddings.shape}: {embeddings}')

    # Save to NumpyLmdb
    with NumpyLmdb.open(output_path, flag=output_flag) as db:
        for key, embedding in zip(data['keys'], embeddings):
            db[key] = embedding

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


def main():
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
    A('-w', '--walk-length', type=int, default=7, help='Length of random walks [12]')
    A('--n-walks-per-node', type=int, default=1, help='Number of walks per node [10]')
    A('--walk-window', type=int, default=5, help='Context window for walks [5]')
    # Architecture parameters
    A('-c', '--hidden-channels', type=int, default=8, help='Hidden channels in GAT layers [64]')
    A('-H', '--heads', type=int, default=4, help='Number of attention heads [8]')
    A('-d', '--dropout', type=float, default=0.6, help='Training dropout rate [0.6]')
    # Training parameters
    A('-e', '--n-epochs', type=int, default=20, help='Number of training epochs [200]')
    A('-b', '--batch-size', type=int, default=4, help=f'Batch size for training [{BATCH_SIZE}]')
    A('-s', '--similarity-threshold', type=float, default=0.5, help='Similarity threshold for edge creation [0.5]')
    A('-j', '--n_jobs', type=int, default=6, help='Number of parallel jobs [6]')
    # 50k with checkpointing, about 10Gb gpu steady, but peaked earlier at 14
    args = parser.parse_args()
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
    )

    # Train model
    logger.info(f"Training {args.learner_type} model for {args.n_epochs} epochs")

    match args.learner_type:
        case 'random_walk': # Generate walks and train
            model = gl.train_random_walks(walk_length=args.walk_length, n_epochs=args.n_epochs, batch_size=args.batch_size)
        case '_':
            raise NotImplementedError(f"Learner type {args.learner_type} not implemented")

    # Save embeddings
    kwargs = {f'kw_{name}': value for name, value in vars(args).items()}
    save_embeddings(model, data, args.output_path, args.output_flag, **kwargs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s', level=logging.INFO)
    logger.info(f'Got device {device}')
    main()
