"""Some utilities for working with graphs, especially in separate processes."""

from __future__ import annotations

import functools
import gc
import logging
import sys
import time

from abc import abstractmethod
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass
from hashlib import sha256
from queue import Queue, Full, Empty
from threading import Thread
from typing import Callable, Sequence, Any, Iterator

import joblib # type: ignore
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

from nkpylib.ml.ml_types import (
    array1d,
    array2d,
    nparray1d,
    nparray2d,
)
from nkpylib.ml.ml_utils import trace

logger = logging.getLogger(__name__)

RNG = npr.default_rng(0)
torch.manual_seed(0)

INVALID_NODE = -1

@dataclass
class WorkItem:
    """A single work item for async processing."""
    cur_edges: nparray2d # The current edges to use
    anchors: Tensor # Anchor nodes
    pos_nodes: Tensor # Positive nodes
    neg_nodes: Tensor # Negative nodes

@dataclass
class WorkerObj:
    """A container for data local to a process worker"""
    walk_gen: WalkGenerator
    edge_sampler: EdgeSampler
    walk_window: int
    neg_samples_factor: int

# global worker obj
_worker_obj: WorkerObj|None = None

def initialize_worker(n_nodes: int,
                      edge_index: nparray2d,
                      walk_length: int,
                      max_edges_per_node: int,
                      walk_window: int,
                      neg_samples_factor: int,
                      ):
    """Initialize the global worker object for the process."""
    global _worker_obj
    if _worker_obj is None:
        logger.info('Initializing worker process')
        walk_gen = WalkGenerator(
            n_nodes=n_nodes,
            edge_index=edge_index,
            walk_length=walk_length,
            n_jobs=1,
        )
        edge_sampler = EdgeSampler(
            edge_index=edge_index,
            max_edges_per_node=max_edges_per_node,
            proportional=True,
            global_sampling=True,
        )
        _worker_obj = WorkerObj(walk_gen=walk_gen,
                                edge_sampler=edge_sampler,
                                walk_window=walk_window,
                                neg_samples_factor=neg_samples_factor)


@trace
def worker_one_step(n_pos: int) -> WorkItem:
    """Runs "one step" of processing in the worker process."""
    global _worker_obj
    assert _worker_obj is not None
    cur_edges = _worker_obj.edge_sampler.sample()
    anchors, pos_nodes = pos_pair_generator(
        cur_edges=cur_edges,
        batch_size=n_pos,
        walk_gen=_worker_obj.walk_gen,
        walk_window=_worker_obj.walk_window,
    )
    neg_nodes = cpu_neg_pair_generator(
        n_nodes=_worker_obj.walk_gen.N,
        anchors=anchors,
        pos_nodes=pos_nodes,
        edge_index=cur_edges,
        shape=(len(anchors), _worker_obj.neg_samples_factor),
    )
    return WorkItem(
        cur_edges=cur_edges,
        anchors=anchors,
        pos_nodes=pos_nodes,
        neg_nodes=neg_nodes,
    )


def gen_random(n_edges, size):
    return RNG.choice(n_edges, size=size, replace=False)


class WalkGenerator:
    """Generates random walks through a graph with stateful batch generation."""
    def __init__(self,
                 n_nodes: int,
                 edge_index: nparray2d,
                 walk_length: int = 12,
                 n_jobs: int = 6):
        """Initialize walk generator with graph data and parameters.

        Args:
        - n_nodes: Total number of nodes
        - edge_index: Edge index tensor of shape [2, num_edges]
        - walk_length: Length of each random walk
        - n_jobs: Number of parallel jobs for walk generation (if > 1)
        """
        self.N = n_nodes
        self.edge_index = edge_index
        assert len(edge_index.shape) == 2 and edge_index.shape[0] == 2, "edge_index must be of shape [2, num_edges]"
        self.walk_length = walk_length
        self.n_jobs = n_jobs
        self.current_index = 0

        # Build adjacency matrix for fast neighbor lookup
        #edges = self.edge_index.cpu().numpy()
        self._edge_hash = ''
        self._adj: np.ndarray|None = None
        self._maybe_rebuild_adj(edge_index)

    @trace
    def _maybe_rebuild_adj(self, edge_index=None):
        if edge_index is None:
            return
        # Only rebuild if edges changed
        new_hash = sha256(edge_index.tobytes()).hexdigest()
        if not self._edge_hash or self._edge_hash != new_hash:
            self._adj = csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                                   shape=(self.N, self.N))
            self._edge_hash = new_hash
        mem = sys.getsizeof(self._adj.data) + sys.getsizeof(self._adj.indptr) + sys.getsizeof(self._adj.indices)
        mb = lambda x: f'{x / 1024 / 1024:.2f}MB'
        logger.debug(f'Size of adj: {mb(mem)}, vs edges {mb(edge_index.nbytes)} vs global edges {mb(self.edge_index.nbytes)}')

    @trace
    def _gen_batch(self, batch_start: int, batch_size: int) -> nparray2d:
        """Generate a single batch of walks starting at `batch_start`"""
        assert batch_size > 0
        assert self._adj is not None
        logger.debug(f'Generating batch from {batch_start} size {batch_size}, walk_length {self.walk_length}, {self.current_index}, {self.N}, {self.edge_index.shape}')
        times = [time.time()]
        walks = np.ones((batch_size, self.walk_length), dtype=np.int32) * INVALID_NODE
        walks[:, 0] = np.arange(batch_start, batch_start + batch_size) % self.N
        times.append(time.time())
        #max_degree = max(len(self._adj[i].indices) for i in range(self.N)) # very slow!!
        max_degree = 200
        times.append(time.time())
        random_choices = RNG.integers(0, max_degree, size=(batch_size, self.walk_length-1))
        times.append(time.time())
        #print(f'Initial walks: {walks}')
        # Generate walks (vectorized)
        for step in range(1, self.walk_length):
            current_nodes = walks[:, step-1]
            for i, node in enumerate(current_nodes):
                if node != INVALID_NODE:
                    neighbors = self._adj[node].indices
                    if len(neighbors) > 0:
                        #walks[i, step] = RNG.choice(neighbors)
                        choice_idx = random_choices[i, step-1] % len(neighbors)
                        walks[i, step] = neighbors[choice_idx]
            #print(f'  Step {step}, cur: {current_nodes}, walks {walks}')
        times.append(time.time())
        print(f'Times (max {max_degree}): {[t1-t0 for t0, t1 in zip(times, times[1:])]}')
        return walks

    @trace
    def gen_walks(self, n_walks: int, cur_edges=None) -> nparray2d:
        """Generate `n_walks` random walks, continuing from current index.

        This is a wrapper around self._gen_batch that handles parallelization.

        Args:
        - n_walks: Number of walks to generate

        Returns:
        - Array of walks with shape (n_walks, walk_length)
        """
        self._maybe_rebuild_adj(cur_edges)
        assert self._adj is not None
        if False and self.n_jobs > 1: #FIXME this doesn't seem to help for small batches
            # Parallel generation
            batch_size = max(1, n_walks // self.n_jobs)
            results = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._gen_batch)(
                    self.current_index + (i * batch_size), batch_size,
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
                 edge_index: nparray2d,
                 max_edges_per_node: int,
                 proportional: bool = True,
                 global_sampling: bool = True):
        """Initialize edge sampler with caching.

        Args:
        - edge_index: Graph edge index tensor of shape [2, num_edges]
        - max_edges_per_node: Maximum number of edges to sample per node
        - proportional: If True, sample edges proportionally to their degree (capping to max_edges_per_node)
        - global_sampling: If True, use fast global random sampling instead of per-node sampling
        """
        self.max_edges_per_node = max_edges_per_node
        self.edge_index = edge_index
        self.proportional = proportional
        self.global_sampling = global_sampling
        assert len(edge_index.shape) == 2 and edge_index.shape[0] == 2, "edge_index must be of shape [2, num_edges]"
        # Cache the edge grouping (do once, reuse many times) - only needed for per-node sampling
        if not self.global_sampling:
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
    def sample(self) -> nparray2d:
        """Sample edges efficiently.

        Returns:
        - Sampled edge_index tensor of shape [2, num_sampled_edges]
        """
        if self.global_sampling:
            return self._sample_global()
        else:
            return self._sample_per_node()

    def _sample_global(self) -> nparray2d:
        """Fast global random sampling of edges."""
        #TODO the most expensive part is the random call, so do that in parallel in batch
        n_edges = self.edge_index.shape[1]
        if n_edges == 0:
            return torch.empty((2, 0), dtype=self.edge_index.dtype).numpy()

        # Calculate target number of edges based on max_edges_per_node
        # Estimate: if we have N nodes and want max_edges_per_node per node on average
        times = [time.time()]
        n_nodes = max(self.edge_index.max().item() + 1, 1)
        times.append(time.time())
        target_edges = min(n_edges, int(n_nodes * self.max_edges_per_node))
        pool = ProcessPoolExecutor(max_workers=6)
        times.append(time.time())

        # Simple random sampling
        # torch randperm is ~2s, RNG.choice is ~0.3s, scales almost linearly with size
        #indices = torch.randperm(n_edges)[:target_edges] # type: ignore[misc]
        indices = RNG.choice(n_edges, size=target_edges, replace=False)
        #futures = [pool.submit(gen_random, n_edges, target_edges) for _ in range(12)]
        #all_indices = [future.result() for future in futures]
        #indices = all_indices[0]
        times.append(time.time())
        ret = self.edge_index[:, indices]
        times.append(time.time())
        per = n_edges / (times[-1] - times[0])
        print(f'S times ({per}): {[t1 - t0 for t0, t1 in zip(times, times[1:])]}')
        #print(f'All inds: {len(all_indices)}: {all_indices}')
        return ret

    def _sample_per_node(self) -> nparray2d:
        """Per-node sampling preserving degree distribution."""
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
            _sampled_indices = torch.cat(sampled_indices)
            ret = self.edge_index[:, _sampled_indices]
        else:
            ret = torch.empty((2, 0), dtype=self.edge_index.dtype).numpy()
        return ret

@trace
def pos_pair_generator(cur_edges: Tensor, batch_size: int, walk_gen: WalkGenerator, walk_window: int) -> tuple[Tensor, Tensor]:
    """Generate walks on-demand and yield positive pairs.

    This generates only as many walks as needed to produce a batch of positive pairs,
    reducing memory usage compared to pre-generating all walks.

    Args:
    - cur_edges: Current set of sampled edges
    - batch_size: Target number of positive pairs per batch

    Returns:
    - `(anchor_nodes, positive_nodes)` tensors
    """
    logger.debug(f'Starting pos_pair_generator with batch_size={batch_size}, walk_window={walk_window}')
    pairs_generated = 0
    iteration = 0
    ret_anchors, ret_pos = [], []
    while pairs_generated < batch_size:
        iteration += 1
        # Estimate how many walks we need for remaining pairs
        # Because we filter out various pairs, generate a bunch more than we need
        remaining_pairs = batch_size - pairs_generated
        walks_needed = max(1, remaining_pairs)

        # Generate a small batch of walks on-demand
        logger.debug(f'Iteration {iteration}: Generating {walks_needed} walks to get {remaining_pairs} more pairs ({batch_size}, {pairs_generated}), window {walk_window}')
        batch_walks = walk_gen.gen_walks(walks_needed, cur_edges=cur_edges)
        logger.debug(f'Generated batch_walks shape: {batch_walks.shape}, dtype: {batch_walks.dtype}: {batch_walks[:10]}')

        # Convert to tensor and extract pairs
        walks_tensor = torch.tensor(batch_walks)
        valid_mask = walks_tensor != INVALID_NODE
        walk_length = walks_tensor.shape[1]
        logger.debug(f'walks_tensor shape: {walks_tensor.shape}, valid_mask shape: {valid_mask.shape}')
        logger.debug(f'Total valid nodes in walks: {valid_mask.sum().item()}/{valid_mask.numel()}')

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
                start = max(0, i - walk_window)
                end = min(walk_length, i + walk_window + 1)
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
        logger.debug(f'Iteration {iteration}: Generated {len(batch_anchors)} positive pairs from {walks_tensor.shape} walks, needed {remaining_pairs}')
        # Accumulate pairs if we have any
        if batch_anchors:
            # Limit to exactly the number of pairs we need
            n_pairs = min(len(batch_anchors), remaining_pairs)
            logger.debug(f'Yielding {n_pairs} pairs (anchors: {batch_anchors[:3]}..., positives: {batch_positives[:3]}...)')
            ret_anchors.extend(batch_anchors[:n_pairs])
            ret_pos.extend(batch_positives[:n_pairs])
            pairs_generated += n_pairs
            logger.debug(f'Total pairs generated so far: {pairs_generated}/{batch_size}')
        else:
            logger.warning(f'Iteration {iteration}: No pairs generated from {walks_tensor.shape[0]} walks!')
            # Prevent infinite loop if no pairs can be generated
            if iteration > 10:
                logger.error(f'Breaking after {iteration} iterations with no pairs generated')
                break
    return torch.tensor(ret_anchors), torch.tensor(ret_pos)


@trace
def neg_pair_generator(n_nodes: int,
                       anchors: Tensor,
                       pos_nodes: Tensor,
                       edge_index: Tensor,
                       shape: tuple[int, int],
                       walks: Tensor|None = None) -> Tensor:
    """Generate negative samples while filtering out invalid nodes.

    Args:
    - n_nodes: total number of nodes in graph
    - anchors: Current batch anchor nodes to exclude
    - pos_nodes: Current batch positive nodes to exclude
    - edge_index: Graph edges to identify actual neighbors to exclude
    - shape: Target shape (batch_size, neg_samples_factor)
    - walks: Optional tensor of current walks to exclude walk neighbors

    Returns:
    - Tensor of negative node indices with shape `shape`
    """
    batch_size, neg_samples = shape

    # Build exclusion sets for each anchor
    exclude_sets = []
    for i, anchor in enumerate(anchors):
        exclude = set([anchor.item(), pos_nodes[i].item()])
        # Add graph neighbors
        neighbors = edge_index[1][edge_index[0] == anchor]
        exclude.update(neighbors.tolist())
        # Add walk neighbors if provided
        if walks is not None:
            # Find walks containing this anchor and exclude those nodes
            walk_mask = (walks == anchor).any(dim=1)
            if walk_mask.any():
                walk_neighbors = walks[walk_mask].flatten()
                # Filter out invalid nodes
                valid_walk_neighbors = walk_neighbors[walk_neighbors != INVALID_NODE]
                exclude.update(valid_walk_neighbors.tolist())
        exclude_sets.append(exclude)
    # Sample negatives while avoiding exclusions
    neg_nodes = torch.zeros((batch_size, neg_samples), dtype=torch.long)
    for i in range(batch_size):
        valid_nodes = [n for n in range(n_nodes) if n not in exclude_sets[i]]
        if len(valid_nodes) >= neg_samples:
            sampled = torch.tensor(RNG.choice(valid_nodes, neg_samples, replace=False))
        else:
            # Fallback: sample with replacement if not enough valid nodes
            sampled = torch.tensor(RNG.choice(valid_nodes, neg_samples, replace=True))
        neg_nodes[i] = sampled
    assert neg_nodes.shape == shape
    return neg_nodes

@trace
def cpu_neg_pair_generator(n_nodes: int,
                           anchors: Tensor,
                           pos_nodes: Tensor,
                           edge_index: nparray2d,
                           shape: tuple[int, int],
                           walks: Tensor|None = None) -> Tensor:
    """Vectorized CPU-based negative sampling to reduce GPU memory usage.

    This version uses vectorized numpy operations for better performance
    compared to the previous loop-based approach.

    Args:
    - n_nodes: total number of nodes in graph
    - anchors: Current batch anchor nodes to exclude
    - pos_nodes: Current batch positive nodes to exclude
    - edge_index: Graph edges to identify actual neighbors to exclude
    - shape: Target shape (batch_size, neg_samples_factor)
    - walks: Optional tensor of current walks to exclude walk neighbors

    Returns:
    - Tensor of negative node indices with shape `shape`
    """
    batch_size, neg_samples = shape

    # Convert to numpy for CPU processing
    anchors_np = anchors.cpu().numpy()
    pos_nodes_np = pos_nodes.cpu().numpy()
    walks_np = walks.cpu().numpy() if walks is not None else None

    # Build global exclusion matrix (batch_size x n_nodes boolean mask)
    exclude_mask = np.zeros((batch_size, n_nodes), dtype=bool)

    # Exclude anchors and positives (vectorized)
    exclude_mask[np.arange(batch_size), anchors_np] = True
    exclude_mask[np.arange(batch_size), pos_nodes_np] = True

    # Exclude neighbors (vectorized where possible)
    for i, anchor in enumerate(anchors_np):
        neighbors = edge_index[1][edge_index[0] == anchor]
        if len(neighbors) > 0:
            exclude_mask[i, neighbors] = True

    # Exclude walk neighbors if provided
    if walks_np is not None:
        for i, anchor in enumerate(anchors_np):
            walk_mask = (walks_np == anchor).any(axis=1)
            if walk_mask.any():
                walk_neighbors = walks_np[walk_mask].flatten()
                valid_walk_neighbors = walk_neighbors[walk_neighbors != INVALID_NODE]
                if len(valid_walk_neighbors) > 0:
                    exclude_mask[i, valid_walk_neighbors] = True

    # Sample negatives for all anchors (vectorized where possible)
    neg_nodes_np = np.zeros((batch_size, neg_samples), dtype=np.int64)
    for i in range(batch_size):
        valid_indices = np.where(~exclude_mask[i])[0]
        if len(valid_indices) >= neg_samples:
            neg_nodes_np[i] = RNG.choice(valid_indices, neg_samples, replace=False)
        else:
            # Fallback: sample with replacement if not enough valid nodes
            neg_nodes_np[i] = RNG.choice(valid_indices, neg_samples, replace=True)

    # Convert back to torch tensor only at the end
    neg_nodes = torch.from_numpy(neg_nodes_np).long()
    assert neg_nodes.shape == shape
    return neg_nodes
