"""Graph learning tools.

Implementation of Graph Attention Networks (GAT) using PyTorch and PyTorch Geometric.
"""

from __future__ import annotations

import functools
import gc
import logging
import os
import sys
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
import torch.nn as nn
import torch.nn.functional as F
import yaml

from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC, SVC # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer, RobustScaler, StandardScaler # type: ignore
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
from nkpylib.ml.graph_worker import initialize_worker, WorkItem, EdgeSampler
from nkpylib.ml.ml_utils import trace, list_gpu_tensors
from nkpylib.script_utils import YamlConfigManager

CFG: Namespace|None = None
RNG = npr.default_rng(0)
torch.manual_seed(0)

logger = logging.getLogger(__name__)

# default batch size
BATCH_SIZE = 128

# default temperature
TEMPERATURE = 0.15 # was 0.07 initially

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
        # Store the original config for serialization
        self.model_config = {
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'heads': heads,
            'dropout': dropout,
            **kw
        }

        ModelCls = GATv2Conv if kw.get('v2', False) else GATConv
        logger.info(f'Initializing model {ModelCls}')
        self.conv1 = ModelCls(in_channels, hidden_channels, heads=heads)
        #self.conv2 = ModelCls(hidden_channels * heads, out_channels, heads=1, concat=False) # concat=False for final layer
        self.conv2 = ModelCls(hidden_channels * heads, hidden_channels * heads, heads=1)
        self.dropout = dropout
        self.process = psutil.Process()

    #@trace
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

    def get_embeddings(self, x, edge_index, use_half=False, device=None):
        """Extract node embeddings from both GAT layers.

        Returns tensor of shape [num_nodes, hidden_channels * heads * 2]
        """
        self.eval()
        with torch.no_grad():
            if use_half:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    ret = self.embedding_forward(x, edge_index)
            else:
                ret = self.embedding_forward(x, edge_index)
        # l2 normalize
        ret = F.normalize(ret, p=2, dim=1)
        return ret

    def get_config(self) -> dict:
        """Get the model configuration for serialization."""
        return self.model_config.copy()

    def log_memory(self, msg):
        mem = self.process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB
        #print(f"{msg}: {mem:.2f}GB")


    @classmethod
    @abstractmethod
    def worker_one_step(cls, batch_size: int) -> WorkItem:
        """Worker function to generate a single work item for training."""
        pass

    #@trace
    @abstractmethod
    def compute_loss(self,
                     item: WorkItem,
                     x: Tensor,
                     gpu_batch_size: int = BATCH_SIZE,
                     use_checkpoint: bool = False):
        """Compute loss for a given work item."""
        pass


class NodeClassificationGAT(GATBase):
    """GAT model for node classification tasks.

    """
    def __init__(self, hidden_channels: int, heads: int, task_config: dict, **kw):
        """Initialize this model with a `task_config` defining classification tasks.

        The `task_config` is a dictionary mapping task names to one or more node classification
        tasks, each containing at least:
        - n_classes: Number of classes for this task
        - type: 'multiclass' or 'multilabel'
        - weight: Weight for this task in the overall loss [optional, default=1]

        This adds a linear layer to the base model for each task, which generates the final output.
        During training, the model computes cross-entropy loss for each task and sums them up,
        weighted by the specified weights.

        All other `kw` are passed to the base GAT model.
        """
        super().__init__(hidden_channels=hidden_channels, heads=heads, **kw)
        # Add task_config to the stored config
        self.model_config['task_config'] = task_config
        self.task_config = task_config
        self.task_heads = nn.ModuleDict()
        input_dims = hidden_channels * heads
        for task_name, config in task_config.items():
            self.task_heads[task_name] = torch.nn.Linear(input_dims, config['n_classes'])

    def forward(self, x, edge_index) -> dict[str, Tensor]:
        """Runs the base model and then the final linear layers (one per task).

        Returns a dictionary mapping task names to output tensors. Each output tensor has length
        [num_nodes, n_classes] for that task.
        """
        x = super().forward(x, edge_index)
        x = F.elu(x)
        # apply each task head
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(x)
        return outputs

    @classmethod
    def worker_one_step(cls, batch_size: int) -> WorkItem:
        """Worker function to generate a single work item for training."""
        from nkpylib.ml.graph_worker import node_classification_one_step
        return node_classification_one_step(batch_size)

    def top_k_loss(self, logits: Tensor, targets: Tensor, k: int = 5) -> Tensor:
        """Custom loss for top-k multilabel prediction.

        Args:
        - logits: Model outputs [batch_size, num_classes]
        - targets: Multi-hot encoded targets [batch_size, num_classes]
        - k: Number of top predictions to consider

        Returns:
        - Binary cross-entropy loss computed only on top-k predictions
        """
        # Get top-k predictions
        _, top_k_indices = torch.topk(logits, k, dim=1)

        # Create top-k mask
        top_k_mask = torch.zeros_like(logits)
        top_k_mask.scatter_(1, top_k_indices, 1.0)

        # Only compute loss on top-k predictions
        masked_logits = logits * top_k_mask
        masked_targets = targets * top_k_mask
        return F.binary_cross_entropy_with_logits(masked_logits, masked_targets)

    def compute_loss(self, item: WorkItem, x: Tensor, gpu_batch_size: int = BATCH_SIZE) -> Tensor:
        """Compute node classification loss for multiple tasks.

        Args:
        - item: WorkItem containing labels and train_masks dicts
        - x: Node features tensor
        - gpu_batch_size: Batch size for GPU processing (unused for node classification)

        Returns:
        - Weighted sum of losses across all tasks
        """
        # Get current edges and compute outputs
        cur_edges = torch.tensor(item.cur_edges).to(x.device)
        logger.info(f'Got edges of shape {cur_edges.shape} [{cur_edges.dtype}]')
        outputs = self(x, cur_edges)
        total_loss = 0.0
        # Compute loss for each task
        for task_name, task_config in self.task_config.items():
            if task_name not in item.labels or task_name not in item.train_masks:
                logger.warning(f'Task {task_name} missing from WorkItem, skipping')
                continue
            task_output = outputs[task_name]
            train_mask = item.train_masks[task_name]
            targets = item.labels[task_name]
            task_weight = task_config.get('weight', 1.0)
            masked_output = task_output[train_mask]
            masked_targets = targets[train_mask]
            if len(masked_targets) == 0:
                logger.warning(f'No training samples for task {task_name}, skipping')
                continue
            # Compute task-specific loss
            match task_config['type']:
                case 'multiclass':
                    task_loss = F.cross_entropy(masked_output, masked_targets)
                case 'multilabel':
                    task_loss = F.binary_cross_entropy_with_logits(masked_output, masked_targets.float())
                case 'top_k':
                    k = task_config.get('k', 5)
                    task_loss = self.top_k_loss(masked_output, masked_targets, k=k)
                case _:
                    raise ValueError(f"Unknown task type: {task_config['type']} for task {task_name}")
            logger.debug(f'Task {task_name}: loss={task_loss:.4f}, weight={task_weight}')
            total_loss += task_weight * task_loss
        return total_loss


class ContrastiveGAT(GATBase):
    """GAT model trained using contrastive learning.

    This class implements the core contrastive learning logic, processing pairs
    of positive and negative examples to learn node embeddings.
    """
    def __init__(self,
                 temperature: float = TEMPERATURE,
                 **kw):
        """Initialize this ContrastiveGAT model.

        Args:
            - temperature: Temperature for similarity scaling (higher = softer attention)
        """
        super().__init__(**kw)
        # Add temperature to the stored config
        self.model_config['temperature'] = temperature
        self.temperature = temperature

    @classmethod
    def worker_one_step(cls, batch_size: int) -> WorkItem:
        """Worker function to generate a single work item for training."""
        from nkpylib.ml.graph_worker import contrastive_worker_one_step
        return contrastive_worker_one_step(batch_size)

    #@trace
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

    #@trace
    def compute_loss(self,
                     item: WorkItem,
                     x: Tensor,
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
        logger.debug(f'Computing contrastive loss with batch size {gpu_batch_size}, use_checkpoint {use_checkpoint}')
        # Get embeddings
        cur_edges = torch.tensor(item.cur_edges).to(x.device)
        logger.debug(f'Got edges of shape {cur_edges.shape} [{cur_edges.dtype}]')
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
            initargs = dict(
                n_nodes=data.num_nodes,
                edge_index=data.edge_index.to('cpu').numpy(),
                walk_length=walk_length,
                max_edges_per_node=sample_edges,
                walk_window=walk_window,
                neg_samples_factor=neg_samples_factor,
                task_config=kw.get('task_config', {}),
                keys=data['keys'],
            )
            if getattr(data, 'user_pos', None) is not None:
                initargs['user_pos'] = data.user_pos
            if getattr(data, 'user_neg', None) is not None:
                initargs['user_neg'] = data.user_neg
            self.pool = ProcessPoolExecutor(max_workers=n_jobs, initializer=initialize_worker, initargs=(initargs,))

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
            logger.debug(f'Work item: {[(x.shape, x.dtype) for x in (item.anchors, item.pos_nodes, item.neg_nodes)]}, q {self.queue.qsize()}')
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
                #pbar.set_description(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Memory (MB): {mem_s}')
                pbar.set_description(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
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

    def train_node_classification(self, dataset, n_epochs:int=100) -> tuple[GATBase, Tensor]:
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


    def train_contrastive(self,
                          n_epochs: int = 200,
                          gpu_batch_size: int = BATCH_SIZE,
                          temperature: float = TEMPERATURE,
                          existing_model: torch.nn.Module|None=None) -> tuple[GATBase, Tensor]:
        """Train a graph model using contrastive learning with direct neighbor sampling.

        This uses the direct neighbor approach where positive pairs are directly connected nodes
        and negative pairs are randomly sampled nodes that avoid direct neighbors.

        Args:
        - n_epochs: Number of training epochs
        - gpu_batch_size: Batch size for GPU processing
        - temperature: Temperature parameter for contrastive loss
        - existing_model: If provided, use this pre-initialized model instead of creating a new one

        Returns:
        - Tuple of (trained_model, loss_history)
        """
        if existing_model is not None:
            model = existing_model
        else:
            model = ContrastiveGAT(
                in_channels=self.data.num_features,
                hidden_channels=self.hidden_channels,
                heads=self.heads,
                dropout=self.dropout,
                v2=self.v2,
                temperature=temperature,
            )

        def loss_fn(model):
            # This is a placeholder - actual loss computation happens in train_model via async workers
            raise NotImplementedError("Loss computation handled by async workers")

        losses = self.train_model(model, loss_fn, n_epochs=n_epochs, gpu_batch_size=gpu_batch_size)
        return model, losses

    def train_from_config(self, cfg) -> tuple[GATBase, Tensor]:
        """Train a model based on the provided configuration.

        This method handles both fresh training and resuming from checkpoints.

        Args:
        - cfg: NestedNamespace configuration object

        Returns:
        - Tuple of (trained_model, loss_history)
        """
        # Check for resume
        if cfg.io.resume:
            logger.info(f'Resuming training from checkpoint: {cfg.io.resume}')
            model, losses = resume_from_checkpoint(
                checkpoint_path=cfg.io.resume,
                data_path=cfg.io.input_path,
                additional_epochs=cfg.train.n_epochs,
            )
            return model, losses
        # Fresh training based on learner type
        logger.info(f"Training {cfg.model.learner_type} model for {cfg.train.n_epochs} epochs")
        kw = dict(n_epochs=cfg.train.n_epochs, gpu_batch_size=cfg.train.gpu_batch_size)
        match cfg.model.learner_type:
            case 'random_walk':
                model, losses = self.train_random_walks(walk_length=cfg.model.walk_length, **kw)
            case 'contrastive':
                model, losses = self.train_contrastive(**kw)
            case _:
                raise NotImplementedError(f"Learner type {cfg.model.learner_type} not implemented")
        return model, losses

    def train_random_walks(self, walk_length: int, n_epochs=5, gpu_batch_size:int=BATCH_SIZE) -> tuple[GATBase, Tensor]:
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
                    use_full: bool=False,
                    **kwargs,
                    ):
    """Extract learned embeddings and save to NumpyLmdb.

    Args:
    - model: Trained GAT model
    - data: PyG Data object with original keys
    - output_path: Path to output NumpyLmdb (and .lmdb -> _model.pt for the model itself)
    - output_flag: LMDB flag for opening
    - use_full: whether to use the full graph edges or a sampler
    - kwargs: Additional metadata to save in the database
    """
    #sys.exit()#FIXME
    # first clear some memory
    logger.info(f'Clearing some memory')
    torch.cuda.empty_cache()
    gc.collect()
    if use_full:
        edges = data.edge_index
    else:
        edge_sampler = EdgeSampler(
            edge_index=data.edge_index.numpy(),
            max_edges_per_node=10,
            global_sampling=True,
        )
        edges = torch.tensor(edge_sampler.sample())
        logger.info(f'Sampled from {data.edge_index.shape} to {edges.shape}')
    # Extract embeddings (using cpu mode, since we're using data which might overflow gpu)
    cur_device = 'cpu'
    logger.info(f'Moving model to {cur_device} to extract embeddings')
    model = model.to(cur_device)
    logger.info(f'Trying to output embeddings now')
    embeddings = model.get_embeddings(data.x.to(cur_device), edges.to(cur_device), device=cur_device).cpu().numpy()
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


def save_model_with_checkpoint(model: torch.nn.Module,
                              data: Data,
                              output_path: str,
                              training_args: dict,
                              losses: Tensor,
                              epoch: int = None,
                              **kwargs):
    """Save model state dict and config to avoid pickle issues.

    Args:
    - model: Trained model
    - data: Training data
    - output_path: Base path for saving (will create .pt file)
    - training_args: Dictionary of training arguments used
    - losses: Loss history from training
    - epoch: Current epoch (for partial training saves)
    - kwargs: Additional metadata
    """
    model_output_path = output_path.replace('.lmdb', '-model.pt')

    # Debug model before saving
    param_count = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    logger.info(f"Saving model with {param_count:,} parameters ({param_size / 1024 / 1024:.2f} MB)")

    # Get model configuration automatically
    model_config = model.get_config()
    logger.info(f"Saving model config: {model_config}")

    # Save state dict and reconstruction info (no full model to avoid pickle issues)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': model_config,
        'training_args': training_args,
        'losses': losses,
        'epoch': epoch or len(losses),
        'data_info': {
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'num_features': data.num_features,
        },
        'timestamp': time.time(),
        **kwargs
    }

    torch.save(checkpoint, model_output_path)

    # Verify the save worked
    file_size = os.path.getsize(model_output_path)
    logger.info(f'Saved model checkpoint to {model_output_path} ({file_size:,} bytes)')
    if file_size < 10000:  # Less than 10KB is suspicious
        logger.warning("WARNING: Model checkpoint file is very small - may indicate save failure!")

def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> dict:
    """Load checkpoint and reconstruct model from state dict.

    Args:
    - checkpoint_path: Path to checkpoint file
    - device: Device to load the model on

    Returns:
    - Dictionary containing reconstructed model, training args, losses, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f'saved config: {checkpoint["model_config"]}')

    # Verify checkpoint structure
    required_keys = ['model_state_dict', 'model_class', 'model_config', 'training_args', 'losses']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        logger.warning(f"Checkpoint missing keys: {missing_keys}")

    # Reconstruct model from saved config
    model_class = checkpoint['model_class']
    model_config = checkpoint['model_config']
    if model_class == 'ContrastiveGAT':
        model = ContrastiveGAT(**model_config)
    elif model_class == 'NodeClassificationGAT':
        model = NodeClassificationGAT(**model_config)
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.debug(f'Loaded state dict from {checkpoint["model_state_dict"]}')
    model = model.to(device)
    # Add reconstructed model to checkpoint
    checkpoint['model'] = model
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Model type: {model_class}")
    logger.info(f"Previous epochs: {checkpoint.get('epoch', 'Unknown')}")
    logger.info(f"Loss history length: {len(checkpoint.get('losses', []))}")
    return checkpoint


def resume_from_checkpoint(checkpoint_path: str,
                          data_path: str,
                          additional_epochs: int = 100,
                          **override_kwargs) -> tuple[torch.nn.Module, Tensor]:
    """Resume training from a comprehensive checkpoint.

    Args:
    - checkpoint_path: Path to checkpoint file
    - data_path: Path to training data
    - additional_epochs: Additional epochs to train
    - **override_kwargs: Override any training arguments from checkpoint

    Returns:
    - Tuple of (model, combined_loss_history)
    """
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model = checkpoint['model']
    previous_losses = checkpoint['losses']
    training_args = checkpoint['training_args'].copy()
    last_epoch = checkpoint.get('epoch', len(previous_losses))

    # Override training args if provided
    training_args.update(override_kwargs)

    logger.info(f"Resuming from epoch {last_epoch} with {len(previous_losses)} previous losses")

    # Load data
    data = torch.load(data_path, weights_only=False)

    # Create GraphLearner with training args
    gl = GraphLearner(data, **training_args)

    # Resume training based on model type
    if isinstance(model, ContrastiveGAT):
        model, new_losses = gl.train_contrastive(
            n_epochs=additional_epochs,
            gpu_batch_size=training_args.get('gpu_batch_size', 256),
            existing_model=model
        )
    elif isinstance(model, NodeClassificationGAT):
        raise NotImplementedError("Node classification resume not fully implemented")
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    # Combine loss histories (move both to cpu first)
    previous_losses = torch.tensor([l for l in previous_losses]).cpu()
    new_losses = torch.tensor(new_losses).cpu()
    combined_losses = torch.cat([previous_losses, new_losses])

    return model, combined_losses


def load_and_prepare_data(cfg) -> Data:
    """Load and prepare graph data based on configuration.
    
    This handles:
    - Loading the PyG data from file
    - Optional feature replacement with random values
    - Feature scaling based on config
    - Edge index type conversion
    - Node sampling if specified
    - Optional duplicate edge analysis
    - Optional edge analysis for specific nodes
    
    Args:
    - cfg: NestedNamespace configuration object
    
    Returns:
    - Prepared PyG Data object
    """
    # Load input graph
    data = torch.load(cfg.io.input_path, weights_only=False)
    assert data.num_nodes < 2**31, "Number of nodes exceeds int32 range"
    
    if 0:
        # Replace node features with random ones
        data.x = torch.randn(data.x.shape, dtype=torch.float32) #FIXME
    
    # Apply feature scaling based on config
    match cfg.model.scaler:
        case 'std_mean':
            scaler = StandardScaler(with_mean=True, with_std=False)
        case 'std_std':
            scaler = StandardScaler(with_mean=False, with_std=True)
        case 'std_both':
            scaler = StandardScaler(with_mean=True, with_std=True)
        case 'robust':
            scaler = RobustScaler()
        case 'maxabs':
            scaler = MaxAbsScaler()
        case 'quantile':
            scaler = QuantileTransformer(output_distribution='uniform')
        case 'quantile_normal':
            scaler = QuantileTransformer(output_distribution='normal')
        case _:
            raise ValueError(f"Unknown scaler type: {cfg.model.scaler}")
    
    logger.info(f'Applying {cfg.model.scaler} feature scaling')
    data.x = torch.tensor(scaler.fit_transform(data.x.cpu()), dtype=torch.float32)
    data.edge_index = data.edge_index.to(torch.int32)
    
    logger.info(f'Loaded PyG from {cfg.io.input_path} with {data.num_nodes}x{data.num_features} nodes, {data.num_edges} edges, {data.x.dtype}, {data.edge_index.dtype}')
    logger.info(f'All data keys: {data.keys()}')
    
    if 0:
        # Check for duplicates in the edge index
        dupes = Counter()
        for a, b in data.edge_index.t().tolist():
            dupes[(a, b)] += 1
        num_dupes = sum(count - 1 for count in dupes.values() if count > 1)
        logger.info(f'Found {num_dupes} duplicate edges in edge_index: {dupes.most_common(5)}')
        sys.exit()
    
    # Sample nodes if specified
    if cfg.model.n_nodes and data.num_nodes > cfg.model.n_nodes:
        data.x = data.x[:cfg.model.n_nodes]
        data.edge_index = data.edge_index[:, (data.edge_index[0] < cfg.model.n_nodes) & (data.edge_index[1] < cfg.model.n_nodes)]
        logger.info(f'Sampled to {data.num_nodes} nodes, {data.num_edges} edges')
    
    if 0:
        # Print all pairs in edge_index where one of them is a given idx
        print(data.edge_index)
        for idx in [1, 9739, 764, 55542]:
            indices = torch.where(data.edge_index == idx)
            pairs = data.edge_index[:, indices[1]].T
            print(f'{len(pairs)} Edges involving node {idx}: {pairs.T}')
        #return
    
    return data


def main():
    """Sets up the configuration, loads the data, trains the model, and saves the results."""
    global CFG
    with YamlConfigManager() as config_mgr:
        io = config_mgr.add_parser('io', description='Input/Output Configuration')
        io.add_argument('input_path', help='Input PyG.Data path (as .pt)')
        io.add_argument('output_path', help='Output NumpyLmdb path for learned embeddings')
        io.add_argument('-f', '--output-flag', default='c', choices=['c', 'w', 'n'], help='LMDB flag for output [c]')
        io.add_argument('--resume', help='Path to model checkpoint to resume training from')
        model = config_mgr.add_parser('model', description='Model Configuration')
        model.add_argument('-t', '--learner-type', default='contrastive', choices=LEARNERS, help='GAT learner [contrastive]')
        model.add_argument('-n', '--n-nodes', type=int, default=5000000, help='Number of nodes to sample from feature set')
        model.add_argument('-w', '--walk-length', type=int, default=12, help='Length of random walks [12]')
        model.add_argument('--walk-window', type=int, default=5, help='Context window for walks [5]')
        model.add_argument('--scaler', default='quantile_normal',
                          choices=['std_mean', 'std_std', 'std_both', 'robust', 'maxabs', 'quantile', 'quantile_normal'],
                          help='Feature scaling method [quantile_normal]')
        arch = config_mgr.add_parser('arch', description='Model Architecture Parameters')
        arch.add_argument('--hidden-channels', type=int, default=48, help='Hidden channels in GAT layers [64]')
        arch.add_argument('-H', '--heads', type=int, default=4, help='Number of attention heads [8]')
        arch.add_argument('-d', '--dropout', type=float, default=0.6, help='Training dropout rate [0.6]')
        train = config_mgr.add_parser('train', description='Training Parameters')
        train.add_argument('-e', '--n-epochs', type=int, default=500, help='Number of training epochs [500]')
        #NOTE set the following to 128 for my home cpu
        # tune gpu batch size to max first
        train.add_argument('--gpu-batch-size', type=int, default=256, help=f'Batch size for GPU [{BATCH_SIZE}]')
        # in general, gpu batch size should be a multiple of cpu
        train.add_argument('--cpu-batch-size', type=int, default=256, help=f'Batch size for CPU [{BATCH_SIZE}]')
        train.add_argument('-j', '--n_jobs', type=int, default=2, help='Number of parallel jobs [6]')
    
    CFG = config_mgr.parse_all()
    print(f'Final config: {CFG}')
    
    # Load and prepare data
    data = load_and_prepare_data(CFG)
    
    # Create learner
    gl = create_learner(
        CFG.model.learner_type,
        data,
        hidden_channels=CFG.arch.hidden_channels,
        heads=CFG.arch.heads,
        dropout=CFG.arch.dropout,
        n_jobs=CFG.train.n_jobs,
        cpu_batch_size=CFG.train.cpu_batch_size,
        v2=False,
    )
    
    # Train model (handles both fresh training and resume)
    model, losses = gl.train_from_config(CFG)
    
    # Save embeddings and model checkpoint
    config_dict = CFG.to_flat_dict()
    save_embeddings(model, data, CFG.io.output_path, CFG.io.output_flag, **config_dict)
    save_model_with_checkpoint(model, data, CFG.io.output_path, config_dict, losses, **config_dict)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s', level=logging.INFO)
    logger.info(f'Got device {device}')
    main()
