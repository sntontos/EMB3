"""GIN through PyTorch Geometric (main environment). Supervised, trained end to end.

The network follows PyTorch Geometric's reference example (examples/mutag_gin.py): a stack
of GINConv layers, each with a two-layer MLP and batch normalisation, a pooling readout,
and an MLP classifier. The pooled vector is the graph embedding, so the hidden width is the
embedding dimension.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GINConv, global_add_pool, global_mean_pool

from emb3.graphs import GraphDataset
from emb3.measure import Meter, thread_budget

# Default training settings; any of them can be overridden per call to fit().
# The reference example uses a learning rate of 0.01. With it the training loss on ENZYMES
# stayed at chance level, so the default here is 0.001.
DEFAULTS: Dict[str, Any] = {
    "num_layers": 5,
    "pool": "sum",
    "dropout": 0.5,
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 100,
    "patience": None,   # epochs without a better validation accuracy before stopping; None = never
}

# When the label vocabulary (largest possible label + 1) reaches this size, the label is fed
# as one standardised number instead of a one-hot vector. PyTorch Geometric's benchmark
# suite draws the same line for degree features (benchmark/kernel/datasets.py: one-hot
# below a maximum degree of 1000). Of our datasets only REDDIT-MULTI-12K exceeds it.
ONE_HOT_LIMIT = 1000


class NodeInput(torch.nn.Module):
    """Turns integer node labels into the network's input features."""

    def __init__(self, num_labels: int, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.one_hot = num_labels < ONE_HOT_LIMIT
        self.num_labels = num_labels
        self.out_channels = num_labels if self.one_hot else 1
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        if self.one_hot:
            # Labels beyond the vocabulary (possible after perturbing edges) share the last slot.
            return F.one_hot(labels.view(-1).clamp(max=self.num_labels - 1), self.num_labels).float()
        return (labels.float().view(-1, 1) - self.mean) / self.std


class GINClassifier(torch.nn.Module):
    def __init__(self, node_input: NodeInput, hidden: int, num_layers: int, num_classes: int, pool: str, dropout: float) -> None:
        super().__init__()
        self.node_input = node_input
        self.convs = torch.nn.ModuleList()
        in_channels = node_input.out_channels
        for _ in range(num_layers):
            self.convs.append(GINConv(MLP([in_channels, hidden, hidden]), train_eps=False))
            in_channels = hidden
        self.pool = {"sum": global_add_pool, "mean": global_mean_pool}[pool]
        self.classifier = MLP([hidden, hidden, num_classes], norm=None, dropout=dropout)

    def embed(self, batch: Any) -> torch.Tensor:
        x = self.node_input(batch.x)
        for conv in self.convs:
            x = conv(x, batch.edge_index).relu()
        return self.pool(x, batch.batch)

    def forward(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embed(batch)
        return self.classifier(embedding), embedding


class TrainedGIN:
    """A fitted model. Predicts class probabilities and produces graph embeddings."""

    def __init__(self, model: GINClassifier, batch_size: int) -> None:
        self.model = model
        self.batch_size = batch_size

    @torch.no_grad()
    def _run(self, graphs: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        logits, embeddings = [], []
        for batch in DataLoader(list(graphs), batch_size=self.batch_size, shuffle=False):
            out, embedding = self.model(batch)
            logits.append(out)
            embeddings.append(embedding)
        return torch.cat(logits), torch.cat(embeddings)

    def predict_proba(self, graphs: Sequence[Any]) -> np.ndarray:
        return torch.softmax(self._run(graphs)[0], dim=1).numpy()

    def embed(self, graphs: Sequence[Any]) -> np.ndarray:
        return self._run(graphs)[1].numpy().astype(np.float32)


def _input_size(dataset: GraphDataset) -> int:
    # A node's degree is below its graph's size, so with degree labels the largest graph
    # bounds the vocabulary even after edges are added.
    if dataset.label_kind == "degree":
        return max(dataset.num_node_labels, int(dataset.num_nodes.max()))
    return dataset.num_node_labels


def fit(
    dataset: GraphDataset,
    graphs: Sequence[Any],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    dimension: int,
    seed: int,
    **overrides: Any,
) -> Tuple[TrainedGIN, Dict[str, Any]]:
    """Trains on train_index and keeps the weights of the epoch with the best validation accuracy.

    `graphs` is the result of emb3.convert.to_pyg(dataset). Node-feature statistics come
    from the training graphs only.
    """
    params = dict(DEFAULTS, **overrides)
    torch.set_num_threads(thread_budget())
    torch.manual_seed(seed)

    train_graphs = [graphs[i] for i in train_index]
    validation_graphs = [graphs[i] for i in validation_index]
    train_labels = torch.cat([graph.x.view(-1) for graph in train_graphs]).float()
    node_input = NodeInput(_input_size(dataset), mean=train_labels.mean().item(), std=max(train_labels.std().item(), 1e-12))
    model = GINClassifier(node_input, dimension, params["num_layers"], dataset.num_classes, params["pool"], params["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    shuffle = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_graphs, batch_size=params["batch_size"], shuffle=True, generator=shuffle)
    trained = TrainedGIN(model, batch_size=params["batch_size"])
    validation_y = torch.cat([graph.y for graph in validation_graphs])

    best = {"accuracy": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    history: List[Dict[str, float]] = []
    with Meter() as train:
        for epoch in range(1, params["max_epochs"] + 1):
            model.train()
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(batch)[0], batch.y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * batch.num_graphs
            logits, _ = trained._run(validation_graphs)
            accuracy = float((logits.argmax(dim=1) == validation_y).float().mean())
            validation_loss = float(F.cross_entropy(logits, validation_y))
            history.append({"train_loss": total_loss / len(train_graphs), "validation_loss": validation_loss, "validation_accuracy": accuracy})
            if (accuracy, -validation_loss) > (best["accuracy"], -best["loss"]):
                best = {"accuracy": accuracy, "loss": validation_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            elif params["patience"] is not None and epoch - best["epoch"] >= params["patience"]:
                break
    model.load_state_dict(best["state"])

    record = {
        "method": "gin",
        "params": dict(params, dimension=dimension, seed=seed, node_input="one-hot" if node_input.one_hot else "standardised"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "best_validation_accuracy": best["accuracy"],
        "history": history,
        "train": train.as_dict(),
    }
    return trained, record
