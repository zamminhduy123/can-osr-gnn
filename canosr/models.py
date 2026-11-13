"""Encoders, heads, and the autoencoder that ties them together."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_batch

from .registry import build, register


# ----------------------------------------------------------------------- encoders


class _MessagePassingEncoder(nn.Module):
    conv: type
    uses_edge_attr: bool = True

    def __init__(
        self,
        in_dim: int,
        hidden: int = 8,
        heads: int = 5,
        layers: int = 2,
        dropout: float = 0.3,
        edge_dim: int = 1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        width = hidden * heads
        for index in range(layers):
            source = in_dim if index == 0 else width
            kwargs: dict[str, Any] = dict(heads=heads, concat=True)
            if self.uses_edge_attr:
                kwargs["edge_dim"] = edge_dim
            self.layers.append(self.conv(source, hidden, **kwargs))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        accumulated = None
        for conv in self.layers:
            if self.uses_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            accumulated = x if accumulated is None else accumulated + x
        return accumulated / len(self.layers)


@register("encoder", "gatv2")
class GATv2Encoder(_MessagePassingEncoder):
    conv = GATv2Conv


@register("encoder", "gat")
class GATEncoder(_MessagePassingEncoder):
    conv = GATConv


@register("encoder", "sage")
class SAGEEncoder(_MessagePassingEncoder):
    conv = SAGEConv
    uses_edge_attr = False


@register("encoder", "gcn")
class GCNEncoder(_MessagePassingEncoder):
    conv = GCNConv
    uses_edge_attr = False


@register("encoder", "mlp")
class MLPEncoder(nn.Module):
    """No message passing. Baseline for 'is the graph doing anything'."""

    def __init__(self, in_dim: int, hidden: int = 8, heads: int = 5, layers: int = 2, dropout: float = 0.3, **_: Any):
        super().__init__()
        width = hidden * heads
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        return self.net(x)


# -------------------------------------------------------------------------- readout


class AttnReadout(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, batch):
        weights = torch.sigmoid(self.attn(x))
        return global_add_pool(x * weights, batch)


# ---------------------------------------------------------------------------- heads


class PrototypeHead(nn.Module):
    def __init__(self, n_classes: int, dim: int, **_: Any):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.register_buffer("prototypes", torch.zeros(n_classes, dim))

    @torch.no_grad()
    def set_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        for index in range(self.n_classes):
            mask = labels == index
            if mask.any():
                self.prototypes[index] = F.normalize(embeddings[mask].mean(0), dim=0)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return -torch.cdist(embeddings, self.prototypes)


@register("head", "prototype")
def _prototype(**kwargs: Any) -> PrototypeHead:
    return PrototypeHead(**kwargs)


@register("head", "linear")
class LinearHead(nn.Module):
    def __init__(self, n_classes: int, dim: int, **_: Any):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(embeddings)


@register("head", "none")
class IdentityHead(nn.Module):
    def __init__(self, **_: Any):
        super().__init__()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings


# --------------------------------------------------------------------------- model


class GraphAE(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, cfg: Any, edge_dim: int = 3):
        super().__init__()
        self.cfg = cfg
        self.encoder = build(
            "encoder",
            cfg.encoder,
            in_dim=in_dim,
            hidden=cfg.hidden,
            heads=cfg.heads,
            layers=cfg.layers,
            dropout=cfg.dropout,
            edge_dim=edge_dim,
        )
        width = cfg.hidden * cfg.heads
        self.readout = AttnReadout(width)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(width * 2, cfg.projection),
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, in_dim)
        )
        self.head = build("head", cfg.head, n_classes=n_classes, dim=cfg.projection)
        self.id_in_recon = bool(cfg.id_in_recon)

    def encode(self, data):
        return self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)

    def embed(self, h, batch):
        return F.normalize(self.projection(self.readout(h, batch)), dim=1)

    def node_reconstruction(self, h):
        return self.node_decoder(h)

    def adjacency_logits(self, h, batch, max_nodes: int | None = None):
        dense, mask = to_dense_batch(h, batch, max_num_nodes=max_nodes)
        return torch.einsum("bid,bjd->bij", dense, dense), mask

    def forward(self, data):
        h = self.encode(data)
        z = self.embed(h, data.batch)
        return {
            "h": h,
            "z": z,
            "node_hat": self.node_reconstruction(h),
            "adj_logits": self.adjacency_logits(h, data.batch)[0],
            "logits": self.head(z),
        }

    def trainable_parameters(self, phase: int):
        if phase == 1:
            return [p for p in self.parameters() if p.requires_grad]
        return [
            p
            for name, p in self.named_parameters()
            if p.requires_grad and (name.startswith("projection") or name.startswith("readout") or name.startswith("head"))
        ]

    def freeze_for_phase2(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith(("projection", "readout", "head"))
