"""Losses. Every one is used; no unreachable branches."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .registry import register


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020), one view."""
    with torch.cuda.amp.autocast(enabled=False):
        features = F.normalize(features.float(), dim=1)
        similarity = features @ features.t() / temperature
        n = features.shape[0]
        self_mask = torch.eye(n, dtype=torch.bool, device=features.device)
        positives = (labels[:, None] == labels[None, :]) & ~self_mask

        similarity = similarity.masked_fill(self_mask, -1e9)
        log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
        counts = positives.sum(dim=1)
        valid = counts > 0
        if not valid.any():
            return features.sum() * 0.0
        per_anchor = -(log_prob * positives).sum(dim=1) / counts.clamp_min(1)
        return per_anchor[valid].mean()


def adjacency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor | None = None,
    graph_weight: torch.Tensor | None = None,
    pos_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Binary BCE on the dense adjacency. Target is ``edge exists``, not edge count."""
    binary = (target > 0).float()
    loss = F.binary_cross_entropy_with_logits(logits, binary, pos_weight=pos_weight, reduction="none")
    if node_mask is not None:
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        loss = loss * pair_mask
        per_graph = loss.sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1)
    else:
        per_graph = loss.mean(dim=(1, 2))
    if graph_weight is not None:
        return (per_graph * graph_weight).sum() / graph_weight.sum().clamp_min(1)
    return per_graph.mean()


def node_recon_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor | None = None,
    graph_weight: torch.Tensor | None = None,
    skip_first: bool = False,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    if skip_first:
        prediction, target = prediction[:, 1:], target[:, 1:]
    error = ((prediction - target) ** 2).mean(dim=1)
    if batch is not None:
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        sums = torch.zeros(n_graphs, device=error.device, dtype=error.dtype).index_add(0, batch, error)
        counts = torch.bincount(batch, minlength=n_graphs).to(error.dtype).clamp_min(1)
        per_graph = sums / counts
        if graph_weight is not None:
            return (per_graph * graph_weight).sum() / graph_weight.sum().clamp_min(1)
        return per_graph.mean()
    if node_mask is not None:
        error = error * node_mask
        return error.sum() / node_mask.sum().clamp_min(1)
    return error.mean()


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=1)
    prob = log_prob.gather(1, labels[:, None]).exp()
    return (-((1 - prob) ** gamma) * log_prob.gather(1, labels[:, None])).mean()


LOSSES = {
    "supcon": supcon_loss,
    "node_recon": node_recon_loss,
    "adj_recon": adjacency_loss,
    "ce": cross_entropy,
    "focal": focal_loss,
}


@register("loss", "supcon")
def _supcon(**kwargs: Any):
    return lambda *args, **kw: supcon_loss(*args, **kw, **kwargs)


@register("loss", "node_recon")
def _node_recon(**kwargs: Any):
    return lambda *args, **kw: node_recon_loss(*args, **kw, **kwargs)


@register("loss", "adj_recon")
def _adj_recon(**kwargs: Any):
    return lambda *args, **kw: adjacency_loss(*args, **kw, **kwargs)


@register("loss", "ce")
def _ce(**kwargs: Any):
    return lambda *args, **kw: cross_entropy(*args, **kw, **kwargs)


@register("loss", "focal")
def _focal(**kwargs: Any):
    return lambda *args, **kw: focal_loss(*args, **kw, **kwargs)


def get_loss(name: str):
    try:
        return LOSSES[name]
    except KeyError as exc:
        raise KeyError(f"loss:{name} not available; have {sorted(LOSSES)}") from exc
