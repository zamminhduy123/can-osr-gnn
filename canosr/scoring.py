"""Open-set scoring: per-graph channels, normalisation, fusion, threshold.

One scalar per graph. Channels are z-scored on benign validation data only;
fusion weights are equal by default and never fitted on unknowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .registry import register


def per_graph_mean(values: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    sums = torch.zeros(n_graphs, device=values.device, dtype=values.dtype)
    sums = sums.index_add(0, batch, values)
    counts = torch.bincount(batch, minlength=n_graphs).to(values.dtype)
    return sums / counts.clamp_min(1)


@torch.no_grad()
def channel_scores(model, data, skip_id: bool = True) -> dict[str, torch.Tensor]:
    """Per-graph anomaly channels. Higher means more anomalous."""
    out = model(data)
    h, z, logits = out["h"], out["z"], out["logits"]
    batch = data.batch
    n_graphs = int(batch.max().item()) + 1

    node_error = ((out["node_hat"][:, 1:] - data.x[:, 1:]) ** 2).mean(dim=1) if skip_id else (
        (out["node_hat"] - data.x) ** 2
    ).mean(dim=1)
    node = per_graph_mean(node_error, batch, n_graphs)

    target = to_dense_adj(data.edge_index, batch, max_num_nodes=out["adj_logits"].shape[1])
    binary = (target > 0).float()
    pair_error = (torch.sigmoid(out["adj_logits"]) - binary) ** 2
    dense_mask, _ = to_dense_batch(torch.ones_like(h[:, 0]), batch, max_num_nodes=out["adj_logits"].shape[1])
    pair_mask = dense_mask[:, :, None] * dense_mask[:, None, :]
    adj = (pair_error * pair_mask).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1)

    distance = -logits.max(dim=1).values  # logits are negative distances
    proto = distance

    return {
        "proto": proto,
        "node": node,
        "adj": adj,
        "msp": torch.softmax(logits, dim=1).max(dim=1).values,
        "energy": torch.logsumexp(logits, dim=1),
        "distance": distance,
    }


@dataclass
class ScoreNormalizer:
    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def fit(cls, channels: dict[str, torch.Tensor], keys: list[str]) -> "ScoreNormalizer":
        mean, std = {}, {}
        for key in keys:
            values = channels[key].detach().float()
            mean[key] = float(values.mean().item())
            std[key] = float(values.std(unbiased=False).item()) or 1.0
        return cls(mean=mean, std=std)

    def transform(self, channels: dict[str, torch.Tensor], keys: list[str], weights: dict[str, float] | None = None) -> torch.Tensor:
        parts = []
        for key in keys:
            z = (channels[key].float() - self.mean[key]) / self.std[key]
            weight = 1.0 if weights is None else float(weights.get(key, 1.0))
            parts.append(z * weight)
        return torch.stack(parts, dim=0).sum(dim=0) / max(len(parts), 1)


def threshold_from_quantile(scores: torch.Tensor, quantile: float) -> float:
    return float(torch.quantile(scores.float(), quantile).item())


def decide(scores: torch.Tensor, threshold: float, logits: torch.Tensor, n_known: int) -> torch.Tensor:
    """Unknown label is ``n_known``; otherwise argmax over known classes."""
    known = logits.argmax(dim=1)
    unknown = scores > threshold
    return torch.where(unknown, torch.full_like(known, n_known), known)


@register("scorer", "fused")
def fused_scorer(normalizer: ScoreNormalizer, keys: list[str], weights: dict[str, float] | None = None):
    def score(channels: dict[str, torch.Tensor]) -> torch.Tensor:
        return normalizer.transform(channels, keys, weights)

    return score


@register("scorer", "recon")
def recon_scorer(normalizer: ScoreNormalizer, keys: list[str] | None = None, weights: dict[str, float] | None = None):
    keys = keys or ["node", "adj"]

    def score(channels: dict[str, torch.Tensor]) -> torch.Tensor:
        return normalizer.transform(channels, keys, weights)

    return score


@register("scorer", "proto_distance")
def proto_scorer(normalizer: ScoreNormalizer, keys: list[str] | None = None, weights: dict[str, float] | None = None):
    def score(channels: dict[str, torch.Tensor]) -> torch.Tensor:
        return channels["proto"]

    return score


@register("scorer", "msp")
def msp_scorer(normalizer: ScoreNormalizer, keys: list[str] | None = None, weights: dict[str, float] | None = None):
    def score(channels: dict[str, torch.Tensor]) -> torch.Tensor:
        return -channels["msp"]

    return score


@register("scorer", "energy")
def energy_scorer(normalizer: ScoreNormalizer, keys: list[str] | None = None, weights: dict[str, float] | None = None):
    def score(channels: dict[str, torch.Tensor]) -> torch.Tensor:
        return -channels["energy"]

    return score


@register("scorer", "mahalanobis")
def mahalanobis_scorer(normalizer: ScoreNormalizer, keys: list[str] | None = None, weights: dict[str, float] | None = None):
    """Placeholder: falls back to prototype distance until a covariance is fitted."""

    def score(channels: dict[str, torch.Tensor]) -> torch.Tensor:
        return channels["distance"]

    return score
