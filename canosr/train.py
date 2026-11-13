"""Two-phase training, calibration, and evaluation for one (fold, seed)."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .config import Cfg, config_hash, save_config
from .data import GraphStore, assert_disjoint, load_captures, load_splits
from .losses import adjacency_loss, node_recon_loss, supcon_loss
from .metrics import binary_report, closed_set_report, open_set_report
from .models import GraphAE
from .registry import build as build_registry
from .scoring import ScoreNormalizer, channel_scores, decide, threshold_from_quantile
from sklearn.metrics import roc_auc_score

UNKNOWN_LABEL_OFFSET = 1  # benign is 0; the unknown label is n_known


# --------------------------------------------------------------------------- setup


def _device(cfg: Cfg) -> torch.device:
    requested = cfg.run.device
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def resolve_fold(cfg: Cfg, fold: str) -> dict[str, Any]:
    data_root = Path(cfg.data.root)
    meta = load_captures(cfg.dataset, data_root)
    splits = load_splits(cfg.dataset, data_root, cfg.split, cfg.data.split_seed)
    if fold not in splits["folds"]:
        raise KeyError(f"fold {fold!r} not in {sorted(splits['folds'])}")
    spec = splits["folds"][fold]

    holdout = spec.get("holdout")
    attacks = sorted({c["attack"] for c in meta["captures"] if c["kind"] == "attack"})
    known = ["benign"] + [a for a in attacks if a != holdout]
    class_of_attack = {name: index for index, name in enumerate(known) if name != "benign"}

    if spec["mode"] == "capture":
        assert_disjoint(spec["roles"])
    return {
        "spec": spec,
        "captures": {c["id"]: c for c in meta["captures"]},
        "known": known,
        "class_of_attack": class_of_attack,
        "holdout": holdout,
        "n_known": len(known),
    }


def _relabel(graphs: list, capture: dict, class_of_attack: dict[str, int]) -> None:
    class_id = class_of_attack.get(capture["attack"], 0) if capture["kind"] == "attack" else 0
    for graph in graphs:
        graph.y = torch.tensor([float(class_id)]) if float(graph.y.item()) > 0 else torch.tensor([0.0])


def load_role_graphs(cfg: Cfg, fold_info: dict[str, Any]) -> dict[str, list]:
    store = GraphStore(cfg.data.root, cfg.dataset, cfg.data.window, cfg.data.stride)
    spec, captures = fold_info["spec"], fold_info["captures"]
    buckets: dict[str, list] = defaultdict(list)

    if spec["mode"] == "capture":
        for capture_id, role in spec["roles"].items():
            graphs = store.load_capture(capture_id)
            _relabel(graphs, captures[capture_id], fold_info["class_of_attack"])
            buckets[role].extend(graphs)
        return buckets

    # window protocol: split each capture's windows, reproducing the submitted split
    ratios = spec["ratios"]
    rng = np.random.default_rng(cfg.seed)
    for capture_id, capture in sorted(captures.items()):
        graphs = store.load_capture(capture_id)
        _relabel(graphs, capture, fold_info["class_of_attack"])
        if spec.get("holdout") and capture["attack"] == spec["holdout"]:
            benign = [g for g in graphs if int(g.y.item()) == 0]
            attack = [g for g in graphs if int(g.y.item()) != 0]
            order = rng.permutation(len(benign))
            n_train = int(round(ratios["train"] * len(benign)))
            n_val = int(round(ratios["val"] * len(benign)))
            buckets["train"].extend(benign[i] for i in order[:n_train])
            buckets["val"].extend(benign[i] for i in order[n_train : n_train + n_val])
            buckets["test"].extend(benign[i] for i in order[n_train + n_val :])
            buckets["test"].extend(attack)
            continue
        order = rng.permutation(len(graphs))
        n_train = int(round(ratios["train"] * len(graphs)))
        n_val = int(round(ratios["val"] * len(graphs)))
        buckets["train"].extend(graphs[i] for i in order[:n_train])
        buckets["val"].extend(graphs[i] for i in order[n_train : n_train + n_val])
        buckets["test"].extend(graphs[i] for i in order[n_train + n_val :])
    return buckets


def fit_scalers(train_graphs: list) -> dict[str, Any]:
    benign = [g for g in train_graphs if int(g.y.item()) == 0]
    nodes = torch.cat([g.x for g in benign], dim=0).float()
    mean = nodes[:, 1:].mean(dim=0)
    std = nodes[:, 1:].std(dim=0).clamp_min(1e-6)
    edges = torch.cat([g.edge_attr for g in benign if g.edge_attr.numel()], dim=0).float()
    edge_min = edges[:, 0].min()
    edge_max = edges[:, 0].max().clamp_min(edge_min + 1e-6)
    return {
        "max_id": float(nodes[:, 0].max().item()),
        "mean": mean,
        "std": std,
        "edge_min": edge_min,
        "edge_max": edge_max,
    }


def apply_scalers(graphs: list, scalers: dict[str, Any]) -> None:
    max_id = max(scalers["max_id"], 1.0)
    for graph in graphs:
        graph.x = graph.x.clone()
        graph.x[:, 0] = graph.x[:, 0] / max_id
        graph.x[:, 1:] = (graph.x[:, 1:] - scalers["mean"]) / scalers["std"]
        if graph.edge_attr.numel():
            edge = graph.edge_attr.clone()
            edge[:, 0] = (edge[:, 0] - scalers["edge_min"]) / (scalers["edge_max"] - scalers["edge_min"])
            graph.edge_attr = edge


def make_loader(graphs: list, batch_size: int, shuffle: bool, sampler=None, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
    )


def class_aware_sampler(graphs: list) -> torch.utils.data.WeightedRandomSampler:
    labels = np.array([int(g.y.item()) for g in graphs])
    counts = Counter(labels.tolist())
    weights = np.array([1.0 / counts[int(label)] for label in labels])
    return torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(graphs), replacement=True
    )


# ------------------------------------------------------------------------ training


def _batch_losses(model, batch, cfg: Cfg, pos_weight: torch.Tensor | None):
    h = model.encode(batch)
    z = model.embed(h, batch.batch)
    node_hat = model.node_reconstruction(h)
    adj_logits, node_mask = model.adjacency_logits(h, batch.batch)

    n_graphs = int(batch.batch.max().item()) + 1
    benign = batch.y == 0
    node_weight = benign[batch.batch].float()
    graph_weight = benign.float()

    target = to_dense_adj(batch.edge_index, batch, max_num_nodes=adj_logits.shape[1])
    loss_node = node_recon_loss(
        node_hat,
        batch.x,
        node_mask=node_weight,
        graph_weight=None,
        skip_first=not cfg.model.id_in_recon,
    )
    loss_adj = adjacency_loss(
        adj_logits,
        target,
        node_mask=node_mask,
        graph_weight=graph_weight,
        pos_weight=pos_weight,
    )
    loss_sup = supcon_loss(z, batch.y.long(), temperature=cfg.loss.temperature)
    return loss_sup, loss_node, loss_adj, z, h


def _auto_pos_weight(train_graphs: list, max_graphs: int = 256) -> torch.Tensor:
    positive = negative = 0.0
    for graph in train_graphs[:max_graphs]:
        n = graph.num_nodes
        possible = n * n
        edges = graph.edge_index.shape[1]
        positive += edges
        negative += possible - edges
    ratio = negative / max(positive, 1.0)
    return torch.tensor([ratio], dtype=torch.float32)


@torch.no_grad()
def compute_prototypes(model, loader, n_classes: int, device) -> None:
    model.eval()
    embeddings, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        h = model.encode(batch)
        embeddings.append(model.embed(h, batch.batch))
        labels.append(batch.y.long())
    model.head.set_prototypes(torch.cat(embeddings), torch.cat(labels))
    model.train()


def _evaluate_known(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        h = model.encode(batch)
        z = model.embed(h, batch.batch)
        preds.append(model.head(z).argmax(dim=1).cpu())
        labels.append(batch.y.long().cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


@torch.no_grad()
def collect_channels(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for batch in loader:
        batch = batch.to(device)
        for key, value in channel_scores(model, batch, skip_id=not model.id_in_recon).items():
            chunks[key].append(value.cpu().numpy())
    return {key: np.concatenate(values) for key, values in chunks.items()}


# ----------------------------------------------------------------------------- run


def run(cfg: Cfg, fold: str, seed: int, force: bool = False) -> dict[str, Any]:
    cfg = Cfg(json.loads(json.dumps(cfg.to_dict())))
    cfg["seed"] = seed
    device = _device(cfg)

    fold_info = resolve_fold(cfg, fold)
    n_known = fold_info["n_known"]
    run_id = f"{cfg.dataset}_{cfg.split}_{fold.replace('/', '-')}_{seed}_{config_hash(cfg)[:8]}"
    run_dir = Path(cfg.run.root) / run_id
    if run_dir.exists() and not force:
        raise FileExistsError(f"{run_dir} exists; pass --force to overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    buckets = load_role_graphs(cfg, fold_info)
    for role in ("train", "val", "test"):
        if not buckets[role]:
            raise RuntimeError(f"fold {fold}: empty {role} split")

    scalers = fit_scalers(buckets["train"])
    for graphs in buckets.values():
        apply_scalers(graphs, scalers)

    pos_weight = _auto_pos_weight(buckets["train"]).to(device) if cfg.loss.adj_pos_weight == "auto" else None

    sampler = class_aware_sampler(buckets["train"]) if cfg.train.sampler == "class_aware" else None
    train_loader = make_loader(buckets["train"], cfg.train.batch_size, True, sampler, cfg.train.num_workers)
    eval_train_loader = make_loader(buckets["train"], cfg.train.batch_size, False, None, cfg.train.num_workers)
    val_loader = make_loader(buckets["val"], cfg.train.batch_size, False, None, cfg.train.num_workers)
    test_loader = make_loader(buckets["test"], cfg.train.batch_size, False, None, cfg.train.num_workers)

    in_dim = buckets["train"][0].num_node_features
    model = GraphAE(in_dim=in_dim, n_classes=n_known, cfg=cfg.model, edge_dim=int(cfg.data.edge_feat)).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(1), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    history: list[dict[str, Any]] = []
    best = {"score": -1.0, "epoch": -1, "state": None}
    started = time.time()

    for epoch in range(cfg.train.epochs):
        phase = 2 if epoch >= cfg.train.phase2_start else 1
        if phase == 2 and not getattr(model, "_frozen", False):
            model.freeze_for_phase2()
            optimizer = torch.optim.AdamW(
                model.trainable_parameters(2), lr=cfg.train.lr_head, weight_decay=cfg.train.weight_decay
            )
            model._frozen = True
        if phase == 2:
            compute_prototypes(model, eval_train_loader, n_known, device)

        model.train()
        totals = defaultdict(float)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_sup, loss_node, loss_adj, z, h = _batch_losses(model, batch, cfg, pos_weight)

            if phase == 1:
                loss = (
                    cfg.loss.w_supcon * loss_sup
                    + cfg.loss.w_node * loss_node
                    + cfg.loss.w_adj * loss_adj
                )
            else:
                logits = z @ model.head.prototypes.t() / 0.1
                loss = cfg.loss.w_ce * F.cross_entropy(logits, batch.y.long()) + 0.1 * loss_sup

            if not torch.isfinite(loss):
                continue
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["supcon"] += float(loss_sup.item())
            totals["node"] += float(loss_node.item())
            totals["adj"] += float(loss_adj.item())

        record = {"epoch": epoch, "phase": phase, **{k: v / max(len(train_loader), 1) for k, v in totals.items()}}
        if phase == 2 and (epoch % cfg.eval.every == 0 or epoch == cfg.train.epochs - 1):
            labels, preds = _evaluate_known(model, val_loader, device)
            report = closed_set_report(labels, preds, n_known)
            record.update({f"val_{k}": v for k, v in report.items() if not isinstance(v, dict)})
            if report["macro_recall"] > best["score"]:
                best = {
                    "score": report["macro_recall"],
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
        history.append(record)

    if best["state"] is None:
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best["epoch"] = cfg.train.epochs - 1
    model.load_state_dict(best["state"])
    model.eval()
    compute_prototypes(model, eval_train_loader, n_known, device)
    model.eval()

    # calibration on validation benign only
    val_channels = collect_channels(model, val_loader, device)
    benign_val = np.concatenate([g.y.numpy() for g in buckets["val"]]) == 0
    normalizer = ScoreNormalizer.fit(
        {k: torch.from_numpy(v[benign_val]) for k, v in val_channels.items()}, cfg.osr.channels
    )
    scorer = build_registry("scorer", cfg.osr.scorer, normalizer=normalizer, keys=cfg.osr.channels, weights=cfg.osr.weights)
    val_fused = scorer({k: torch.from_numpy(v) for k, v in val_channels.items()})
    threshold = threshold_from_quantile(val_fused[torch.from_numpy(benign_val)], cfg.osr.threshold_quantile)

    test_channels = collect_channels(model, test_loader, device)
    test_fused = scorer({k: torch.from_numpy(v) for k, v in test_channels.items()})
    test_logits = []
    test_labels = []
    for batch in test_loader:
        batch = batch.to(device)
        h = model.encode(batch)
        z = model.embed(h, batch.batch)
        test_logits.append(model.head(z).cpu())
        test_labels.append(batch.y.long().cpu())
    test_logits = torch.cat(test_logits)
    test_labels = torch.cat(test_labels)

    predictions = decide(test_fused, threshold, test_logits, n_known)
    y_true = test_labels.numpy()
    y_pred = predictions.numpy()
    scores = test_fused.numpy()

    binary = binary_report(y_true != 0, y_pred != 0)
    open_report = open_set_report(
        y_true, y_pred, n_known=n_known, n_unknown_classes=1 if fold_info["holdout"] else 0,
        scores=scores, threshold=threshold,
    )
    per_channel = {
        key: float(roc_auc_score((y_true == n_known).astype(int), value))
        for key, value in test_channels.items()
        if len(np.unique(y_true == n_known)) > 1
    }

    metrics = {
        "run_id": run_id,
        "fold": fold,
        "seed": seed,
        "holdout": fold_info["holdout"],
        "n_known": n_known,
        "n_test": int(len(y_true)),
        "threshold": threshold,
        "binary": binary,
        "open_set": open_report,
        "per_channel_auroc": per_channel,
        "best_epoch": best["epoch"],
        "best_val_macro_recall": best["score"],
        "train_seconds": time.time() - started,
        "history": history,
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=1)
    torch.save(model.state_dict(), run_dir / "model.pt")
    pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "score": scores,
            **{f"ch_{k}": v for k, v in test_channels.items()},
        }
    ).to_parquet(run_dir / "predictions.parquet")

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "fold": fold,
                "seed": seed,
                "n_known": n_known,
                "known_classes": fold_info["known"],
                "counts": {role: len(graphs) for role, graphs in buckets.items()},
                "split": cfg.split,
            },
            handle,
            indent=1,
        )
    return metrics
