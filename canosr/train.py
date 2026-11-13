"""Two-phase training, calibration, and evaluation for one (fold, seed)."""

from __future__ import annotations

import json
import random
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
from .data import GraphStore, assert_disjoint, load_captures, load_splits, recording_id
from .losses import adjacency_loss, node_recon_loss, supcon_loss
from .metrics import binary_report, closed_set_report, open_set_report
from .models import GraphAE
from .registry import build as build_registry
from .scoring import ScoreNormalizer, channel_scores, decide, threshold_from_quantile, two_stage_score
from sklearn.metrics import roc_auc_score

UNKNOWN_LABEL_OFFSET = 1  # benign is 0; the unknown label is n_known


# --------------------------------------------------------------------------- setup


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    variants = set(cfg.data.variants)
    benign_source = cfg.data.benign_source
    captures = {
        c["id"]: c
        for c in meta["captures"]
        if (c["kind"] == "attack" and c["variant"] in variants)
        or (c["kind"] == "benign" and benign_source != "attack_only")
    }
    if not captures:
        raise RuntimeError(f"no captures left after filters variants={sorted(variants)} benign_source={benign_source}")

    holdout = spec.get("holdout")
    attacks = sorted({c["attack"] for c in captures.values() if c["kind"] == "attack"})
    if holdout is not None and holdout not in attacks:
        raise ValueError(f"holdout {holdout!r} not in attacks for variants={sorted(variants)}: {attacks}")
    known = ["benign"] + [a for a in attacks if a != holdout]
    class_of_attack = {name: index for index, name in enumerate(known) if name != "benign"}

    if spec["mode"] == "capture":
        roles = {cid: role for cid, role in spec["roles"].items() if cid in captures}
        assert_disjoint(roles)
        spec = {**spec, "roles": roles}
    return {
        "spec": spec,
        "captures": captures,
        "known": known,
        "class_of_attack": class_of_attack,
        "holdout": holdout,
        "n_known": len(known),
    }


def _relabel(graphs: list, capture: dict, class_of_attack: dict[str, int], n_known: int) -> None:
    class_id = class_of_attack.get(capture["attack"], n_known) if capture["kind"] == "attack" else 0
    for graph in graphs:
        graph.y = torch.tensor([float(class_id)]) if float(graph.y.item()) > 0 else torch.tensor([0.0])


def _contiguous_split(graphs: list, buckets: dict[str, list]) -> None:
    """Split a chronological capture slice 60/20/20 across train/val/test without fab/mas mixing."""
    n = len(graphs)
    if n < 3:
        buckets["train"].extend(graphs)
        return
    n_val = max(1, round(0.2 * n))
    n_test = max(1, round(0.2 * n))
    n_train = n - n_val - n_test
    buckets["train"].extend(graphs[:n_train])
    buckets["val"].extend(graphs[n_train : n_train + n_val])
    buckets["test"].extend(graphs[n_train + n_val :])


def _split_single_recording_group(group_graphs: list[list], buckets: dict[str, list]) -> None:
    """Split a recording group across variants at shared timestamps without boundary leaks."""
    for is_attack in (False, True):
        subsets = [[g for g in gs if (int(g.y.item()) != 0) == is_attack] for gs in group_graphs]
        ref = subsets[0]
        if len(ref) < 3:
            for sub in subsets:
                buckets["train"].extend(sub)
            continue
        if hasattr(ref[0], "t_start") and hasattr(ref[0], "t_end"):
            n = len(ref)
            n_val = max(1, round(0.2 * n))
            n_test = max(1, round(0.2 * n))
            n_train = n - n_val - n_test
            t_cut1 = float(ref[n_train - 1].t_end)
            t_cut2 = float(ref[n_train + n_val - 1].t_end)
            for sub in subsets:
                for g in sub:
                    t0, t1 = float(g.t_start), float(g.t_end)
                    if t1 <= t_cut1:
                        buckets["train"].append(g)
                    elif t0 >= t_cut1 and t1 <= t_cut2:
                        buckets["val"].append(g)
                    elif t0 >= t_cut2:
                        buckets["test"].append(g)
        else:
            for sub in subsets:
                _contiguous_split(sub, buckets)


def load_role_graphs(cfg: Cfg, fold_info: dict[str, Any]) -> dict[str, list]:
    store = GraphStore(cfg.data.root, cfg.dataset, cfg.data.window, cfg.data.stride)
    spec, captures = fold_info["spec"], fold_info["captures"]
    buckets: dict[str, list] = defaultdict(list)

    if spec["mode"] == "capture":
        roles_per_attack: dict[str, set[str]] = defaultdict(set)
        for capture_id, role in spec["roles"].items():
            capture = captures[capture_id]
            if capture["kind"] == "attack" and capture["attack"] != fold_info["holdout"]:
                roles_per_attack[capture["attack"]].add(role)
        single_rec_attacks = {atk for atk, seen_roles in roles_per_attack.items() if seen_roles == {"train"}}

        single_rec_groups: dict[str, list[list]] = defaultdict(list)
        for capture_id, role in sorted(spec["roles"].items()):
            graphs = store.load_capture(capture_id)
            _relabel(graphs, captures[capture_id], fold_info["class_of_attack"], fold_info["n_known"])
            if captures[capture_id].get("attack") in single_rec_attacks:
                single_rec_groups[recording_id(capture_id)].append(graphs)
            else:
                buckets[role].extend(graphs)
        for group_graphs in single_rec_groups.values():
            _split_single_recording_group(group_graphs, buckets)
        return buckets

    # window protocol: split each capture's windows, reproducing the submitted split
    ratios = spec["ratios"]
    rng = np.random.default_rng(spec.get("seed", cfg.data.split_seed))
    for capture_id, capture in sorted(captures.items()):
        graphs = store.load_capture(capture_id)
        _relabel(graphs, capture, fold_info["class_of_attack"], fold_info["n_known"])
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
    benign = [g for g in train_graphs if int(g.y.item()) == 0] or train_graphs
    nodes = torch.cat([g.x for g in benign], dim=0).float()
    mean = nodes[:, 1:].mean(dim=0)
    std = nodes[:, 1:].std(dim=0)
    std = torch.where(std > 0, std, torch.ones_like(std))  # sklearn: zero variance -> unit scale
    edges = torch.cat([g.edge_attr for g in benign if g.edge_attr.numel()], dim=0).float()
    edge_min = edges[:, 0].min()
    edge_max = edges[:, 0].max().clamp_min(edge_min + 1e-6)
    edge_gap_mean = edges[:, 1:].mean(dim=0) if edges.shape[1] > 1 else torch.zeros(0)
    edge_gap_std = edges[:, 1:].std(dim=0) if edges.shape[1] > 1 else torch.ones(0)
    edge_gap_std = torch.where(edge_gap_std > 0, edge_gap_std, torch.ones_like(edge_gap_std))
    return {
        "max_id": max(float(nodes[:, 0].max().item()), 1.0),
        "mean": mean,
        "std": std,
        "edge_min": edge_min,
        "edge_max": edge_max,
        "edge_gap_mean": edge_gap_mean,
        "edge_gap_std": edge_gap_std,
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
            if edge.shape[1] > 1 and "edge_gap_mean" in scalers and scalers["edge_gap_mean"].numel():
                edge[:, 1:] = (edge[:, 1:] - scalers["edge_gap_mean"]) / scalers["edge_gap_std"]
            graph.edge_attr = edge


def make_loader(graphs: list, batch_size: int, shuffle: bool, sampler=None, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
    )


def class_aware_sampler(graphs: list, seed: int = 1) -> torch.utils.data.WeightedRandomSampler:
    labels = np.array([int(g.y.item()) for g in graphs])
    counts = Counter(labels.tolist())
    weights = np.array([1.0 / counts[int(label)] for label in labels])
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(graphs), replacement=True, generator=generator
    )


# ------------------------------------------------------------------------ training


def _batch_losses(model, batch, cfg: Cfg, pos_weight: torch.Tensor | None):
    h = model.encode(batch)
    z = model.embed(h, batch.batch)
    node_hat = model.node_reconstruction(h)
    adj_logits, node_mask = model.adjacency_logits(h, batch.batch)

    benign = batch.y == 0
    if cfg.loss.recon_on_benign_only:
        node_weight = benign[batch.batch].float()
        graph_weight = benign.float()
    else:
        node_weight = None
        graph_weight = None

    target = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=adj_logits.shape[1])
    loss_node = node_recon_loss(
        node_hat,
        batch.x,
        node_mask=node_weight,
        graph_weight=graph_weight,
        skip_first=not cfg.model.id_in_recon,
        batch=batch.batch,
    )
    loss_adj = adjacency_loss(
        adj_logits,
        target,
        node_mask=node_mask,
        graph_weight=graph_weight,
        pos_weight=pos_weight,
    )
    known_mask = batch.y.long() < model.head.n_classes if hasattr(model.head, "n_classes") else torch.ones_like(batch.y, dtype=torch.bool)
    loss_sup = supcon_loss(z[known_mask], batch.y.long()[known_mask], temperature=cfg.loss.temperature) if known_mask.any() else z.sum() * 0.0
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
    if not hasattr(model.head, "set_prototypes"):
        return
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


def _make_optimizer(params, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    key = name.lower()
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


# ----------------------------------------------------------------------------- run


def run(cfg: Cfg, fold: str, seed: int, force: bool = False) -> dict[str, Any]:
    _seed_everything(seed)
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

    if cfg.run.limit:
        rng = np.random.default_rng(seed)
        for role, graphs in buckets.items():
            if len(graphs) > cfg.run.limit:
                order = rng.permutation(len(graphs))[: cfg.run.limit]
                buckets[role] = [graphs[i] for i in order]

    scalers = fit_scalers(buckets["train"])
    for graphs in buckets.values():
        apply_scalers(graphs, scalers)

    pos_weight = (
        _auto_pos_weight([g for g in buckets["train"] if int(g.y.item()) == 0]).to(device)
        if cfg.loss.adj_pos_weight == "auto"
        else None
    )

    sampler = class_aware_sampler(buckets["train"], seed=seed) if cfg.train.sampler == "class_aware" else None
    train_loader = make_loader(buckets["train"], cfg.train.batch_size, True, sampler, cfg.train.num_workers)
    eval_train_loader = make_loader(buckets["train"], cfg.train.batch_size, False, None, cfg.train.num_workers)
    val_loader = make_loader(buckets["val"], cfg.train.batch_size, False, None, cfg.train.num_workers)
    test_loader = make_loader(buckets["test"], cfg.train.batch_size, False, None, cfg.train.num_workers)

    in_dim = buckets["train"][0].num_node_features
    model = GraphAE(in_dim=in_dim, n_classes=n_known, cfg=cfg.model, edge_dim=int(cfg.data.edge_feat)).to(device)
    optimizer = _make_optimizer(
        model.trainable_parameters(1), cfg.train.optimizer, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    amp_enabled = bool(cfg.run.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history: list[dict[str, Any]] = []
    best = {"score": -1.0, "epoch": -1, "state": None}
    started = time.time()
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.phase2_start + 1, eta_min=cfg.train.lr * 0.01
        )
        if cfg.train.scheduler == "cosine"
        else None
    )

    for epoch in range(cfg.train.epochs):
        phase = 2 if epoch >= cfg.train.phase2_start else 1
        if phase == 2 and cfg.train.phase2_mode == "freeze" and not getattr(model, "_frozen", False):
            model.freeze_for_phase2()
            optimizer = _make_optimizer(
                model.trainable_parameters(2), cfg.train.optimizer, lr=cfg.train.lr_head, weight_decay=cfg.train.weight_decay
            )
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.train.epochs - cfg.train.phase2_start, eta_min=cfg.train.lr_head * 0.01
                )
                if cfg.train.scheduler == "cosine"
                else None
            )
            model._frozen = True
        if phase == 2:
            compute_prototypes(model, eval_train_loader, n_known, device)

        model.train()
        totals: dict[str, float] = defaultdict(float)
        n_steps = 0
        skipped_nonfinite = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss_sup, loss_node, loss_adj, z, h = _batch_losses(model, batch, cfg, pos_weight)

                recon = cfg.loss.w_supcon * loss_sup + cfg.loss.w_node * loss_node + cfg.loss.w_adj * loss_adj
                if phase == 1:
                    loss = recon
                else:
                    logits = (z @ model.head.prototypes.t() / 0.1) if hasattr(model.head, "prototypes") else model.head(z)
                    ce = F.cross_entropy(logits, batch.y.long())
                    if cfg.train.phase2_mode == "freeze":
                        loss = cfg.loss.w_ce * ce + 0.1 * loss_sup
                    else:
                        loss = recon + cfg.loss.w_ce * ce

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                continue
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            n_steps += 1
            totals["loss"] += float(loss.item())
            totals["supcon"] += float(loss_sup.item())
            totals["node"] += float(loss_node.item())
            totals["adj"] += float(loss_adj.item())

        record = {
            "epoch": epoch,
            "phase": phase,
            "lr": optimizer.param_groups[0]["lr"],
            "skipped_nonfinite": skipped_nonfinite,
            **{k: v / max(n_steps, 1) for k, v in totals.items()},
        }
        if scheduler is not None and (epoch <= cfg.train.phase2_start or cfg.train.phase2_mode == "freeze"):
            # In joint mode the Phase-1 cosine schedule reaches eta_min=lr*0.01=1e-4 at phase2_start,
            # matching the submitted Phase-2 LR without further stepping.
            scheduler.step()
        if phase == 2 and (epoch % cfg.eval.every == 0 or epoch == cfg.train.epochs - 1):
            labels, preds = _evaluate_known(model, val_loader, device)
            report = closed_set_report(labels, preds, n_known)
            record.update({f"val_{k}": v for k, v in report.items() if not isinstance(v, dict)})
            metric_name = cfg.eval.select_metric
            score = float(report.get(metric_name, report["macro_recall"]))
            if not np.isnan(score) and score > best["score"]:
                best = {
                    "score": score,
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

    # calibration: fused score on benign val (y == 0), proto score on known-attack val (0 < y < n_known)
    val_channels = collect_channels(model, val_loader, device)
    val_y = np.concatenate([g.y.numpy() for g in buckets["val"]])
    benign_val = val_y == 0
    attack_val = (val_y > 0) & (val_y < n_known)
    if not benign_val.any() or not attack_val.any():
        train_channels = collect_channels(model, eval_train_loader, device)
        train_y = np.concatenate([g.y.numpy() for g in buckets["train"]])
    if not benign_val.any():
        benign_train = train_y == 0
        cal_source = {k: torch.from_numpy(v[benign_train]) for k, v in train_channels.items()}
    else:
        cal_source = {k: torch.from_numpy(v[benign_val]) for k, v in val_channels.items()}
    normalizer = ScoreNormalizer.fit(cal_source, cfg.osr.channels)
    scorer = build_registry("scorer", cfg.osr.scorer, normalizer=normalizer, keys=cfg.osr.channels, weights=cfg.osr.weights)
    cal_fused = scorer(cal_source)
    threshold = threshold_from_quantile(cal_fused, cfg.osr.threshold_quantile)
    if attack_val.any():
        proto_cal = torch.from_numpy(val_channels["proto"][attack_val])
    else:
        attack_train = (train_y > 0) & (train_y < n_known)
        proto_cal = torch.from_numpy(
            train_channels["proto"][attack_train if attack_train.any() else (train_y < n_known)]
        )
    proto_threshold = threshold_from_quantile(proto_cal, cfg.osr.threshold_quantile)
    proto_std = float(proto_cal.float().std(unbiased=False).item()) or 1.0

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

    test_proto = torch.from_numpy(test_channels["proto"])
    test_osr = two_stage_score(test_fused, threshold, test_logits, test_proto, proto_threshold, proto_std)
    predictions = decide(
        test_fused,
        threshold,
        test_logits,
        n_known,
        proto_scores=test_proto,
        proto_threshold=proto_threshold,
    )
    y_true = test_labels.numpy()
    y_pred = predictions.numpy()
    scores = test_osr.numpy()

    known_pred = test_logits.argmax(dim=1).numpy()
    if fold_info["holdout"] is None:
        closed = closed_set_report(y_true, known_pred, n_known)
        binary = binary_report(y_true != 0, known_pred != 0)
        open_report = {
            "osr_acc": closed["acc"],
            "oscr": float("nan"),
            "openness": float("nan"),
            "known_acc": closed["acc"],
            "unknown_recall": float("nan"),
            "n_known": int(len(y_true)),
            "n_unknown": 0,
            "auroc": float("nan"),
            "aupr": float("nan"),
            "fpr_at_95tpr": float("nan"),
        }
        closed_set = closed
    else:
        binary = binary_report(y_true != 0, y_pred != 0)
        open_report = open_set_report(
            y_true, y_pred, n_known=n_known, n_unknown_classes=1,
            scores=scores, threshold=0.0, known_pred=known_pred,
        )
        known_mask = y_true < n_known
        closed_set = closed_set_report(y_true[known_mask], known_pred[known_mask], n_known)

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
        "proto_threshold": proto_threshold,
        "proto_std": proto_std,
        "binary": binary,
        "open_set": open_report,
        "closed_set": closed_set,
        "per_channel_auroc": per_channel,
        "best_epoch": best["epoch"],
        "best_val_metric": cfg.eval.select_metric,
        "best_val_score": best["score"],
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
            "known_pred": known_pred,
            "score": scores,
            "fused_score": test_fused.numpy(),
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
