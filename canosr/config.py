"""Resolved configuration: YAML merge, attribute access, content hash."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "dataset": "road",
    "split": "capture",
    "seed": 1,
    "data": {
        "root": "",  # resolved from CANOSR_DATA
        "window": 300,
        "stride": 300,
        "split_seed": 1,
        "raw_node_feat": 13,
        "edge_feat": 3,
        "feature_version": "v1",
    },
    "model": {
        "encoder": "gatv2",
        "head": "prototype",
        "hidden": 8,
        "heads": 5,
        "layers": 2,
        "dropout": 0.3,
        "projection": 8,
        "readout": "attn",
        "id_embedding": False,
        "id_in_recon": False,
    },
    "loss": {
        "supcon": "supcon",
        "temperature": 0.05,
        "w_supcon": 2.0,
        "w_node": 0.5,
        "w_adj": 0.7,
        "w_ce": 1.0,
        "recon_on_benign_only": True,
        "adj_pos_weight": "auto",
    },
    "train": {
        "epochs": 330,
        "phase2_start": 300,
        "batch_size": 512,
        "lr": 0.01,
        "lr_head": 1e-4,
        "optimizer": "adamw",
        "weight_decay": 5e-4,
        "scheduler": "cosine",
        "grad_clip": 5.0,
        "sampler": "class_aware",
        "num_workers": 0,
    },
    "osr": {
        "scorer": "fused",
        "channels": ["proto", "node", "adj"],
        "weights": None,
        "threshold_quantile": 0.95,
        "fit_weights": False,
    },
    "eval": {
        "every": 10,
        "select_metric": "macro_f1",
    },
    "run": {
        "root": "",
        "device": "cuda",
        "amp": False,
        "force": False,
    },
}


class Cfg(dict):
    """Dict with attribute access, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict[str, Any]:
        def plain(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: plain(v) for k, v in obj.items()}
            return obj

        return plain(dict(self))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def data_root(cfg: Cfg | dict[str, Any] | None = None) -> Path:
    if cfg is not None:
        configured = (cfg.get("data") or {}).get("root")
        if configured:
            return Path(configured).expanduser()
    return Path(os.environ.get("CANOSR_DATA", "/mnt/data2/ntmduy/data/can-osr")).expanduser()


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Cfg:
    merged = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"{path}: top level must be a mapping")
        merged = deep_merge(merged, loaded)
    if overrides:
        merged = deep_merge(merged, overrides)

    cfg = Cfg(merged)
    root = data_root(cfg)
    if not cfg["data"]["root"]:
        cfg["data"]["root"] = str(root)
    if not cfg["run"]["root"]:
        cfg["run"]["root"] = str(root / "runs")
    return cfg


def config_hash(cfg: Cfg) -> str:
    """Stable hash of everything that can change a number."""
    payload = cfg.to_dict()
    payload.pop("run", None)  # device/force do not affect results
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def save_config(cfg: Cfg, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=True)
