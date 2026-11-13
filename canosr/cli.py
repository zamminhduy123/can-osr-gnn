"""Command line entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import Cfg, load_config, data_root
from .data import (
    assert_disjoint,
    build_captures,
    build_graph_cache,
    build_splits,
    load_captures,
    load_splits,
)

PARITY = {
    "closed_set_fab": {"acc": 0.9981, "precision": 0.9908, "recall": 0.9977, "f1": 0.9940},
    "closed_set_mas": {"acc": 0.9980, "precision": 0.9840, "recall": 0.9989, "f1": 0.9911},
    "leave_one_out/reverse_light_off": {"acc": 0.9801, "precision": 0.9625, "recall": 0.9854, "f1": 0.9732},
    "leave_one_out/reverse_light_on": {"acc": 0.9809, "precision": 0.9673, "recall": 0.9869, "f1": 0.9764},
    "leave_one_out/max_speedometer": {"acc": 0.9826, "precision": 0.9763, "recall": 0.9864, "f1": 0.9810},
    "leave_one_out/correlated_signal": {"acc": 0.9711, "precision": 0.9558, "recall": 0.9788, "f1": 0.9661},
    "leave_one_out/fuzzing": {"acc": 0.9790, "precision": 0.6969, "recall": 0.9585, "f1": 0.7724},
    "leave_one_out/max_engine_coolant_temp": {"acc": 0.9684, "precision": 0.7462, "recall": 0.9837, "f1": 0.8216},
}

TOLERANCE = {"acc": 0.02, "recall": 0.02, "precision": 0.05, "f1": 0.05}


def _overrides(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        value = yaml.safe_load(raw)
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def cmd_build_captures(args: argparse.Namespace) -> None:
    payload = build_captures(args.dataset, data_root())
    print(json.dumps({"dataset": payload["dataset"], "captures": len(payload["captures"]), "fingerprint": payload["fingerprint"]}))


def cmd_build_graphs(args: argparse.Namespace) -> None:
    summary = build_graph_cache(
        args.dataset, data_root(), window=args.window, stride=args.stride, force=args.force, limit=args.limit
    )
    print(json.dumps(summary))


def cmd_build_splits(args: argparse.Namespace) -> None:
    payload = build_splits(args.dataset, data_root(), split=args.split, seed=args.seed)
    print(json.dumps({"folds": sorted(payload["folds"]), "split": payload["split"], "seed": payload["seed"]}, indent=1))


def cmd_run(args: argparse.Namespace) -> None:
    from .train import run

    cfg = load_config(args.config, _overrides(args.set or []))
    metrics = run(cfg, fold=args.fold, seed=args.seed, force=args.force)
    print(json.dumps({k: metrics[k] for k in ("run_id", "fold", "seed", "threshold")}, indent=1))


def cmd_aggregate(args: argparse.Namespace) -> None:
    runs = Path(args.runs)
    rows = []
    for path in sorted(runs.glob("*/metrics.json")):
        run_dir = path.parent
        with open(path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        split = "unknown"
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as handle:
                split = json.load(handle).get("split", "unknown")
        variants = "unknown"
        benign_source = "unknown"
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as handle:
                run_cfg = yaml.safe_load(handle) or {}
            data_cfg = run_cfg.get("data") or {}
            variants = "+".join(data_cfg.get("variants") or []) or "none"
            benign_source = data_cfg.get("benign_source", "unknown")
        row = {
            "run_id": metrics["run_id"],
            "cfg_hash": metrics["run_id"].rsplit("_", 1)[-1],
            "split": split,
            "variants": variants,
            "benign_source": benign_source,
            "fold": metrics["fold"],
            "seed": metrics["seed"],
            "n_test": metrics.get("n_test", 0),
            "threshold": metrics["threshold"],
            **{f"binary_{k}": v for k, v in metrics["binary"].items()},
            **{f"open_{k}": v for k, v in metrics["open_set"].items()},
        }
        rows.append(row)
    if not rows:
        raise SystemExit(f"no metrics.json under {runs}")
    frame = pd.DataFrame(rows)
    out = runs / "summary.parquet"
    frame.to_parquet(out)
    group_cols = ["cfg_hash", "split", "variants", "benign_source", "fold"]
    grouped = frame.groupby(group_cols)
    summary = grouped[["open_auroc", "open_oscr", "binary_f1", "binary_recall"]].mean()
    summary["n_runs"] = grouped["seed"].count()
    summary["n_unknown"] = grouped["open_n_unknown"].mean()
    for col in ("open_auroc", "binary_f1"):
        std = grouped[col].std(ddof=1).fillna(0.0)
        summary[f"{col}_ci95"] = 1.96 * std / (summary["n_runs"].clip(lower=1) ** 0.5)
    print(summary.round(4).to_string())
    print(f"\nwrote {out}")


def cmd_gate(args: argparse.Namespace) -> None:
    checks: list[tuple[str, bool, str]] = []

    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
        checks.append(("G1 unit tests", result.returncode == 0, result.stdout.strip().splitlines()[-1] if result.stdout else ""))
    except Exception as exc:  # pragma: no cover
        checks.append(("G1 unit tests", False, str(exc)))

    root = data_root()
    splits = sorted((root / "processed" / args.dataset / "splits").glob("*.json"))
    leakage_ok = True
    detail = f"{len(splits)} split files"
    for path in splits:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for fold, spec in payload["folds"].items():
            if spec.get("mode") != "capture":
                continue
            try:
                assert_disjoint(spec["roles"])
            except AssertionError as exc:
                leakage_ok = False
                detail = f"{path.name}:{fold} {exc}"
    checks.append(("G5 leakage", leakage_ok, detail))

    runs = Path(args.runs)
    run_dirs = sorted(p for p in runs.glob("*/") if (p / "metrics.json").exists())
    required = {"config.yaml", "metrics.json", "predictions.parquet", "model.pt", "manifest.json"}
    incomplete = [p.name for p in run_dirs if not required.issubset({f.name for f in p.iterdir()})]
    checks.append(("G3 run artifacts", not incomplete, f"{len(run_dirs)} runs, {len(incomplete)} incomplete"))

    parity_rows = []
    for path in run_dirs:
        with open(path / "metrics.json", "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("split") != "window":
                continue
        fold_key = metrics["fold"]
        variants: list[str] = []
        if (path / "config.yaml").exists():
            with open(path / "config.yaml", "r", encoding="utf-8") as handle:
                run_cfg = yaml.safe_load(handle) or {}
            variants = (run_cfg.get("data") or {}).get("variants") or []
        if fold_key == "closed_set":
            if variants == ["fab"]:
                fold_key = "closed_set_fab"
            elif variants == ["mas"]:
                fold_key = "closed_set_mas"
        elif fold_key.startswith("leave_one_out/") and variants != ["fab"]:
            continue
        target = PARITY.get(fold_key)
        if target is None:
            continue
        observed = metrics["binary"]
        diffs = {k: observed[k] - v for k, v in target.items()}
        ok = all(abs(diffs[k]) <= TOLERANCE[k] for k in target)
        parity_rows.append((fold_key, metrics["seed"], ok, {k: round(v, 4) for k, v in diffs.items()}))
    parity_ok = bool(parity_rows) and all(row[2] for row in parity_rows)
    checks.append(("G7 parity (window split)", parity_ok, f"{len(parity_rows)} matching runs"))

    print()
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<26} {detail}")
    if parity_rows:
        print("\nparity deltas (observed - paper):")
        for fold, seed, ok, diffs in parity_rows:
            print(f"  {'ok ' if ok else 'BAD'} {fold:<40} seed={seed} {diffs}")
    print("\nG2 smoke / G4 determinism: run `canosr run` twice with --force and compare metrics.json.")
    if not all(ok for _, ok, _ in checks):
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="canosr")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-captures")
    p.add_argument("--dataset", default="road")
    p.set_defaults(func=cmd_build_captures)

    p = sub.add_parser("build-graphs")
    p.add_argument("--dataset", default="road")
    p.add_argument("--window", type=int, default=300)
    p.add_argument("--stride", type=int, default=300)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build_graphs)

    p = sub.add_parser("build-splits")
    p.add_argument("--dataset", default="road")
    p.add_argument("--split", default="capture", choices=["capture", "window"])
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_build_splits)

    p = sub.add_parser("run")
    p.add_argument("--config", default="configs/paper_road.yaml")
    p.add_argument("--fold", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("aggregate")
    p.add_argument("--runs", default=None)
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("gate")
    p.add_argument("--dataset", default="road")
    p.add_argument("--runs", default=None)
    p.set_defaults(func=cmd_gate)

    args = parser.parse_args(argv)
    if getattr(args, "runs", None) is None and args.command in ("aggregate", "gate"):
        args.runs = str(data_root() / "runs")
    args.func(args)


if __name__ == "__main__":
    main()
