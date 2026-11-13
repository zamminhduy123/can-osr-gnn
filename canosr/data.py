"""ROAD frame parsing, capture discovery, graph construction, splits, graph cache.

The graph definition matches the submitted pipeline (see docs/PIPELINE_V2.md §4.3),
with two deliberate fixes recorded in that document:

* ``payload_entropy_std`` is computed over the 64 payload bits. The submitted code
  clipped *byte values* into [0, 1], which makes the feature collapse to ~0.
* feature 0 is the decimal CAN ID, not a per-dataset dense index.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.data import Data

FEATURE_VERSION = "v2"

FEATURE_NAMES = [
    "can_id",
    "frequency",
    "in_degree",
    "out_degree",
    "payload_variance",
    "max_burst",
    "payload_entropy_std",
    "unique_payload_count",
    "payload_delta_norm",
    "periodicity_score",
    "skew_gap",
    "mean_gap",
    "std_gap",
]

ATTACK_NAMES = [
    "max_engine_coolant_temp",
    "max_speedometer",
    "reverse_light_off",
    "reverse_light_on",
    "correlated_signal",
    "fuzzing",
    "accelerator",
]

_CSV_RE = re.compile(
    r"^(?P<attack>.+?)_attack(?:_(?P<n>\d+))?(?:_(?P<variant>masquerade))?_dataset\.csv$"
)
_LOG_RE = re.compile(r"^\((?P<t>[0-9.]+)\)\s+(?P<bus>\S+)\s+(?P<id>[0-9A-Fa-f]+)#(?P<data>[0-9A-Fa-f]*)")


# --------------------------------------------------------------------------- frames


def _payload_from_hex(strings: pd.Series) -> np.ndarray:
    """Hex payload strings -> uint8 [n, 8], right-padded with zeros."""
    values = strings.fillna("").astype(str).str.strip().str.upper().str.ljust(16, "0")
    bad = values.str.len() != 16
    if bad.any():
        raise ValueError(f"{int(bad.sum())} payloads are not 16 hex characters")
    raw = bytes.fromhex("".join(values.tolist()))
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8)


def load_csv_frames(path: str | Path) -> pd.DataFrame:
    """Read a ROAD processed CSV (time, aid, data, time_diffs, label)."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = {"time", "aid", "data", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    payload = _payload_from_hex(df["data"])
    out = pd.DataFrame(
        {
            "t": df["time"].to_numpy(dtype=np.float64),
            "can_id": df["aid"].to_numpy(dtype=np.int32),
            "label": df["label"].astype(str).str.lower().isin(["true", "1"]).to_numpy(),
        }
    )
    for i in range(8):
        out[f"b{i}"] = payload[:, i]
    # the processed CSVs are grouped by CAN ID; the submitted pipeline sorts by time
    return out.sort_values("t", kind="stable").reset_index(drop=True)


def parse_raw_log(path: str | Path) -> tuple[pd.DataFrame, int]:
    """Parse a ROAD ``.log`` file. Returns frames and the count of skipped lines."""
    times: list[float] = []
    ids: list[int] = []
    payloads: list[int] = []
    skipped = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = _LOG_RE.match(line)
            if match is None:
                if line.strip():
                    skipped += 1
                continue
            data = match.group("data")
            if len(data) < 16:
                data = data.ljust(16, "0")
            elif len(data) > 16:
                data = data[:16]
            times.append(float(match.group("t")))
            ids.append(int(match.group("id"), 16))
            payloads.append(int(data, 16))
    payload = np.frombuffer(
        b"".join(v.to_bytes(8, "big") for v in payloads), dtype=np.uint8
    ).reshape(-1, 8)
    df = pd.DataFrame(
        {
            "t": np.asarray(times, dtype=np.float64),
            "can_id": np.asarray(ids, dtype=np.int32),
            "label": np.zeros(len(times), dtype=bool),
        }
    )
    for i in range(8):
        df[f"b{i}"] = payload[:, i]
    return df.sort_values("t", kind="stable").reset_index(drop=True), skipped


def add_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Gap to the previous frame with the same CAN ID, computed per capture."""
    df = df.copy()
    df["gap"] = df.groupby("can_id", sort=False)["t"].diff().fillna(0.0)
    return df


# ------------------------------------------------------------------------ captures


def attack_of(stem: str) -> str | None:
    for name in ATTACK_NAMES:
        if stem.startswith(name):
            return name
    return None


def discover_captures(raw_root: str | Path, processed_root: str | Path) -> list[dict]:
    """Enumerate ROAD captures from processed CSVs plus raw ambient logs."""
    raw_root, processed_root = Path(raw_root), Path(processed_root)
    captures: list[dict] = []

    for folder, variant in ((processed_root / "fab_dataset", "fab"), (processed_root / "mar_dataset", "mas")):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.csv")):
            match = _CSV_RE.match(path.name)
            if match is None:
                continue
            attack = attack_of(match.group("attack"))
            if attack is None:
                raise ValueError(f"{path.name}: unrecognised attack name")
            captures.append(
                {
                    "id": f"{variant}/{path.stem.replace('_dataset', '')}",
                    "variant": variant,
                    "kind": "attack",
                    "attack": attack,
                    "path": str(path),
                    "format": "csv",
                    "bytes": path.stat().st_size,
                }
            )

    ambient = raw_root / "ambient"
    if ambient.is_dir():
        for path in sorted(ambient.glob("*.log")):
            captures.append(
                {
                    "id": f"benign/{path.stem}",
                    "variant": "ambient",
                    "kind": "benign",
                    "attack": None,
                    "path": str(path),
                    "format": "log",
                    "bytes": path.stat().st_size,
                }
            )

    captures.sort(key=lambda c: c["id"])
    if not captures:
        raise FileNotFoundError(f"no captures under {processed_root} or {ambient}")
    return captures


def captures_fingerprint(captures: Iterable[dict]) -> str:
    blob = json.dumps(
        sorted((c["id"], c["path"], c["bytes"]) for c in captures), sort_keys=True
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build_captures(
    dataset: str,
    data_root: str | Path,
    processed_root: str | Path | None = None,
    raw_root: str | Path | None = None,
) -> dict:
    data_root = Path(data_root)
    raw_root = raw_root or data_root / "raw" / dataset
    processed_root = processed_root or data_root / "raw" / dataset
    captures = discover_captures(raw_root, processed_root)
    payload = {
        "dataset": dataset,
        "feature_version": FEATURE_VERSION,
        "fingerprint": captures_fingerprint(captures),
        "captures": captures,
    }
    out = data_root / "processed" / dataset / "captures.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    return payload


def load_captures(dataset: str, data_root: str | Path) -> dict:
    path = Path(data_root) / "processed" / dataset / "captures.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# -------------------------------------------------------------------------- graphs


def _payload_array(df: pd.DataFrame) -> np.ndarray:
    return df[[f"b{i}" for i in range(8)]].to_numpy(dtype=np.float32)


def _entropy_per_message(payload: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(payload.astype(np.uint8), axis=1)
    p = np.clip(bits.mean(axis=1), 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _edge_tables(inv: np.ndarray, t: np.ndarray):
    """Deduplicated transitions with count/mean/std of destination gaps."""
    if len(inv) < 2:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    src, dst = inv[:-1], inv[1:]
    order = np.lexsort((t[1:], dst, src))
    src, dst, ts = src[order], dst[order], t[1:][order]

    new_group = np.ones(len(src), dtype=bool)
    new_group[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
    group = np.cumsum(new_group) - 1
    count = np.bincount(group).astype(np.float64)

    dt = np.diff(ts)
    valid = ~new_group[1:]
    sum_dt = np.bincount(group[1:][valid], weights=dt[valid], minlength=len(count))
    sum_dt2 = np.bincount(group[1:][valid], weights=dt[valid] ** 2, minlength=len(count))

    mean = np.zeros_like(count)
    std = np.zeros_like(count)
    many = count > 1
    mean[many] = sum_dt[many] / (count[many] - 1)
    var = np.zeros_like(count)
    var[many] = np.maximum(sum_dt2[many] / (count[many] - 1) - mean[many] ** 2, 0.0)
    std[many] = np.sqrt(var[many])

    edge_index = np.stack([src[new_group], dst[new_group]], axis=0)
    edge_attr = np.stack([count, mean, std], axis=1).astype(np.float32)
    return edge_index, edge_attr


def _node_features(
    can_id: np.ndarray,
    t: np.ndarray,
    payload: np.ndarray,
    gap: np.ndarray,
    uids: np.ndarray,
) -> np.ndarray:
    """Per-node features. ``can_id``/``t``/``payload``/``gap`` are per message."""
    n_nodes = len(uids)
    order = np.argsort(can_id, kind="stable")
    ids_sorted = can_id[order]
    t_sorted = t[order]
    p_sorted = payload[order]
    g_sorted = gap[order]

    starts = np.searchsorted(ids_sorted, uids, side="left")
    ends = np.searchsorted(ids_sorted, uids, side="right")

    feats = np.zeros((n_nodes, len(FEATURE_NAMES)), dtype=np.float32)
    for node in range(n_nodes):
        lo, hi = starts[node], ends[node]
        if hi <= lo:
            continue
        times = t_sorted[lo:hi]
        values = p_sorted[lo:hi]
        gaps = g_sorted[lo:hi]
        frequency = hi - lo

        payload_variance = float(values.var())
        payload_delta_norm = (
            float(np.linalg.norm(np.diff(values, axis=0))) if frequency > 1 else 0.0
        )

        max_burst = 0.0
        if frequency >= 10:
            durations = times[9:] - times[:-9]
            ok = durations > 1e-9
            if ok.any():
                max_burst = float((10.0 / durations[ok]).max())

        entropy = _entropy_per_message(values)
        entropy_std = float(entropy.std()) if len(entropy) else 0.0

        unique_payloads = int(np.unique(values, axis=0).shape[0])

        mean_gap = std_gap = skew_gap = periodicity = 0.0
        if frequency > 1:
            mean_gap = float(gaps.mean())
            std_gap = float(gaps.std())
            if frequency > 2 and np.ptp(gaps) > 1e-9:
                skew_gap = float(stats.skew(gaps, bias=False))
            if frequency > 2 and mean_gap > 1e-9:
                periodicity = float(np.log1p(frequency) / (std_gap / mean_gap + 1e-6))

        feats[node] = [
            float(uids[node]),
            float(frequency),
            0.0,  # in_degree, filled by caller
            0.0,  # out_degree, filled by caller
            payload_variance,
            max_burst,
            entropy_std,
            float(unique_payloads),
            payload_delta_norm,
            periodicity,
            skew_gap,
            mean_gap,
            std_gap,
        ]
    return feats


def _window_graph(
    t: np.ndarray,
    can_id: np.ndarray,
    payload: np.ndarray,
    label: np.ndarray,
    gap: np.ndarray,
) -> Data | None:
    uids, inv = np.unique(can_id, return_inverse=True)
    n_nodes = len(uids)
    if n_nodes == 0:
        return None

    edge_index, edge_attr = _edge_tables(inv, t)
    feats = _node_features(can_id, t, payload, gap, uids)
    if edge_index.shape[1]:
        feats[:, 2] = np.bincount(edge_index[1], minlength=n_nodes)
        feats[:, 3] = np.bincount(edge_index[0], minlength=n_nodes)

    return Data(
        x=torch.from_numpy(feats),
        edge_index=torch.from_numpy(edge_index.astype(np.int64)),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.tensor([float(label.max())]),
        t_start=float(t[0]),
        t_end=float(t[-1]),
    )


def build_graphs(
    df: pd.DataFrame,
    window: int = 300,
    stride: int = 300,
    limit: int | None = None,
) -> list[Data]:
    t = df["t"].to_numpy(dtype=np.float64)
    can_id = df["can_id"].to_numpy(dtype=np.int32)
    label = df["label"].to_numpy()
    gap = df["gap"].to_numpy(dtype=np.float64) if "gap" in df else np.zeros_like(t)
    payload = _payload_array(df)

    graphs: list[Data] = []
    for start in range(0, max(len(df) - window + 1, 0), stride):
        stop = start + window
        graph = _window_graph(
            t[start:stop], can_id[start:stop], payload[start:stop], label[start:stop], gap[start:stop]
        )
        if graph is not None:
            graphs.append(graph)
        if limit is not None and len(graphs) >= limit:
            break
    return graphs


def graph_cache_dir(data_root: str | Path, dataset: str, window: int, stride: int) -> Path:
    return Path(data_root) / "processed" / dataset / "graphs" / f"ws{window}_st{stride}"


def build_graph_cache(
    dataset: str,
    data_root: str | Path,
    window: int = 300,
    stride: int = 300,
    force: bool = False,
    limit: int | None = None,
) -> dict:
    data_root = Path(data_root)
    meta = load_captures(dataset, data_root)
    out_dir = graph_cache_dir(data_root, dataset, window, stride)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"built": 0, "skipped": 0, "graphs": 0}
    for capture in meta["captures"]:
        target = out_dir / f"{capture['id'].replace('/', '__')}.pt"
        if target.exists() and not force:
            summary["skipped"] += 1
            continue
        df = load_csv_frames(capture["path"]) if capture["format"] == "csv" else parse_raw_log(capture["path"])[0]
        df = add_gaps(df)
        graphs = build_graphs(df, window=window, stride=stride, limit=limit)
        torch.save(
            {
                "capture_id": capture["id"],
                "window": window,
                "stride": stride,
                "feature_version": FEATURE_VERSION,
                "fingerprint": meta["fingerprint"],
                "graphs": graphs,
            },
            target,
        )
        summary["built"] += 1
        summary["graphs"] += len(graphs)
    return summary


class GraphStore:
    """Lazy access to cached graphs, tagged with their capture id."""

    def __init__(self, data_root: str | Path, dataset: str, window: int = 300, stride: int = 300):
        self.dir = graph_cache_dir(data_root, dataset, window, stride)
        self.dataset = dataset
        self.window = window
        self.stride = stride
        self.fingerprint = load_captures(dataset, data_root)["fingerprint"]

    def _path(self, capture_id: str) -> Path:
        return self.dir / f"{capture_id.replace('/', '__')}.pt"

    def has(self, capture_id: str) -> bool:
        return self._path(capture_id).exists()

    def load_capture(self, capture_id: str) -> list[Data]:
        payload = torch.load(self._path(capture_id), weights_only=False)
        if payload.get("feature_version") != FEATURE_VERSION:
            raise RuntimeError(
                f"{capture_id}: cached feature_version {payload.get('feature_version')!r} "
                f"!= {FEATURE_VERSION!r}; rerun build-graphs --force"
            )
        if payload.get("fingerprint") != self.fingerprint:
            raise RuntimeError(
                f"{capture_id}: capture fingerprint changed; rerun build-captures and build-graphs --force"
            )
        graphs = payload["graphs"]
        for graph in graphs:
            graph.capture_id = capture_id
        return graphs

    def load_many(self, capture_ids: Iterable[str]) -> list[Data]:
        graphs: list[Data] = []
        for capture_id in capture_ids:
            graphs.extend(self.load_capture(capture_id))
        return graphs


# -------------------------------------------------------------------------- splits


def recording_id(capture_id: str) -> str:
    """Strip variant prefix and ``_masquerade`` suffix to get the underlying recording id."""
    base = capture_id.split("/", 1)[1] if "/" in capture_id else capture_id
    return base.removesuffix("_masquerade")


def _role_split(captures: list[dict], seed: int) -> dict[str, str]:
    """60/20/20 by capture, guaranteeing every role is non-empty when possible."""
    ordered = sorted(captures, key=lambda c: c["id"])
    n = len(ordered)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    if n == 1:
        roles = ["train"]
    elif n == 2:
        roles = ["train", "test"]
    else:
        n_val = max(1, round(0.2 * n))
        n_test = max(1, round(0.2 * n))
        n_train = n - n_val - n_test
        roles = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return {ordered[i]["id"]: roles[rank] for rank, i in enumerate(order)}


def build_splits(dataset: str, data_root: str | Path, split: str = "capture", seed: int = 1) -> dict:
    data_root = Path(data_root)
    meta = load_captures(dataset, data_root)
    captures = meta["captures"]
    attacks = sorted({c["attack"] for c in captures if c["kind"] == "attack"})

    if split == "window":
        folds = {
            "closed_set": {"mode": "window", "seed": seed, "ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
            **{
                f"leave_one_out/{attack}": {
                    "mode": "window",
                    "seed": seed,
                    "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                    "holdout": attack,
                }
                for attack in attacks
            },
        }
    elif split == "capture":
        folds = {"closed_set": _capture_fold(captures, None, seed)}
        for attack in attacks:
            folds[f"leave_one_out/{attack}"] = _capture_fold(captures, attack, seed)
    else:
        raise ValueError(f"unknown split protocol {split!r}")

    payload = {"dataset": dataset, "split": split, "seed": seed, "folds": folds}
    out = data_root / "processed" / dataset / "splits" / f"{split}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    return payload


def _capture_fold(captures: list[dict], holdout: str | None, seed: int) -> dict:
    roles: dict[str, str] = {}
    if holdout is not None:
        for capture in captures:
            if capture["attack"] == holdout:
                roles[capture["id"]] = "test"

    groups: dict[str, list[dict]] = defaultdict(list)
    for capture in captures:
        if capture["id"] in roles:
            continue
        key = capture["attack"] if capture["attack"] else "__benign__"
        groups[key].append(capture)

    for key in sorted(groups):
        by_rec: dict[str, list[dict]] = defaultdict(list)
        for capture in groups[key]:
            by_rec[recording_id(capture["id"])].append(capture)
        rec_roles = _role_split([{"id": rec} for rec in sorted(by_rec)], seed + _stable_int(key))
        for rec, role in rec_roles.items():
            for capture in by_rec[rec]:
                roles[capture["id"]] = role

    return {"mode": "capture", "holdout": holdout, "roles": dict(sorted(roles.items()))}


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def load_splits(dataset: str, data_root: str | Path, split: str, seed: int) -> dict:
    path = Path(data_root) / "processed" / dataset / "splits" / f"{split}_seed{seed}.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_disjoint(roles: dict[str, str]) -> None:
    """Hard failure if any role is invalid or any underlying recording appears in multiple roles."""
    valid_roles = {"train", "val", "test"}
    sets: dict[str, set[str]] = {r: set() for r in valid_roles}
    for capture_id, role in roles.items():
        if role not in valid_roles:
            raise AssertionError(f"capture {capture_id} has invalid role {role!r}")
        sets[role].add(recording_id(capture_id))
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise AssertionError(f"{left}/{right} recording overlap: {sorted(overlap)[:5]}")


def iter_windows(df: pd.DataFrame, window: int, stride: int) -> Iterator[pd.DataFrame]:
    for start in range(0, max(len(df) - window + 1, 0), stride):
        yield df.iloc[start : start + window]
