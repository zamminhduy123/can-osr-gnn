import numpy as np
import pandas as pd
import pytest
import torch

from canosr.data import (
    FEATURE_NAMES,
    _capture_fold,
    add_gaps,
    assert_disjoint,
    build_graphs,
    attack_of,
)


def synthetic_frames(n: int = 40, attack_at: int | None = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ids = rng.choice([0x100, 0x200, 0x300], size=n).astype(np.int32)
    t = np.cumsum(rng.uniform(0.005, 0.02, size=n))
    df = pd.DataFrame({"t": t, "can_id": ids, "label": np.zeros(n, dtype=bool)})
    for i in range(8):
        df[f"b{i}"] = rng.integers(0, 256, size=n, dtype=np.uint8)
    if attack_at is not None:
        df.loc[attack_at, "label"] = True
    return add_gaps(df)


def test_attack_name_is_longest_prefix():
    assert attack_of("max_engine_coolant_temp") == "max_engine_coolant_temp"
    assert attack_of("max_speedometer") == "max_speedometer"
    assert attack_of("correlated_signal") == "correlated_signal"


def test_window_count_and_label_rule():
    df = synthetic_frames(n=40, attack_at=25)
    graphs = build_graphs(df, window=10, stride=10)
    assert len(graphs) == 4
    labels = [int(g.y.item()) for g in graphs]
    assert labels == [0, 0, 1, 0], labels


def test_features_are_finite_and_named():
    df = synthetic_frames(n=30, attack_at=None)
    graph = build_graphs(df, window=30, stride=30)[0]
    assert graph.x.shape[1] == len(FEATURE_NAMES)
    assert torch.isfinite(graph.x).all()
    assert graph.edge_index.max() < graph.num_nodes
    assert graph.edge_attr.shape[1] == 3


def test_degree_features_match_edge_list():
    df = synthetic_frames(n=30, attack_at=None)
    graph = build_graphs(df, window=30, stride=30)[0]
    out_degree = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)
    in_degree = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
    assert torch.equal(graph.x[:, 2], in_degree.float())
    assert torch.equal(graph.x[:, 3], out_degree.float())


def test_build_is_deterministic():
    df = synthetic_frames(n=30, attack_at=5)
    first = build_graphs(df, window=15, stride=15)
    second = build_graphs(df, window=15, stride=15)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert torch.equal(a.x, b.x)
        assert torch.equal(a.edge_index, b.edge_index)


def test_capture_fold_is_disjoint_and_deterministic():
    captures = []
    for attack in ("a", "b"):
        for index in range(3):
            captures.append({"id": f"fab/{attack}_{index}", "attack": attack, "kind": "attack"})
    for index in range(6):
        captures.append({"id": f"benign/{index}", "attack": None, "kind": "benign"})

    fold = _capture_fold(captures, "a", seed=1)
    assert_disjoint(fold["roles"])
    assert all(role == "test" for cid, role in fold["roles"].items() if cid.startswith("fab/a_"))
    assert fold == _capture_fold(captures, "a", seed=1)


def test_holdout_capture_never_trains():
    captures = [{"id": f"fab/mec_{i}", "attack": "mec", "kind": "attack"} for i in range(1)]
    captures += [{"id": f"fab/ms_{i}", "attack": "ms", "kind": "attack"} for i in range(3)]
    captures += [{"id": f"benign/{i}", "attack": None, "kind": "benign"} for i in range(6)]
    fold = _capture_fold(captures, "mec", seed=1)
    assert fold["roles"]["fab/mec_0"] == "test"
    assert "train" in fold["roles"].values()
