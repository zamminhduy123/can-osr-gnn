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
    load_csv_frames,
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


def test_node_features_use_real_can_ids():
    df = synthetic_frames(n=60, attack_at=None)
    graph = build_graphs(df, window=60, stride=60)[0]
    assert set(graph.x[:, 0].tolist()) == {0x100, 0x200, 0x300}
    assert float(graph.x[:, 1].sum()) == 60.0  # one row per message
    assert float(graph.x[:, 4:].abs().sum()) > 0  # payload/timing features populated


def test_loader_sorts_grouped_csv_by_time(tmp_path):
    rows = []
    for cid in (0x100, 0x200):
        for i in range(20):
            rows.append((i * 0.01 + (0 if cid == 0x100 else 0.005), cid, "00" * 8, 0.0, False))
    frame = pd.DataFrame(rows, columns=["time", "aid", "data", "time_diffs", "label"])
    path = tmp_path / "grouped_dataset.csv"
    frame.to_csv(path, index=False)
    loaded = load_csv_frames(path)
    assert loaded["t"].is_monotonic_increasing
    assert set(loaded["can_id"].tolist()) == {0x100, 0x200}


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


def test_relabel_maps_holdout_attack_to_n_known():
    from torch_geometric.data import Data
    from canosr.train import _relabel

    class_of_attack = {"fuzzing": 1, "accelerator": 2}
    n_known = 3

    known_graphs = [Data(y=torch.tensor([0.0])), Data(y=torch.tensor([1.0]))]
    _relabel(known_graphs, {"kind": "attack", "attack": "fuzzing"}, class_of_attack, n_known)
    assert [g.y.item() for g in known_graphs] == [0.0, 1.0]

    holdout_graphs = [Data(y=torch.tensor([0.0])), Data(y=torch.tensor([1.0]))]
    _relabel(holdout_graphs, {"kind": "attack", "attack": "correlated_signal"}, class_of_attack, n_known)
    assert [g.y.item() for g in holdout_graphs] == [0.0, 3.0]

    benign_graphs = [Data(y=torch.tensor([0.0])), Data(y=torch.tensor([1.0]))]
    _relabel(benign_graphs, {"kind": "benign", "attack": None}, class_of_attack, n_known)
    assert [g.y.item() for g in benign_graphs] == [0.0, 0.0]


def test_fab_and_mas_share_role_in_capture_fold():
    import pytest
    from canosr.data import recording_id

    captures = []
    for index in range(1, 6):
        captures.append({"id": f"fab/x_{index}", "attack": "x", "kind": "attack", "variant": "fab"})
        captures.append({"id": f"mas/x_{index}_masquerade", "attack": "x", "kind": "attack", "variant": "mas"})
    for index in range(3):
        captures.append({"id": f"ambient/{index}", "attack": None, "kind": "benign", "variant": "ambient"})

    assert recording_id("mas/x_1_masquerade") == "x_1"
    for seed in (1, 2, 3):
        fold = _capture_fold(captures, holdout=None, seed=seed)
        assert_disjoint(fold["roles"])
        for index in range(1, 6):
            assert fold["roles"][f"fab/x_{index}"] == fold["roles"][f"mas/x_{index}_masquerade"]

    with pytest.raises(AssertionError):
        assert_disjoint({"fab/x_1": "train", "mas/x_1_masquerade": "test"})


def test_contiguous_split_separates_single_recording():
    from torch_geometric.data import Data
    from canosr.train import _contiguous_split, _split_single_recording_group

    graphs = [Data(y=torch.tensor([1.0]), t=torch.tensor([i])) for i in range(10)]
    buckets = {"train": [], "val": [], "test": []}
    _contiguous_split(graphs, buckets)
    assert [int(g.t.item()) for g in buckets["train"]] == [0, 1, 2, 3, 4, 5]
    assert [int(g.t.item()) for g in buckets["val"]] == [6, 7]
    assert [int(g.t.item()) for g in buckets["test"]] == [8, 9]

    fab = [Data(y=torch.tensor([1.0]), t_start=float(i), t_end=float(i + 0.9)) for i in range(10)]
    # Mas shifted by +0.5 at window 5->6 boundary so it straddles t_cut1=5.9
    mas = [Data(y=torch.tensor([1.0]), t_start=float(i) + 0.5, t_end=float(i + 1.4)) for i in range(9)]
    ts_buckets = {"train": [], "val": [], "test": []}
    _split_single_recording_group([fab, mas], ts_buckets)
    max_train = max(float(g.t_end) for g in ts_buckets["train"])
    min_val = min(float(g.t_start) for g in ts_buckets["val"])
    max_val = max(float(g.t_end) for g in ts_buckets["val"])
    min_test = min(float(g.t_start) for g in ts_buckets["test"])
    assert max_train <= min_val
    assert max_val <= min_test
