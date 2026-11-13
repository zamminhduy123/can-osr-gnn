import numpy as np
import pytest

from canosr.metrics import (
    binary_report,
    closed_set_report,
    fpr_at_tpr,
    openness,
    open_set_report,
    oscr_score,
    summarise,
)


def test_binary_report_perfect():
    report = binary_report(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert report["acc"] == 1.0
    assert report["f1"] == 1.0


def test_binary_report_counts():
    report = binary_report(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 1]))
    assert (report["tp"], report["tn"], report["fp"], report["fn"]) == (1, 1, 1, 1)
    assert report["precision"] == pytest.approx(0.5)


def test_openness_matches_definition():
    assert openness(4, 1) == pytest.approx(1 - np.sqrt(2 * 4 / 5))
    assert openness(1, 1) == 0.0
    assert openness(1, 0) == pytest.approx(1 - np.sqrt(2))


def test_oscr_counts_known_and_unknown():
    y_true = np.array([0, 1, 2, 2])  # 2 is the unknown label
    y_pred = np.array([0, 1, 2, 0])
    report = open_set_report(y_true, y_pred, n_known=2, n_unknown_classes=1)
    assert report["osr_acc"] == pytest.approx(0.75)
    assert report["unknown_recall"] == pytest.approx(0.5)
    assert report["known_acc"] == 1.0
    assert np.isnan(report["oscr"])


def test_fpr_at_95tpr_is_monotone():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(100)]
    scores = np.r_[rng.normal(0, 1, 100), rng.normal(2, 1, 100)]
    value = fpr_at_tpr(y, scores, 0.95)
    assert 0.0 <= value <= 1.0


def test_closed_set_per_class_recall():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    report = closed_set_report(y_true, y_pred, n_classes=2)
    assert report["per_class_recall"]["recall_0"] == 0.5
    assert report["per_class_recall"]["recall_1"] == 1.0
    assert report["per_class_support"] == {"n_0": 2, "n_1": 2}
    assert report["macro_precision"] == pytest.approx(5.0 / 6.0)
    assert report["macro_f1"] == pytest.approx(11.0 / 15.0)


def test_oscr_score_separates_known_and_unknown():
    y_true = np.array([0, 1, 2, 2])
    known_pred = np.array([0, 1, 0, 1])
    assert oscr_score(y_true, known_pred, np.array([0.1, 0.2, 0.8, 0.9]), n_known=2) == pytest.approx(1.0)
    assert oscr_score(y_true, known_pred, np.array([0.8, 0.9, 0.1, 0.2]), n_known=2) == pytest.approx(0.0)
    report = open_set_report(
        y_true,
        np.array([0, 1, 2, 0]),
        n_known=2,
        n_unknown_classes=1,
        scores=np.array([0.1, 0.2, 0.8, 0.9]),
        known_pred=known_pred,
    )
    assert report["oscr"] == pytest.approx(1.0)
    assert report["osr_acc"] == pytest.approx(0.75)


def test_summarise_ignores_nan():
    rows = [{"x": 1.0}, {"x": 3.0}, {"x": float("nan")}]
    out = summarise(rows)
    assert out["x"]["mean"] == 2.0
    assert out["x"]["n"] == 2
