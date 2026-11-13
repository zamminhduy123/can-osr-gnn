import torch

from canosr.scoring import ScoreNormalizer, decide, threshold_from_quantile


def test_normalizer_is_zero_mean_unit_std_on_fit_data():
    channels = {"proto": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    normalizer = ScoreNormalizer.fit(channels, ["proto"])
    fused = normalizer.transform(channels, ["proto"])
    assert abs(float(fused.mean())) < 1e-6
    assert abs(float(fused.std(unbiased=False)) - 1.0) < 1e-6


def test_fusion_is_mean_of_z_scores():
    channels = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([2.0, 1.0])}
    normalizer = ScoreNormalizer(mean={"a": 0.0, "b": 0.0}, std={"a": 1.0, "b": 1.0})
    fused = normalizer.transform(channels, ["a", "b"])
    assert torch.allclose(fused, torch.tensor([1.5, 1.5]))


def test_threshold_is_benign_quantile():
    scores = torch.arange(0, 101, dtype=torch.float32)
    assert abs(threshold_from_quantile(scores, 0.95) - 95.0) <= 1.0


def test_decide_assigns_unknown_label():
    scores = torch.tensor([0.1, 0.9])
    logits = torch.tensor([[-0.1, -0.5], [-0.2, -0.3]])
    pred = decide(scores, threshold=0.5, logits=logits, n_known=2)
    assert pred.tolist() == [0, 2]


def test_decide_two_stage_preserves_known_attacks():
    scores = torch.tensor([0.9, 5.0, 5.0])  # all exceed benign threshold=0.5
    logits = torch.tensor([
        [2.0, -1.0],   # predicted benign (known=0) -> uses fused threshold -> unknown (2)
        [-1.0, 2.0],   # predicted known attack (known=1), low proto -> stays 1
        [-1.0, 2.0],   # predicted known attack (known=1), high proto -> unknown (2)
    ])
    proto_scores = torch.tensor([0.1, 0.2, 0.9])
    pred = decide(scores, threshold=0.5, logits=logits, n_known=2, proto_scores=proto_scores, proto_threshold=0.5)
    assert pred.tolist() == [2, 1, 2]


def test_config_hash_includes_run_limit():
    from canosr.config import config_hash, load_config

    full = load_config("configs/paper_road.yaml")
    smoke = load_config("configs/paper_road.yaml", overrides={"run": {"limit": 50}})
    forced = load_config("configs/paper_road.yaml", overrides={"run": {"force": True, "device": "cpu"}})
    relocated = load_config("configs/paper_road.yaml", overrides={"data": {"root": "/tmp/other-root"}})
    assert config_hash(full) != config_hash(smoke)
    assert config_hash(full) == config_hash(forced)
    assert config_hash(full) == config_hash(relocated)


def test_two_stage_score_matches_decide():
    from sklearn.metrics import roc_auc_score
    from canosr.scoring import two_stage_score

    # benign (known=0): low fused, high fused (false alarm/unknown);
    # known attack (known=1): high fused (recon error) but low proto distance;
    # unknown (y_true=2): high proto distance
    fused = torch.tensor([0.1, 2.0, 4.0, 2.5])
    logits = torch.tensor([
        [3.0, -1.0],
        [3.0, -1.0],
        [-1.0, 3.0],
        [-1.0, 2.0],
    ])
    proto = torch.tensor([0.1, 0.2, 0.2, 1.5])
    osr_score = two_stage_score(fused, threshold=0.5, logits=logits, proto_scores=proto, proto_threshold=0.5, proto_std=0.25)
    pred = decide(fused, threshold=0.5, logits=logits, n_known=2, proto_scores=proto, proto_threshold=0.5)
    assert (osr_score > 0.0).tolist() == (pred == 2).tolist()
    y_unk = [0, 0, 0, 1]
    assert roc_auc_score(y_unk, osr_score.numpy()) > roc_auc_score(y_unk, fused.numpy())


def test_supcon_loss_half_precision_does_not_overflow():
    from canosr.losses import supcon_loss

    feats = torch.randn(8, 4, dtype=torch.float16)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    loss = supcon_loss(feats, labels)
    assert torch.isfinite(loss)
