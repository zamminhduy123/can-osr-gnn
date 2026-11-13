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
