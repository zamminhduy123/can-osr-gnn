# Changelog

All notable changes and fixes across pipeline iterations live here.

## v2.2 — Round-2 Metric, Calibration & AMP Fixes

- **Unified two-stage OSR score (`canosr/scoring.py`, `canosr/train.py`)**: Added `two_stage_score()` (`fused - threshold` when `known == 0`; `(proto - proto_threshold) / proto_std` when `known > 0`) and passed it to `open_set_report(..., scores=scores, threshold=0.0)` so AUROC, AUPR, FPR@95, OSCR, and `unknown_recall_at_threshold` match `decide()`. Stored both `score` and `fused_score` in `predictions.parquet`.
- **Attack-only `proto_threshold` calibration (`canosr/train.py`)**: Calibrated `proto_threshold` and `proto_std` strictly on known-attack validation windows `(val_y > 0) & (val_y < n_known)`.
- **AMP float16 overflow fix (`canosr/losses.py`)**: Normalized `supcon_loss` features in `float32` (`F.normalize(features.float(), dim=1)`) so `masked_fill(self_mask, -1e9)` succeeds under CUDA AMP.
- **Shared-timestamp MECT cut (`canosr/data.py`, `canosr/train.py`)**: Attached `t_start` and `t_end` to cached `Data` windows and added `_split_single_recording_group()` to cut `fab` and `mas` copies at shared timestamps while dropping boundary-straddling windows.
- **Closed-set OSCR, support & CI95 (`canosr/metrics.py`, `canosr/train.py`, `canosr/cli.py`, `canosr/config.py`)**: Set closed-set `oscr` to `NaN`, added `per_class_support` to `closed_set_report()` and `ci95` to `summarise()`, stripped `data.root` from `config_hash()`, grouped `canosr aggregate` by `["cfg_hash", "split", "variants", "benign_source", "fold"]` with `n_runs`, `n_unknown`, and `ci95`, and added step-by-step stage docs under `docs/`.

## v2.1 — Audit & Leakage Fixes

### Leakage & Split Integrity
- **Unified `fab` ↔ `mas` recording assignment (`canosr/data.py`)**: Added `recording_id()` to strip `<variant>/` and `_masquerade` before `_capture_fold` splits recordings, guaranteeing matching `fab/*` and `mas/*_masquerade` captures always land in the same role.
- **Recording-level `assert_disjoint` (`canosr/data.py`, `canosr/cli.py`)**: Validated role names and checked recording-level overlap across `train`, `val`, and `test` inside `assert_disjoint()` and gate `G5`.
- **Single-recording attack temporal split (`canosr/train.py`)**: Added `_contiguous_split()` (chronological 60/20/20 over benign and attack subsets separately) in `load_role_graphs()` so single-recording known attacks such as `max_engine_coolant_temp` appear in validation and test metrics without fab/mas overlap.
- **Holdout validation (`canosr/train.py`)**: `resolve_fold()` now raises `ValueError` if a fold's `holdout` attack is absent from the selected `variants` (for example, `leave_one_out/fuzzing` under `variants: [mas]`).
- **Ambient benign filtering (`canosr/train.py`)**: Applied `cfg.data.variants` filtering only to `kind == "attack"`, so `benign_source: with_ambient` retains ambient captures even when `variants` is `[fab, mas]`.
- **Window split RNG (`canosr/train.py`)**: Seeded `window`-protocol splitting with `split_seed` rather than training `seed`.

### Calibration, Scoring & Config Wiring
- **Two-stage unknown rejection (`canosr/scoring.py`, `canosr/train.py`)**: Predictions already classified as known attacks (`known > 0`) are rejected as unknown (`n_known`) only when `proto > proto_threshold` (calibrated on all known validation graphs), preventing benign-trained reconstruction errors (`node`, `adj`) from rejecting known attacks as unknown. Benign predictions (`known == 0`) continue using the fused score against the benign validation threshold.
- **Channel polarity alignment (`canosr/scoring.py`)**: Negated `msp` and `energy` inside `channel_scores()` so every channel follows higher = more anomalous for `per_channel_auroc` and fusion.
- **Run hashing (`canosr/config.py`)**: Excluded only `device`, `force`, and `root` inside `config_hash()`, preserving `run.limit` and `run.amp` so smoke runs never collide with full runs.
- **Wired config knobs (`canosr/models.py`, `canosr/train.py`)**: Added `PoolReadout` (`add` / `mean`) for `model.readout`, `_make_optimizer()` for `train.optimizer` (`adamw`, `adam`, `sgd`), CUDA AMP via `run.amp`, and `skipped_nonfinite` batch tracking in epoch history.
- **Aggregation & Gate G7 (`canosr/cli.py`, `canosr/train.py`)**: Grouped `canosr aggregate` by `["split", "variants", "fold"]`, limited Gate G7 `leave_one_out/*` parity checks to `variants == ["fab"]`, and recorded `best_val_metric` / `best_val_score` plus `known_pred` in `predictions.parquet`.

## v2.0 — Rewrite from Submitted Code

Summary of structural defects resolved relative to the original monolithic training script:

| Problem in the submitted code | Fix in v2 |
|---|---|
| Random window split leaks near-duplicate windows | Capture-disjoint splits by default, asserted at load time |
| Class index from `os.listdir` order | Manifest with explicit, sorted class IDs |
| Adjacency loss returns plain MSE, omega unused | Binary target + `pos_weight` BCE, single code path |
| SVDD and repulsion dead code | Removed |
| Model selected on test F1 | Model selection on validation metrics (`macro_recall` / `macro_f1`) |
| Four OR-ed flags as the OSR rule | Calibrated fused score + quantile threshold |
| `open_f1` guard ends in `or True` | Removed |
| Edge timing features silently dropped | All edge attributes (`can_id`, normalized `dt_id`, `dt_global`) used; CSV rows time-sorted (`v2` cache) |
| One model class copied across variants | Unified `GraphAE` with five pluggable registries |
