# Stage 5: Inference, Testing & Aggregation (`canosr/train.py`, `canosr/scoring.py`, `canosr/metrics.py`, `canosr/cli.py`)

## 1. Two-Stage Decision & Unified OSR Score

On `test_loader`, inference computes `test_channels`, `test_fused`, `test_logits`, and `known_pred = test_logits.argmax(dim=1)`:

1. **Predicted Label (`decide` in `canosr/scoring.py`)**:
   - If `known_pred == 0` (benign): reject to unknown (`n_known`) when `test_fused > threshold`.
   - If `known_pred > 0` (known attack): reject to unknown (`n_known`) only when `test_channels["proto"] > proto_threshold`.
   - Otherwise predict `known_pred`.
2. **Unified Threshold-Free Score (`two_stage_score`)**:
   ```text
   osr_score = (test_fused - threshold)                    if known_pred == 0
             = (proto - proto_threshold) / proto_std       if known_pred > 0
   ```
   Using `osr_score > 0.0` aligns threshold-free metrics (`AUROC`, `AUPR`, `FPR@95`, `OSCR`) and thresholded metrics (`unknown_precision`, `unknown_recall_at_threshold`) identically with `decide()`.

## 2. Evaluation Suites (`canosr/metrics.py`)

- **Open-Set Folds (`leave_one_out/*`)**:
  - `binary`: `binary_report(y_true != 0, y_pred != 0)` (`acc`, `precision`, `recall`, `f1`, confusion counts).
  - `open_set`: `open_set_report(..., scores=osr_score, threshold=0.0, known_pred=known_pred)` reporting `osr_acc`, `oscr` (`oscr_score` AUC of CCR vs FPR), `openness`, `known_acc`, `unknown_recall`, `n_known`, `n_unknown`, `auroc`, `aupr`, `fpr_at_95tpr`, `unknown_precision`, and `unknown_recall_at_threshold`.
  - `closed_set`: `closed_set_report` evaluated over known test windows (`y_true < n_known`) with `per_class_recall` and `per_class_support`.
  - `per_channel_auroc`: Individual ROC-AUC of each raw channel (`proto`, `node`, `adj`, `msp`, `energy`, `distance`) on `(y_true == n_known)`.
- **Closed-Set Fold (`closed_set`)**: Reports `closed_set` and `binary` over `known_pred`, with `oscr`, `auroc`, `aupr`, `fpr_at_95tpr`, and `unknown_recall` set to `NaN`.

## 3. Run Artifacts, Aggregation & Gates

Each `canosr run` writes `$CANOSR_DATA/runs/<run_id>/`:
- `config.yaml` (resolved config)
- `manifest.json` (`fold`, `seed`, `n_known`, `known_classes`, split counts, `split`)
- `model.pt` (`state_dict` of best validation checkpoint)
- `metrics.json` (all metrics, `threshold`, `proto_threshold`, `proto_std`, `best_val_metric`, `best_val_score`, epoch `history`)
- `predictions.parquet` (`y_true`, `y_pred`, `known_pred`, `score` (two-stage margin), `fused_score`, and raw `ch_*` columns)

CLI verification & reporting:
- `canosr aggregate --runs "$CANOSR_DATA/runs"` groups runs by `["cfg_hash", "split", "variants", "benign_source", "fold"]` into `summary.parquet`, printing `open_auroc`, `open_oscr`, `binary_f1`, `binary_recall`, `n_runs`, `n_unknown`, and 95% CIs (`ci95`).
- `canosr gate --dataset road` verifies `G1` (pytest), `G5` (`assert_disjoint` over split files), `G3` (run completeness), and `G7` (`window`-split parity on `variants == ["fab"]` LOO runs).
