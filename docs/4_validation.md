# Stage 4: Validation & Open-Set Calibration (`canosr/train.py`, `canosr/scoring.py`)

## 1. Periodic Validation & Checkpoint Selection

During Phase 2 (`epoch % cfg.eval.every == 0` or final epoch `epochs - 1`):

1. `_evaluate_known(model, val_loader, device)` runs closed-set inference `model.head(z).argmax(dim=1)` over validation graphs (`buckets["val"]`).
2. `closed_set_report()` computes `acc`, `macro_precision`, `macro_recall`, `macro_f1`, per-class recall, and per-class support.
3. Checkpoint selection retains the state dict maximizing `cfg.eval.select_metric` (default `macro_f1`), falling back to the final epoch only if every validation check produced `NaN`.
4. After restoring `best["state"]`, `compute_prototypes(model, eval_train_loader, n_known, device)` recomputes final class prototypes on training embeddings before validation calibration.

## 2. Two-Stage Threshold Calibration

`collect_channels(model, val_loader, device)` extracts all per-graph anomaly channels (all oriented so higher = more anomalous):
- `proto` / `distance`: `-logits.max(dim=1).values`
- `node`: per-graph mean squared node feature reconstruction error (`x[:, 1:]` when `id_in_recon=False`)
- `adj`: per-graph squared probability error `(sigmoid(adj_logits) - binary_adj)^2`
- `msp`: `-softmax(logits, dim=1).max(dim=1).values`
- `energy`: `-logsumexp(logits, dim=1)`

Calibration fits two separate thresholds at `cfg.osr.threshold_quantile = 0.95`:

1. **Stage A — Benign Fused Threshold (`threshold`)**:
   - Uses validation benign windows (`val_y == 0`, falling back to `train_y == 0` only if validation has zero benign windows).
   - Fits `ScoreNormalizer` (`mean`, `std` per channel in `osr.channels = ["proto", "node", "adj"]`) on benign validation windows only.
   - Evaluates the configured scorer (`fused` mean z-score by default) on benign calibration windows and sets `threshold = quantile(cal_fused, 0.95)`.
2. **Stage B — Known-Attack Prototype Threshold (`proto_threshold`, `proto_std`)**:
   - Uses known-attack validation windows only (`0 < val_y < n_known`, falling back to `0 < train_y < n_known` if validation has none).
   - Sets `proto_threshold = quantile(proto_cal, 0.95)` and records `proto_std = std(proto_cal) or 1.0`.
   - Restricting `proto_threshold` calibration to known attacks prevents benign-trained `node` and `adj` reconstruction errors from falsely rejecting known-attack predictions (`argmax > 0`) as unknown.
