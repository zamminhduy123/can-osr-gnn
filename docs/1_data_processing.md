# Stage 1: Data Processing (`canosr/data.py`)

Pipeline entrypoints: `canosr build-captures`, `canosr build-graphs`, `canosr build-splits`.

## 1. Frame Ingestion & Timing

1. **Processed attack CSVs (`load_csv_frames`)**: Reads `time`, `aid`, `data`, `label` from `fab_dataset/` and `mar_dataset/`. Left-pads hex payloads into 8 byte columns `b0..b7` (`uint8`) and sorts frames chronologically (`df.sort_values("t", kind="stable")`) because processed CSVs arrive grouped by CAN ID.
2. **Raw ambient logs (`parse_raw_log`)**: Matches `(timestamp) bus ID#HEXDATA`, parses 64-bit hex payloads into `b0..b7`, assigns `label = False`, sorts by `t`, and tracks non-empty malformed line counts (`skipped`).
3. **Inter-frame gaps (`add_gaps`)**: Computes per-CAN-ID inter-arrival delta `df.groupby("can_id", sort=False)["t"].diff().fillna(0.0)` inside each capture.

## 2. Capture Discovery & Fingerprinting

- `discover_captures()` scans:
  - `fab_dataset/*.csv` → `variant="fab"`, `kind="attack"`, `id="fab/<stem>"`
  - `mar_dataset/*.csv` → `variant="mas"`, `kind="attack"`, `id="mas/<stem>"`
  - `ambient/*.log` → `variant="ambient"`, `kind="benign"`, `id="benign/<stem>"`
- `build_captures()` writes `$CANOSR_DATA/processed/<dataset>/captures.json` tagged with `FEATURE_VERSION = "v2"` and a SHA-256 `fingerprint` over `(id, path, bytes)`.

## 3. Window Graph Construction (`build_graphs`, `_window_graph`)

Sliding windows of `window=300` frames at `stride=300` produce one PyG `Data` graph per window:

- **Nodes (`x`, shape `[n_nodes, 13]`)**: One node per unique CAN ID (`uids`) in the window, with `FEATURE_NAMES`:
  1. `can_id` (decimal CAN ID)
  2. `frequency` (message count in window)
  3. `in_degree`, 4. `out_degree` (computed from deduplicated transition edges)
  5. `payload_variance`, 6. `max_burst` (`10 / min_duration_10_msgs`), 7. `payload_entropy_std` (bit-level Shannon entropy std over 64 bits), 8. `unique_payload_count`, 9. `payload_delta_norm`
  10. `periodicity_score` (`log1p(frequency) / (std_gap / mean_gap + 1e-6)`), 11. `skew_gap`, 12. `mean_gap`, 13. `std_gap`
- **Edges (`edge_index` `[2, E]`, `edge_attr` `[E, 3]`)**: Sequential frame transitions `inv[i] -> inv[i+1]`, deduplicated per `(src, dst)` pair with `[count, mean_dt, std_dt]`.
- **Window Label & Bounds**: `y = float(label.max())` (`0.0` benign, `1.0` attack window), plus `t_start = float(t[0])` and `t_end = float(t[-1])` for leakage-free temporal cuts.
- **Graph Cache (`build_graph_cache`, `GraphStore`)**: Saved under `processed/<dataset>/graphs/ws<window>_st<stride>/<capture_id>.pt`; `GraphStore.load_capture()` enforces matching `feature_version` (`"v2"`) and `fingerprint`.

## 4. Splits & Protocol (`_capture_fold`, `resolve_fold`)

- **Capture protocol (`split: capture`)**: Leave-one-out folds assign all captures with `capture["attack"] == holdout` to `"test"`. Remaining captures are grouped by `attack` (or `"__benign__"`) and split `60/20/20` over `recording_id(capture_id)` via `_role_split()`.
- **Window protocol (`split: window`)**: Random per-capture window split seeded by `data.split_seed`, retained only for legacy parity comparisons against the submitted code.
- **Variant & benign filtering (`resolve_fold`)**: `data.variants` filters only `kind == "attack"` captures (`kind == "benign"` is controlled by `data.benign_source`), and missing `holdout` attacks for the chosen variants (e.g. `leave_one_out/fuzzing` on `mas`) raise `ValueError`.
- **Low-sample caveat (`fuzzing`)**: Across all 3 `fuzzing` captures there are 16 attack windows total (`11 + 4 + 1`), yielding small evaluation counts (`n_unknown = 16` in `leave_one_out/fuzzing` and `test = 1` in `closed_set`); report `n_test` / `n_unknown` and multi-seed 95% confidence intervals alongside metrics.

## 5. Data Leakage Sources & Prevention (`canosr/data.py`, `canosr/train.py`, `canosr/cli.py`)

| Leakage Vector | Legacy Risk / Cause | Prevention in `can-osr-gnn` |
|---|---|---|
| Adjacent-window / same-session split | Random split over windows (`stride <= window`) puts near-duplicate fragments across `train`/`test` | Non-overlapping `window=300, stride=300` paired with recording-level `split: capture` in `_capture_fold()` |
| `fab` ↔ `mas` masquerade copies | `mas/<name>_masquerade` edits copies of `fab/<name>`; splitting by `f"{variant}/{attack}"` put copies in opposite splits | `recording_id()` strips `<variant>/` and `_masquerade`; `_capture_fold()` groups by `attack` over unique `recording_id`s so both variants always share the same role; `assert_disjoint()` and Gate `G5` reject any cross-role recording overlap |
| Single-recording boundary drift (`MECT`) | `fab/max_engine_coolant_temp_attack` (`57,913` frames) and `mas/..._masquerade` (`57,871` frames) differ by `42` injected frames, shifting index cuts by 1–2 windows | `_window_graph()` records `t_start` and `t_end`; `_split_single_recording_group()` cuts benign and attack subsets at shared reference timestamps `t_cut1` and `t_cut2`, dropping straddling boundary windows (`train <= 22.661409s`, `val 22.665360..23.461676s`, `test >= 23.461677s`) |
| Holdout-recording benign frames in LOO | Pooling all benign frames allowed benign windows from the holdout drive into `train`/`val` | `_capture_fold()` assigns the entire holdout capture (`y == 0` and `y != 0`) to `"test"` when `holdout` matches |
| Feature & edge scaler statistics | Fitting scalers across all captures leaks `val`/`test` variance | `fit_scalers()` computes `max_id`, node `mean`/`std`, `edge_min`/`edge_max`, and `edge_gap_mean`/`edge_gap_std` strictly on training benign windows (`buckets["train"]`, `y == 0`) after splitting |
| Cross-capture timing & CSV ordering | Processed CSVs arrive grouped by CAN ID rather than chronological order | `load_csv_frames()` and `parse_raw_log()` sort by `t` (`kind="stable"`) and `add_gaps()` computes per-ID `t.diff()` independently within each capture |

Verification on every split build and gate run:
1. `_capture_fold()` + `resolve_fold()` call `assert_disjoint(roles)`, asserting valid roles (`train`/`val`/`test`) and empty intersection of `recording_id`s across roles.
2. `canosr gate --dataset road` (`G5 leakage`) re-runs `assert_disjoint()` on every `split: capture` fold in `$CANOSR_DATA/processed/road/splits/*.json`.
