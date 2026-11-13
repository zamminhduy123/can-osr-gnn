# can-osr-gnn

Graph Autoencoder training and evaluation pipeline for open-set intrusion detection on Controller Area Networks (ROAD dataset).

See [CHANGELOG.md](changelogs/CHANGELOG.md) for revision history and differences from the original submitted scripts.

## Setup

```bash
conda activate py311          # Python 3.11, torch 2.2.1+cu121, PyG 2.6.1
pip install -e ".[dev]"
pytest -v
```

`CANOSR_DATA` sets the workspace root for raw logs, processed caches, and run outputs (defaults to `/mnt/data2/ntmduy/data/can-osr`):

```
$CANOSR_DATA/
  raw/road/          ROAD raw logs + processed CSVs
  raw/canfd/         optional external CAN-FD raw captures
  processed/road/    captures.json, splits/, graphs/
  runs/              per-run checkpoints, configs, metrics, predictions
```

## Pipeline Quickstart

Prepare captures, graph windows, and splits once per dataset:

```bash
canosr build-captures --dataset road
canosr build-graphs   --dataset road --window 300 --stride 300 --force
canosr build-splits   --dataset road --split capture --seed 1
canosr build-splits   --dataset road --split window  --seed 1
```

Train and evaluate one fold:

```bash
canosr run --config configs/paper_road_fab.yaml --fold leave_one_out/fuzzing --seed 1
```

Aggregate completed runs and run acceptance gates:

```bash
canosr aggregate --runs "$CANOSR_DATA/runs"
canosr gate --dataset road
```

## Configs & Protocols

- [`configs/paper_road.yaml`](configs/paper_road.yaml): joint `fab + mas` evaluation using capture-disjoint splits.
- [`configs/paper_road_fab.yaml`](configs/paper_road_fab.yaml): fabrication captures only (matches paper Table 5 leave-one-out protocol).
- [`configs/paper_road_mas.yaml`](configs/paper_road_mas.yaml): masquerade captures only (note: excludes `fuzzing`).

Split and OSR details:
- **Capture-disjoint (`split: capture`)**: `fab/*` and `mas/*_masquerade` copies of the same underlying recording are always assigned the same role (`train`, `val`, or `test`), verified at recording granularity by `assert_disjoint`. Known single-recording attacks (e.g., `max_engine_coolant_temp`) use shared-timestamp 60/20/20 slicing in `_split_single_recording_group` with straddling windows dropped so `fab` and `mas` never overlap across splits.
- **Window (`split: window`)**: per-capture random window split seeded by `data.split_seed`, retained for legacy parity checks.
- **Two-stage OSR decision & score**: predictions with `argmax == 0` (benign) use `fused - threshold` against benign validation windows; predictions with `argmax > 0` (known attack) use `(proto - proto_threshold) / proto_std` calibrated on known-attack validation windows (`0 < y < n_known`), evaluated at threshold `0.0` across threshold-free and thresholded metrics.

## Pipeline Documentation

1. [Stage 1: Data Processing](docs/1_data_processing.md)
2. [Stage 2: Model Assembling](docs/2_model_assembling.md)
3. [Stage 3: Training](docs/3_training.md)
4. [Stage 4: Validation](docs/4_validation.md)
5. [Stage 5: Inference & Testing](docs/5_inference_testing.md)

## Repository Layout

```
canosr/
  config.py     YAML config loading, CLI overrides, stable run hashing
  registry.py   component registry (encoder, readout, head, loss, scorer)
  data.py       ROAD log/CSV loading, graph construction, recording-level splits
  models.py     GNN encoders, readouts, prototype/linear heads, GraphAE
  losses.py     SupCon, node/adjacency reconstruction, CE, focal losses
  scoring.py    anomaly channels, z-score normalization, fusion, two-stage decide
  metrics.py    binary, closed-set, AUROC/AUPR/FPR@95, OSCR, openness
  train.py      two-phase training loop, calibration, evaluation, artifacts
  cli.py        CLI entrypoints (build-*, run, aggregate, gate)
configs/        experiment YAML presets
docs/           stage-by-stage technical guides (1_data_processing..5_inference_testing)
changelogs/     revision history (CHANGELOG.md)
tests/          unit tests
```

## Experiment Rules

- Never commit datasets, checkpoints, or `runs/` outputs.
- Cite every reported number from an archived `runs/<run_id>/config.yaml` and `metrics.json`.
- Keep long experiments reproducible by running from a clean commit and checking `canosr gate --dataset road` first.
