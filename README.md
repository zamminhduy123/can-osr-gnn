# can-osr-gnn

Rewrite of the GraphAE training/evaluation pipeline for
*Open-Set Recognition with Multi-Objective Graph Neural Network in Controller Area Networks*.

The design contract is [`docs/PIPELINE_V2.md`](../codebase/docs/PIPELINE_V2.md). Read it before
changing behaviour.

## What this fixes

| Problem in the submitted code | Fix here |
|---|---|
| random window split leaks near-duplicate windows | capture-disjoint splits, asserted at load time |
| class index from `os.listdir` order | manifest with explicit class ids |
| adjacency loss returns plain MSE, omega unused | binary target + `pos_weight` BCE, one code path |
| SVDD and repulsion are dead code | removed |
| model selected on test F1 | selection on validation |
| four OR-ed flags as the OSR rule | one calibrated fused score |
| `open_f1` guard ends in `or True` | no such guard |
| edge timing features silently dropped | all three edge attributes used |
| one model class among 23 near-duplicates | one model, five plug points |

## Install

```bash
conda activate py311          # torch 2.2.1+cu121, PyG 2.6.1
pip install -e ".[dev]"
```

## Data

`CANOSR_DATA` defaults to `/mnt/data2/ntmduy/data/can-osr`.

```
raw/road/          ROAD raw logs + processed CSVs
raw/canfd/         reserved, empty until the archive is supplied
processed/road/    captures.json, splits/, graphs/
runs/              one directory per run
```

Build once per dataset:

```bash
canosr build-captures --dataset road
canosr build-graphs   --dataset road --window 300 --stride 300
canosr build-splits   --dataset road --split capture --seed 1
```

## Run

```bash
canosr run --config configs/paper_road.yaml --fold leave_one_out/mec --seed 1
canosr aggregate --runs "$CANOSR_DATA/runs"
canosr gate
```

## Layout

```
canosr/
  config.py     resolved config, YAML merge, hashing
  registry.py   five plug points
  data.py       ROAD parsing, captures, graphs, splits
  models.py     encoders, heads, GraphAE
  losses.py     supcon, reconstruction, ce
  scoring.py    channel scores, normalisation, fusion, threshold
  metrics.py    auroc, aupr, fpr@95, oscr, openness
  train.py      two-phase loop, calibration, evaluation
  cli.py        build-captures / build-graphs / build-splits / run / aggregate / gate
tests/
```

## Rules

- Never commit data, checkpoints, or run artifacts.
- Every claim must come from a `runs/<run_id>/config.yaml`.
- Long jobs run from a clean commit, never from a dirty tree.
- Pull before working, push after a green test run.
