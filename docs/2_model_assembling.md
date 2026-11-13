# Stage 2: Model Assembling (`canosr/models.py`, `canosr/registry.py`, `canosr/config.py`)

## 1. Configuration Resolution & Stable Run ID

- `load_config(path, overrides)` merges `DEFAULTS` (`canosr/config.py`) → YAML preset (`configs/paper_road*.yaml`) → CLI `--set` overrides.
- `config_hash(cfg)` hashes the resolved dictionary after stripping non-numerical fields (`data.root`, `run.device`, `run.force`, `run.root`). Settings that alter outcomes (`run.limit`, `run.amp`, `model.*`, `loss.*`, `train.*`, `osr.*`) produce distinct `run_id` hashes.

## 2. Feature Standardization (`fit_scalers`, `apply_scalers` in `canosr/train.py`)

Before loading batches into `GraphAE`, training-benign (`y == 0`) statistics normalize every split (`train`, `val`, `test`) without test leakage:

- **Nodes (`x`)**: `x[:, 0]` (`can_id`) is divided by `max_id`; continuous columns `x[:, 1:]` are z-scored (`(x[:, 1:] - mean) / std`, zero-variance guarded to `1.0`).
- **Edges (`edge_attr`)**: Column `0` (transition count) is min-max scaled to `[0, 1]`; timing columns `1:` (`mean_dt`, `std_dt`) are z-scored (`edge_gap_mean`, `edge_gap_std`).

## 3. `GraphAE` Architecture & Plug Points (`canosr/models.py`)

Components are built through `registry.build(kind, name, **kwargs)` across five pluggable registries (`encoder`, `readout`, `head`, `loss`, `scorer`). Default `GraphAE` assembly:

1. **Encoder (`model.encoder`, default `gatv2`)**: Multi-layer residual message passing (`GATv2Encoder`, `GATEncoder`, `SAGEEncoder`, `GCNEncoder`, or `MLPEncoder`). With `hidden=8`, `heads=5`, `layers=2`, node embedding width is `width = hidden * heads = 40`, with `GELU`, `dropout=0.3`, edge attributes (`edge_dim=3` for `gatv2`/`gat`), and mean across layer outputs: `h ∈ R^[N, 40]`.
2. **Readout (`model.readout`, default `attn`)**:
   - `AttnReadout`: Gated sum pool `global_add_pool(x * sigmoid(Linear(x)), batch)`.
   - `PoolReadout`: `global_add_pool` (`"add"`) or `global_mean_pool` (`"mean"`).
3. **Projection (`embed`)**: Two-layer MLP (`40 -> 80 -> GELU -> Dropout -> 8`) followed by L2 normalization (`F.normalize(..., dim=1)`) to produce hyperspherical graph embeddings `z ∈ R^[B, 8]`.
4. **Decoders**:
   - `node_decoder`: MLP `40 -> 128 -> ReLU -> 13` outputting `node_hat` for node feature reconstruction.
   - `adjacency_logits`: Inner-product decoder `einsum("bid,bjd->bij", dense_h, dense_h)` over `to_dense_batch(h, batch)` producing `adj_logits ∈ R^[B, N_max, N_max]`.
5. **Classifier / Open-Set Head (`model.head`, default `prototype`)**:
   - `PrototypeHead`: Stores L2-normalized class centroids `prototypes ∈ R^[n_known, 8]` updated from `eval_train_loader`; outputs negative Euclidean distances `-cdist(z, prototypes)` as logits.
   - `LinearHead` (`linear`) / `IdentityHead` (`none`): Supervised or bypass baselines.
