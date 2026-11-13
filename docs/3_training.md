# Stage 3: Two-Phase Training (`canosr/train.py`, `canosr/losses.py`)

Training runs in `run()` for `train.epochs = 330` epochs using `class_aware` weighted random sampling (`WeightedRandomSampler` seeded by `seed`, replacement=True) and `batch_size = 512`.

## 1. Objectives & Loss Computation (`_batch_losses`, `canosr/losses.py`)

For each minibatch, forward pass produces `h`, `z`, `node_hat`, and `adj_logits`:

1. **Supervised Contrastive Loss (`supcon_loss`, weight `w_supcon = 2.0`)**:
   - Computed over known-labeled graphs (`y < n_classes`) at `temperature = 0.05`.
   - Upcasts features to `float32` (`F.normalize(features.float(), dim=1)`) so self-similarity masking (`-1e9`) never overflows `float16` under CUDA AMP.
2. **Node Reconstruction Loss (`node_recon_loss`, weight `w_node = 0.5`)**:
   - Per-graph averaged squared error over continuous node features (`skip_first=True` skips normalized `can_id` unless `model.id_in_recon=True`).
   - When `loss.recon_on_benign_only=True` (default), `graph_weight = (batch.y == 0).float()` restricts reconstruction gradients to benign windows only.
3. **Adjacency Reconstruction Loss (`adjacency_loss`, weight `w_adj = 0.7`)**:
   - Binary BCE-with-logits against binary edge existence `(to_dense_adj(...) > 0).float()`, masked by valid `node_mask` pairs and weighted by benign `graph_weight`.
   - `pos_weight = _auto_pos_weight(...)` sets `negative / max(positive, 1)` over up to `256` training benign graphs.
4. **Total Objective**:
   - `recon = w_supcon * loss_sup + w_node * loss_node + w_adj * loss_adj`
   - **Phase 1 (`epoch < phase2_start`, default `0..299`)**: `loss = recon`.
   - **Phase 2 (`epoch >= phase2_start`, default `300..329`)**: Prototypes are refreshed via `compute_prototypes(model, eval_train_loader, n_known, device)` before each epoch, and logits are scaled by `/ 0.1` for `PrototypeHead` CE:
     - `phase2_mode == "joint"` (default): `loss = recon + w_ce * ce` with all parameters trainable.
     - `phase2_mode == "freeze"`: Calls `model.freeze_for_phase2()` once, reinitializes optimizer at `lr_head = 1e-4`, and minimizes `loss = w_ce * ce + 0.1 * loss_sup`.

## 2. Optimizer, Schedule, AMP & Stability

- **Optimizer (`_make_optimizer`)**: Supports `adamw` (default, `lr = 0.01`, `weight_decay = 1e-4`), `adam`, and `sgd`.
- **LR Schedule (`scheduler = "cosine"`)**: `CosineAnnealingLR(T_max=phase2_start + 1, eta_min=lr * 0.01)`. In `"joint"` mode, scheduler stepping stops after `phase2_start` so Phase 2 intentionally runs at `eta_min = 1e-4` (matching submitted behavior); `"freeze"` creates a fresh cosine schedule over `epochs - phase2_start`.
- **Mixed Precision & Safety**: `torch.cuda.amp.autocast(enabled=amp_enabled)` paired with `torch.cuda.amp.GradScaler(enabled=amp_enabled)`, `grad_clip = 5.0` gradient norm clipping, and `skipped_nonfinite` counter per epoch whenever non-finite batch losses are skipped.
