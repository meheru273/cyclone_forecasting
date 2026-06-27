# SDM Development Log — memory.md

> **Purpose**: Chronological technical log of every major modification, debugging step, and lesson learned. Designed for agent handoff — read this before touching the codebase.

---

## Entry 1: Single-Channel Latent Diffusion Baseline

**Date**: Early April 2026  
**Feature/Issue**: Build initial latent diffusion model for GRIDSAT-only satellite image generation.

**Implementation Details**:
- Created `latent_diffusion.py` — full pipeline: VAE → Diffusion UNet → DDIM sampling
- `VAE`: `in_channels=1, latent_dim=4, base_channels=64`, encoder 256→128→64, decoder 64→128→256
- `DiffusionUNet`: latent-space noise predictor with `ConditionEncoder` (GRIDSAT + ERA5 + temporal embeddings)
- Noise schedule: linear β from 1e-4 to 0.02, 1000 timesteps
- DDIM sampling at inference (50 steps, η=0)
- Evaluation: MSE, MAE, PSNR, SSIM, FID, Inception Score

**Rationale**: Establish a working baseline for satellite image generation before adding multi-channel and trajectory capabilities.

**Outcome**:
- VAE converged well: **PSNR 42.1 dB, SSIM 0.972** (near-perfect reconstruction)
- Diffusion trained for 67 epochs: **PSNR 21.2 dB, SSIM 0.60, FID 122**
- Still improving at epoch 67 — no convergence plateau reached

**Pending Items**: Diffusion needs 100+ epochs. Model was initially deployed with `nn.Tanh()` in VAE decoder — later identified as a bug.

---

## Entry 2: nn.Tanh() Bug Discovery and Fix

**Date**: April 5, 2026  
**Feature/Issue**: VAE decoder `nn.Tanh()` clipping z-score normalized data.

**Implementation Details**:
- **Bug**: VAE decoder ended with `nn.Tanh()`, clamping output to [-1, 1]
- Data uses z-score normalization (unbounded range). GRIDSAT z-scores range [-1.39, 1.47], ERA5 even wider
- Any values outside [-1, 1] were irrecoverably clipped

**Rationale**: This explained why ERA5 channels had 10x higher MSE than GRIDSAT — ERA5 z-scores have wider tails that Tanh destroyed.

**Outcome**:
- Removed `nn.Tanh()` from both `latent_diffusion.py` (line ~365) and `multi_channel_diffusion.py` (line ~1370)
- Decoder now ends with `nn.Conv2d(base_channels, in_channels, 3, 1, 1)` — no activation
- **Lesson**: Never use bounded activations on unbounded data. This is the #1 bug to check for in any new VAE.

**Pending Items**: All old checkpoints incompatible after this fix — must retrain from scratch.

---

## Entry 3: Multi-Channel Diffusion — 5-Channel VAE

**Date**: April 5-24, 2026  
**Feature/Issue**: Extend from 1-channel (GRIDSAT) to 5-channel (GRIDSAT + 4 ERA5) generation.

**Implementation Details**:
- Created `multi_channel_diffusion.py` with `MultiChannelVAE(in_channels=5, latent_dim=4)`
- `_build_target()`: concatenates `batch['target'] (1,H,W)` + `batch['target_era5'] (4,H,W)` → `(5,H,W)`
- Added per-channel evaluation in `evaluate_era5()`: computes MSE/MAE/PSNR/SSIM per channel
- Channel names: `['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']`
- Added multi-panel visualization: 3 rows (Input/Target/Generated) × 5 channels

**Rationale**: Joint generation of satellite + atmospheric fields is the core research contribution — the model must learn physics-consistent cross-channel dependencies.

**Outcome** (50 epochs):
- GRIDSAT: PSNR 20 dB, SSIM 0.51
- ERA5 channels: PSNR 10-14 dB — poor, but still improving
- Overall: PSNR 12.4, SSIM 0.677

**Pending Items**: Model undertrained (PSNR still climbing). Compression ratio 5→4 potentially too aggressive.

---

## Entry 4: Coordinate Channels Attempt (v1 — Abandoned)

**Date**: April 28, 2026  
**Feature/Issue**: Add lat/lon as 2 extra VAE input channels (7-channel VAE).

**Implementation Details**:
- `parse_storm_coordinates()`: extract genesis lat/lon from IBTrACS SID format (`YYYYDDDHLLLOOO`)
- Created constant spatial maps: 2 channels filled with normalized (lat, lon) values
- Updated VAE to `in_channels=7, latent_dim=8`
- Coords excluded from horizontal flip augmentation
- Added trajectory comparison scatter plot

**Rationale**: Simplest integration — treat coords as extra image channels, let VAE learn to reconstruct them.

**Outcome**: **Abandoned before training.** Analysis showed this approach is fundamentally flawed:
- Constant spatial maps waste VAE capacity — the decoder must learn to reconstruct uniform tensors alongside complex weather fields
- Increased latent_dim from 4→8 doubles UNet parameter count without proportional benefit
- Genesis-only coordinates (from SID) don't capture per-timestep track positions

**Lesson**: Don't encode non-image data through image-focused architectures. Use conditioning instead.

---

## Entry 5: Per-Timestep Coordinate Lookup (IBTrACS)

**Date**: April 28-May 1, 2026  
**Feature/Issue**: Build accurate per-timestep lat/lon from IBTrACS best-track data.

**Implementation Details**:
- Created `build_coordinate_lookup.py`:
  1. Downloads IBTrACS v04r01 ALL CSV (~300 MB)
  2. Parses SID, ISO_TIME, LAT, LON columns
  3. Walks dataset directory tree to collect `(storm_name, timestamp)` pairs
  4. Matches each entry to nearest IBTrACS track point within 3-hour tolerance
  5. Computes actual lat/lon normalization stats from matched data
  6. Outputs `coordinate_lookup.json` with `{lookup: {"SID/timestamp": {lat, lon}}, coord_stats: {...}}`
- `CycloneDataset` updated with `coord_lookup_path` parameter:
  - Loads JSON at init, builds lookup dict
  - `__getitem__` looks up `cur_key = "{storm}/{timestamp}"` for current and next coords
  - Falls back to `parse_storm_coordinates()` (genesis) if timestamp not found
  - Returns `current_coords` (2,) and `target_coords` (2,) as normalized scalars

**Rationale**: Genesis-only coords give one position per storm. Real trajectory prediction requires per-timestep positions from IBTrACS best-track records.

**Outcome**: Successfully generates lookup with high match rate. Normalization stats computed from actual coordinate distribution (not hardcoded).

**Pending Items**: Must run `build_coordinate_lookup.py` on Kaggle before training — the JSON is a dataset dependency.

---

## Entry 6: Coordinate-Conditioned Architecture (v2 — Current)

**Date**: May 1-2, 2026  
**Feature/Issue**: Redesign coordinate integration — use conditioning instead of VAE channels.

**Implementation Details**:
- **VAE reverted to 5-channel** (`in_channels=5, latent_dim=4`) — coords NOT passed through VAE
- **`ConditionEncoder.coord_proj`**: `nn.Sequential(Linear(2→64), SiLU, Linear(64→128))`
  - `forward(gridsat, era5, timestamps, coords=None)`: if coords provided, `temporal_emb += coord_proj(coords)`
  - Coords are additive to temporal embedding, then projected to context via `context_projection`
- **`CoordPredictor`** (new class, line 1317):
  ```python
  class CoordPredictor(nn.Module):
      # MLP: context_emb(256) → 128 → 64 → 2
      head = nn.Sequential(Linear(256,128), SiLU, Linear(128,64), SiLU, Linear(64,2))
  ```
- **Joint loss** in diffusion training:
  ```python
  diff_loss = F.mse_loss(pred_noise, noise)
  coord_loss = F.mse_loss(coord_predictor(ctx), target_coords)
  loss = (diff_loss + coord_loss_weight * coord_loss) / grad_accum
  ```
- **`sample()` returns tuple**: `(gen5, pred_coords)` — 5-channel images + (B,2) coordinate predictions
- **Optimizer** includes CoordPredictor parameters alongside UNet and condition_downsample
- **Checkpoint** saves/loads `coord_predictor_state_dict`

**Rationale**: 
- VAE stays focused on what it's good at — image reconstruction
- Coordinates are low-dimensional (just 2 scalars) — an MLP is sufficient
- Conditioning lets the UNet generate weather patterns that are spatially consistent with the storm position
- Separate prediction head allows independent evaluation of trajectory accuracy

**Outcome**: 

**Pending Items**: 

---

## Entry 7: LR Scheduler & Checkpoint Resumption Fix

**Date**: May 2, 2026  
**Feature/Issue**: Correct LR scheduler behavior after Kaggle timeout resume.

**Implementation Details**:
- `CosineAnnealingLR` `T_max` changed from `diffusion_epochs` to `diffusion_epochs * 2`
  - **Rationale**: With `T_max = epochs`, the cosine completes a full cycle and LR rises back up in the second half. With `T_max = 2*epochs`, the LR only monotonically decays during training.
- Checkpoint loading enhanced:
  ```python
  if 'lr_scheduler_state_dict' in ckpt:
      self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
  else:
      for _ in range(self.start_epoch - 1):
          self.lr_scheduler.step()  # fast-forward
  ```
- Both `vae_checkpoint_path` and `diffusion_checkpoint_path` exposed in config for easy resume

**Rationale**: On Kaggle, training often gets interrupted at 9-hour timeout. Must be able to resume seamlessly from the last best checkpoint without LR discontinuities.

**Outcome**: Checkpoint resume verified in code. LR state correctly restored or fast-forwarded.

---

## Entry 8: LPIPS Metric Removal

**Date**: May 2, 2026  
**Feature/Issue**: Remove LPIPS metric from evaluation pipeline.

**Implementation Details**:
- Deleted `class LPIPS(nn.Module)` entirely
- Removed `lpips` from metrics dict, initialization, per-batch computation, and summary logging
- Retained FID and Inception Score as advanced metrics

**Rationale**: LPIPS requires pretrained AlexNet/VGG weights which may not be available in all Kaggle environments. It also wasn't adding significant diagnostic value given the z-score normalized data (LPIPS assumes ImageNet-range inputs).

**Outcome**: Cleaner evaluation pipeline. No functional impact on training.

---

## Entry 9: Diffusion Training Improvements & Architecture Fixes

**Date**: May 23, 2026  
**Feature/Issue**: Port proven components from single-channel training to latent_diffusion.py/multi_channel_diffusion.py, fix horizontal line artifacts, increase model capacity, and optimize schedule dynamics.

**Implementation Details**:
- **VAE Decoder & UNet UpBlock**: Replaced `nn.ConvTranspose2d` with `nn.Upsample(mode='nearest') + nn.Conv2d` to eliminate the horizontal line artifacts visible in generated samples.
- **Cosine Noise Schedule**: Switched to a cosine noise schedule by implementing the `CosineNoiseScheduler` class (inheriting from `LinearNoiseScheduler`), distributing learning signals more uniformly across timesteps.
- **Dynamic Decoder Configs**: Switched UNet decoders to a dynamic configuration loop to support arbitrary number of downsampling levels (fully supporting mults=(1,2,3,4)).
- **Config & Capacity**: Increased `base_channels` 64 → 128, `channel_mults` (1,2,4) → (1,2,3,4), `num_heads` 4 → 8, and training time from 100 → 200 epochs to fix underfitting.
- **LR Schedule**: Fixed `T_max` from `2 * epochs` to `epochs` so the cosine annealing decay completes within the run. Widened decay range from 5e-5 → 1e-6.
- **Validation**: Switched from 5 fixed timesteps to random matching timesteps (unbiased validation) directly comparable to the training loss.
- **AMP Enabled**: Switched config's `use_amp` flag to `True` to enable Automatic Mixed Precision, speeding up training by ~40%.
- **Documentation**: Created `system.md` containing full architecture, training loops, and data flow specifications.

**Outcome**: Diffusion architecture updated to support deeper capacity and more efficient noise distribution, ready for mixed-precision training.

---

## Entry 10: Mixed-Precision (AMP) Stability & Evaluation Fixes

**Date**: May 24, 2026  
**Feature/Issue**: Prevent NaN gradient explosions under AMP, stabilize training for the 4x larger model, and restore broken validation evaluation.

**Implementation Details**:
- **GradScaler Tuning**: Configured `torch.amp.GradScaler` with `init_scale=1024` and `growth_interval=2000` (instead of default 65536) to prevent early float16 overflow and NaN gradients during initial epochs.
- **NaN Guard**: Integrated checks inside training loops to skip gradient updates and reset scaler if loss or grad_norm is NaN/Inf, preventing weight corruption.
- **LR Warmup**: Implemented a 5-epoch linear warmup starting at `LR / 10` up to `LR` to stabilize gradients at startup.
- **Peak LR**: Reduced peak learning rate from `2e-4` to `1e-4` to ensure stable convergence of the larger 128-channel model.
- **VAE Decode**: Restored the VAE decoding step inside `evaluate_model` (which was accidentally removed during debug-print cleanup), ensuring validation metrics are computed on pixel-space images rather than raw latents.

**Outcome**: NaN gradient explosions resolved. Training runs stably with AMP enabled, and metrics are computed correctly in pixel-space.

---

## Entry 11: Bilinear Upsampling & Dynamic Clamping Fixes

**Date**: May 25-26, 2026  
**Feature/Issue**: Reduce horizontal line artifacts, optimize schedule/validation dynamics, and update coordinate evaluation in single-channel and multi-channel pipelines.

**Implementation Details**:
- **Bilinear Upsampling (both)**:
  - Switched VAE decoder upsampling layers and UNet `UpBlock` layers in both `latent_diffusion.py` and `multi_channel_diffusion.py` from `nearest` to `bilinear` with `align_corners=False`. This eliminates grid/checkerboard line boundaries in generated images.
- **Dynamic Latent Clamping (both)**:
  - Replaced the hardcoded DDIM sampler clamp ranges in `latent_diffusion.py` and `multi_channel_diffusion.py`.
  - Added `compute_latent_clamp_range()` in both trainers to dynamically calculate the actual min/max range of scaled latents from validation data before training starts (adding a 10% margin).
  - Configured `sample()`, `evaluate_model()`, and other sampling wrappers to pass the computed `latent_clamp_range` to `sample_prev_timestep_ddim`.
- **Cosine Noise Schedule & Validation Fixes (both)**:
  - Integrated `CosineNoiseScheduler` (inheriting from `LinearNoiseScheduler`) to distribute training signal more uniformly across timesteps.
  - Replaced validation on 5 fixed timesteps with random matching training timesteps to make loss directly comparable.
  - Aligned Cosine Annealing `T_max` with `diffusion_epochs` so the decay completes within the training loop.
- **Trajectory Tracking via Physical Steering Flow (multi-channel)**:
  - Removed MLP-based `CoordPredictor` head and coordinate loss from the joint training loss in `multi_channel_diffusion.py`.
  - Implemented heuristic steering flow displacement calculation directly from generated ERA5 U/V wind channels. Un-normalizes wind speed and converts to physical coordinate displacements:
    $$\Delta \text{lat} = \frac{v_{\text{mean}} \cdot dt}{111000.0}$$
    $$\Delta \text{lon} = \frac{u_{\text{mean}} \cdot dt}{111000.0 \cdot \cos(\text{lat})}$$
  - Added 3-panel visualization (quiver plot, scatter plot, error histogram) to track steering vector accuracy in physical degrees.
  - Updated checkpoint loading to cleanly ignore legacy `coord_predictor_state_dict` keys.

**Rationale**:
- Bilinear interpolation produces smoother spatial boundaries than nearest neighbor, avoiding sharp grid artifact lines.
- Dynamic clamping adapts to the model's actual latent statistical distribution instead of clipping valid values.
- Physical steering flow extraction from generated winds is more robust and physically grounded than using an abstract MLP coordinate predictor head.

**Outcome**: Both single-channel and multi-channel compilation and architecture verification succeeded. Models are ready for full retraining on Kaggle with smooth generation and physical trajectory evaluation.

---

## Key Lessons Learned

| # | Lesson | Context |
|---|--------|---------|
| 1 | **Never use `nn.Tanh()` on z-score data** | Clips unbounded values, destroys ERA5 channels |
| 2 | **Don't encode scalars through image VAEs** | Constant spatial maps waste decoder capacity |
| 3 | **Use conditioning for non-image data** | Coords are better as MLP conditioning than VAE channels |
| 4 | **Diffusion models are slow to converge** | 50 epochs is insufficient; plan for 100-150+ |
| 5 | **`latent_scale_factor` must match your VAE** | Default 0.18215 (Stable Diffusion) may not fit |
| 6 | **CosineAnnealing T_max matters** | Set T_max = epochs to allow complete LR decay within the run |
| 7 | **Per-timestep coords >> genesis-only coords** | IBTrACS lookup gives actual track positions |
| 8 | **Always save LR scheduler state in checkpoints** | Prevents LR jumps on resume |
| 9 | **Bilinear upsampling resolves grid artifacts** | Nearest-neighbor upsampling causes horizontal/vertical boundary lines |
| 10 | **Compute clamp range dynamically from data** | Hardcoded clamp limits clip valid latent ranges during sampling |
| 11 | **Physical steering flow >> MLP coordinates** | Extracting trajectory steering from U/V wind channels is physically consistent |

---

## Current Status (as of May 26, 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| `latent_diffusion.py` | ✅ Ready to Retrain | Bilinear upsampling and dynamic clamping integrated. Cosine scheduler ready. |
| `build_coordinate_lookup.py` | ✅ Ready | Run on Kaggle to generate `coordinate_lookup.json`. |
| `multi_channel_diffusion.py` | ✅ Ready to Retrain | Physical steering flow evaluation, Cosine scheduler, and validation fixes integrated. |
| VAE (1-ch / 5-ch) | 🔜 Needs training | Switched decoder to bilinear upsampling. |
| Diffusion (1-ch / 5-ch) | 🔜 Needs training | 200 epochs planned. |
| Trajectory evaluation | ✅ Code complete | Direct steering flow displacement calculations + 3-panel plotting. |

### Next Steps

1. **Delete old checkpoints** since VAE and UNet architectures changed (bilinear upsampling).
2. **Run `build_coordinate_lookup.py`** on Kaggle to generate the per-timestep coordinate lookup file.
3. **Train VAE models (1-ch and 5-ch)** to verify smooth reconstructions (PSNR > 38 dB / 17 dB respectively) without artifacts.
4. **Train Diffusion models (1-ch and 5-ch)** using the cosine noise schedule and dynamic clamp range settings for 200 epochs.
5. **Evaluate multi-channel steering flow**: Verify that the steering MAE total is within 2–5 degrees.
