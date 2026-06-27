# Phase 1 — Flow Scripts: Feature Parity, Year-Based Split, Single-Frame 2-RF Reference

> **Goal.** Bring `conditional_flow_matching.py` and `rectified_flow_score_matching.py` to feature parity with `multi_channel_diffusion.py`, switch to year-based data splits, and train a single-frame 2-Rectified-Flow model that becomes the *internal reference* (not a "vs DDPM" baseline). The Nath et al. external diffusion comparison lives in Phase 3, not here.
>
> **Out of scope.** Multi-frame conditioning, physics losses, statistical analyses, the Nath et al. reproduction.

---

## 1.1  Why this phase exists

Three motivations:

1. **The two flow scripts are missing improvements** that landed in `multi_channel_diffusion.py` after they were written (see `docs/latent_diffusion_changes.md`). Without porting these, every Phase-2 result inherits the gaps.
2. **The random-storm validation split is non-reproducible.** Two runs with different random seeds get different val storms. Reviewers will reject. We move to year-based splits.
3. **We need one trained single-frame 2-RF checkpoint** as the *starting point* for Phase 2's multi-frame ablations and as the floor for the S1 rolled-out comparison.

---

## 1.2  Gap list — `conditional_flow_matching.py` & `rectified_flow_score_matching.py`

| # | Gap | What `multi_channel_diffusion.py` does | Impact |
|---|---|---|---|
| G1 | `ConditionEncoder` is called without `coords` | Calls with `coords=cur_coords`; adds `coord_proj(coords)` to the temporal embedding | **High** — drops IBTrACS per-timestep location entirely |
| G2 | `latent_scale_factor = ckpt.get('latent_scale_factor', 0.18215)` falls back to SD value | Recomputes `1/z_std` from 50 val batches if checkpoint missing | **High** — wrong scale ⇒ wrong noise regime |
| G3 | No dynamic latent clamp range | `compute_latent_clamp_range(val_loader, num_batches=50)` | **High** for sampling stability |
| G4 | No LR warmup, no NaN/Inf guard | 5-epoch warmup; finite-loss skip block; finite-grad check after `clip_grad_norm_` | **High** — AMP single NaN poisons the EMA |
| G5 | Eval is per-channel MSE/MAE only | Full `MetricsCalculator`: PSNR, SSIM, SID; steering-flow displacement | **High** — incomparable to MC-Diff |
| G6 | Sample plots are 2-row | 3-row Input/Target/Generated layout with `set_ylabel` row tags | **Medium** |
| G7 | Checkpoint lacks `metrics_history`, `train_losses`, `val_losses` | Saved & restored on resume | **Medium** |
| G8 | EMA weights are not applied during epoch-level eval | `evaluate_era5` wraps the whole loop in `ema.apply / ema.restore` | **Medium-high** |
| G9 | Constant 50 ODE steps; no sampler-budget sweep | n/a (would be new) | **Medium** — needed for Phase 3 sampler-step plot |
| G10 | Random-storm val split (`get_dataloaders` `num_val_storms` + `random.sample`) | n/a — replace with year-based split | **High** |

---

## 1.3  Concrete changes

### 1.3.1  `ConditionEncoder` plumbing

Add `coords=None` to `FlowMatchingUNet.forward` and `ScoreUNet.forward`; pass it into `self.condition_encoder(gridsat, era5, timestamps, coords=coords)`. Non-breaking: default `None` is already handled in `ConditionEncoder.forward`.

In every trainer (`FlowMatchingTrainer`, `RectifiedFlowTrainer`, `ScoreMatchingTrainer`):

- pull `cur_coords = batch['current_coords'].to(self.device)` in `train_epoch`, `validate`, `sample`, `save_samples`, `evaluate_era5`
- pass `coords=cur_coords` to both `condition_encoder(...)` and `model(...)`

### 1.3.2  Trainer scaffolding (apply identically to all three flow trainers)

In `__init__`:

- replace `latent_scale_factor = ckpt.get('latent_scale_factor', 0.18215)` with `... = ckpt.get('latent_scale_factor', None)` and recompute in `train()` if `None` (G2)
- add `self.warmup_epochs = config.get('warmup_epochs', 5); self.base_lr = config['learning_rate']` (G4)
- add `self.latent_clamp_range = (-5.0, 5.0)` and copy `compute_latent_clamp_range` from MC-Diff (G3)

In `train()`:

- call `self.compute_latent_clamp_range(val_loader, num_batches=50)` before the epoch loop (G3)
- apply warmup factor inside the epoch loop for `epoch <= self.warmup_epochs` (G4)

In `train_epoch`:

- `if not torch.isfinite(loss): continue` skip block (G4)
- after `scaler.unscale_`, compute `grad_norm = clip_grad_norm_(...)`, then `if not torch.isfinite(grad_norm): skip` (G4)

In `evaluate_era5` (replacing the current thin one):

- copy MC-Diff `evaluate_era5` wholesale; swap the diffusion `sample(...)` for the flow `sample(...)` (G5)
- wrap in `ema.apply / ema.restore` (G8)
- add a sampler-budget sweep at the final epoch only: for `n_steps ∈ {1, 4, 16, 50}`, write `outputs/{run}/metrics/sampler_steps.csv` with `n_steps, overall_psnr, overall_ssim, overall_mse, wallclock_ms_per_sample` (G9)

In `save_samples`:

- use the 3-row Input/Target/Generated layout (G6)

In `save_checkpoint` / `_load_checkpoint`:

- save & restore `metrics_history`, `train_losses`, `val_losses` (G7)

In `sample`:

- clamp the integrated latent to `self.latent_clamp_range` before decoding (G3, flow variant)

### 1.3.3  Year-based split (G10) — applied to all scripts

Edit `get_dataloaders` (`multi_channel_diffusion.py` and the modules importing it):

- remove `num_val_storms`, `random_seed`, the `random.sample(...)` block
- new signature: `get_dataloaders(root_dir, train_years, val_years, test_years, batch_size, num_workers, img_size, coord_lookup_path)`
- `CycloneDataset(... mode='val' ...)` iterates over `val_years` (one folder per year) rather than over train_years filtered by `val_storms`
- `CycloneDataset(... mode='test' ...)` same for `test_years`
- *All* scripts (`latent_diffusion.py`, `multi_channel_diffusion.py`, `conditional_flow_matching.py`, `rectified_flow_score_matching.py`) update their `config` block:

```python
'train_years': ['2012','2013','2014','2015','2016','2017','2018','2019','2020'],
'val_years':   ['2021'],
'test_years':  ['2022'],
# remove: 'num_val_storms', 'random_seed' for val selection
```

`count_storms_per_year.py` (already in `scripts/`) must be run once before the first training run to confirm 2021 has enough storms; if not, widen to `['2020','2021']` and slide train down to 2012–2019.

### 1.3.4  Reflow stage in `rectified_flow_score_matching.py`

When generating synthetic pairs `(x0, x̂1)` for the 2-RF reflow stage, the conditioning vector must come from the **same batch** that seeded the noise. This is the easy-to-miss bug — without it the reflowed pairs are mis-conditioned and the second-stage objective collapses toward unconditional flow. Verify in the reflow loop.

---

## 1.4  Backbone update (everywhere)

All Phase-1 runs use the updated backbone:

```python
'base_channels':    128,        # was 64 — fixes underfitting
'channel_mults':    (1, 2, 3, 4),
'num_heads':        4,
'embed_dim':        128,
't_emb_dim':        128,
'dropout':          0.1,
'use_amp':          True,
'amp_dtype':        'bfloat16', # RTX 4090 supports BF16 natively
```

This is the frozen scaffolding from `phase0_research_plan.md` §0.3. **Do not change between Phase-1 runs.**

---

## 1.5  Reuse the MC-VAE

All Phase-1 flow trainers load `mc_vae_best.pt` and **do not retrain it**. Config:

```python
'mc_vae_checkpoint': './checkpoints/mc_vae/mc_vae_best.pt',
'reuse_vae':         True,
'vae_epochs':        0,    # skip VAE training if checkpoint exists
```

If `result_summary.md` reports MC-VAE FID ≈ 175 / PSNR ≈ 17 and Phase-1 flow runs all hit the same ceiling regardless of objective, the VAE is the bottleneck → see §1.6.

---

## 1.6  Optional VAE touch-up (only gated on §1.5)

If §1.5 says the VAE is the floor:

- Re-train MC-VAE for an extra 30 epochs.
- Use **per-channel reconstruction weights** `[1.0, 1.5, 1.5, 1.5, 1.5]` (down-weight GRIDSAT relative to ERA5 — the dominant-channel imbalance is the suspected cause of the ERA5-channel reconstruction gap).
- Keep `latent_dim = 4` (do not widen — that breaks ablation parity).
- Re-train all subsequent flow models on the new VAE.

This is a *gated* extra; default plan is to keep the existing VAE.

---

## 1.7  Phase-1 runs

| Run | Script | Objective | Epochs | Sampler @ eval | Purpose |
|---|---|---|---|---|---|
| **P1-CFM** | `conditional_flow_matching.py` | OT-CFM | 100 | Heun 50; sweep 1/4/16/50 at final | OT-CFM reference; Phase-2 S3 row "0-RF" |
| **P1-RF1** | `rectified_flow_score_matching.py` (`RectifiedFlowTrainer`, `reflow_iterations=0`) | 1-Rectified Flow | 100 | Euler 50; sweep 1/4/16/50 | Phase-2 S3 row "1-RF" |
| **P1-RF2** | same, `reflow_iterations=1` | 2-Rectified Flow | 100 (60 + 40 reflow) | Euler 4 headline; sweep 1/4/16 | **Phase-2 starting checkpoint** |
| (optional) **P1-Score** | `ScoreMatchingTrainer` | denoising score matching | 80 | annealed Langevin, 100 levels | Only if Phase-2 schedule allows — low priority |

Each run produces:

- `outputs/{run}/metrics/{run}_metrics.csv` with the columns from MC-Diff
- `outputs/{run}/metrics/sampler_steps.csv` at the final epoch
- Sample plots in the 3-row format
- Best-EMA checkpoint, latest checkpoint
- Loss-curve PNG

Three rows go directly into Phase-2 Ablation S3. The winner (most likely P1-RF2 based on FlowCast 2026 evidence) becomes the default for the rest of Phase 2.

---

## 1.8  Acceptance criteria

A reviewer can verify each from the artefacts:

1. `git diff` shows `coords=cur_coords` in every condition-encoder and UNet call in both flow scripts.
2. `get_dataloaders` no longer references `num_val_storms` or `random.sample`; it takes `train_years/val_years/test_years` explicitly.
3. `scripts/count_storms_per_year.py` has been run; output committed as `docs/dataset_year_summary.csv`. The chosen split is documented.
4. All three (or four) Phase-1 runs produce the standard metric CSV and the sampler-step sweep CSV.
5. EMA weights are applied for every `evaluate_era5` call.
6. Sample plots have the 3-row Input/Target/Generated layout.
7. Backbone is `base=128, mults=(1,2,3,4)` in all configs.
8. The MC-VAE checkpoint is loaded but not retrained.

A one-page `phase1_report.md` summarises the three rows in the same format as `result_summary.md`. When complete, hand off to `phase2.md`.
