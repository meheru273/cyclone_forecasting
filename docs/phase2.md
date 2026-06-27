# Phase 2 — Multi-Frame Cyclone Forecasting with Rectified Flow

> **Goal.** Build the headline model: a single latent rectified-flow generator that takes 2–3 past frames at 3-hour cadence and produces 1–3 future frames of joint GRIDSAT-IR + ERA5 (U, V, T, P) at 256 × 256. Then ablate the four design knobs that drive the paper: **context window**, **temporal mixing**, **conditioning fusion**, **physics loss**.
>
> **Scope deliberately kept lean.** No internal LDM/DDPM ablation (covered by the Nath et al. external baseline in `phase3.md`). All ablations are *internal to the flow model*.
>
> **Pre-requisite.** Phase 1 complete: the flow scripts have been brought to feature parity with `multi_channel_diffusion.py`, a 2-RF single-frame model is trained, and the year-based data split (2012–2020 train / 2021 val / 2022 test) is in place.

---

## 2.1  Task

Inputs: `(gridsat_{t-k}, era5_{t-k}) for k = 0..K_in-1`, plus per-frame IBTrACS lat/lon and timestamp.

Outputs: `(gridsat_{t+k}, era5_{t+k}) for k = 1..K_out`.

Defaults: **K_in = 3, K_out = 3**, native 3-hour cadence ⇒ 9-hour forecast horizon. Justified by GRIDSAT-B1's native cadence and confirmed in `phase0_research_plan.md`.

Decision recap — why **not** hourly video (Option B from the user prompt): GRIDSAT-B1 is natively 3-hourly. Hourly satellite ground truth does not exist; training on interpolated GRIDSAT fabricates targets. Option A keeps the joint-channel contribution intact.

---

## 2.2  Backbone (frozen across all Phase-2 ablations)

| Item | Value | Source |
|---|---|---|
| VAE | `MultiChannelVAE(in=5, latent=4, base=64)`, frozen, scale recomputed from data | Phase 1 |
| UNet backbone | `DiffusionUNet`, `base_channels=128`, `channel_mults=(1,2,3,4)`, attention at the middle encoder + MidBlock, dropout 0.1 | Frozen scaffolding in `phase0_research_plan.md` |
| Optimiser | AdamW, lr 1e-4 → 5e-5 cosine, weight decay 1e-5, 5-epoch warmup, EMA decay 0.9999 | Same as MC-Diff |
| Training objective | 2-Rectified Flow (best from Phase 1; revisit if Phase 1 says otherwise) | `phase1.md` |
| Sampler at eval | Euler, 4 steps (also 1 / 16 / 50 in the budget sweep) | §2.7 |
| AMP | BF16 on RTX 4090 (Ampere+ supports BF16 cleanly; avoid FP16 grad-scaler quirks) | Hardware |
| Grad checkpointing | ON for V2 (temporal-attention variant), OFF for V1 | Memory |
| Batch size | 4 for V1 (channel-stack), 2 for V2 with grad-accum 2 | 16 GB VRAM target |

Reference number for sanity checking: single-frame 2-RF (Phase 1) is the floor; multi-frame V1 with K_in=3, K_out=3 is the target. If multi-frame loses to single-frame rolled out, the C3 contribution is dead — see §2.4 below.

---

## 2.3  Two backbone variants (now Ablation B, not the headline choice)

### V1 — Channel-stacked 2-D UNet (lightweight)

- Input latent shape: `(B, K_in·4, 64, 64)` = stack the K_in conditioning latents along the channel axis.
- Output latent shape: `(B, K_out·4, 64, 64)`.
- UNet: same `DiffusionUNet` but with `in_channels = K_in·4, out_channels = K_out·4`.
- Per-frame conditioning: `ConditionEncoder` called K_in times; per-frame context vectors *summed* (default) before being added to the time embedding. Per-frame coords flattened: `Linear(K_in·2 → embed_dim)`.
- Frame-position signal: a small learned embedding indexed by frame slot, added to `t_emb` for each output channel-group.

Pros: parameter count grows ~3 % vs single-frame (only the first/last conv channels widen). Trains in the 16 GB budget with batch 4.

Cons: no explicit cross-frame spatial mixing. For 3 frames in / 3 frames out this is empirically OK; degrades as the horizon grows.

### V2 — Factorised 2-D spatial + 1-D temporal attention

- Input latent shape: `(B, T, 4, 64, 64)`, `T = K_in + K_out`.
- Same `DiffusionUNet`, weight-shared across the time axis (each frame goes through identically).
- After each `MidBlock`, insert a **temporal self-attention** layer that attends across the T-token sequence at each spatial location.
- Cross-attention from output-frame tokens to input-frame context vectors, inside the temporal-attention layer.
- Sinusoidal frame-index PE inside the temporal attention.

Pros: properly handles temporal dependence; aligns with 2501.16003-style video diffusion if a reviewer asks.

Cons: parameter count grows ~15–20 %; activation memory grows roughly T×. On the RTX 4090 we use batch 2 + grad-accum 2 + grad-checkpointing + BF16 to fit.

**V1 vs V2 is Ablation B (§2.6).** We do not pre-commit to one — the result decides.

---

## 2.4  The single-frame rollout baseline (S1 — falsification check)

The Phase-1 single-frame 2-RF model can be rolled out auto-regressively for 3 steps:

```
ẑ_{t+1} = model(z_t)                       # single-frame conditioning
ẑ_{t+2} = model(ẑ_{t+1})                  # feed prediction back as input
ẑ_{t+3} = model(ẑ_{t+2})
```

This is the **floor row** of the context-window ablation. If multi-frame (K_in=2 or 3) does not beat this rolled-out single-frame baseline at every lead time, the multi-frame story is dead and we report it honestly. Critical row — do not skip it.

---

## 2.5  Conditioning fusion (Ablation C)

The current `ConditionEncoder` does two kinds of fusion:

- **Scalar fusion** of `t_emb` and `ctx_emb` — additive: `comb = t_emb + ctx_emb`.
- **Spatial fusion** of `image_features` and the broadcast `context_spatial` — additive: `out = image_features + context_spatial`.

Three variants to compare:

| Variant | Scalar fusion | Spatial fusion |
|---|---|---|
| **C-Add** (current) | `t_emb + ctx_emb` | `image_features + ctx_broadcast` |
| **C-Concat** | `proj(cat([t_emb, ctx_emb]))` (Linear back to emb_dim) | `cat([image_features, ctx_broadcast], dim=1)` → 1×1 conv → reduce channels back |
| **C-CrossAttn** | `t_emb + ctx_emb` (cheap) | cross-attention layer in the UNet's middle level: latent queries, `ctx_emb` keys/values (one extra `MultiheadAttention` block) |

Notes:

- C-CrossAttn is the most expressive but adds one attention block (~ 1 % params). It is the standard for class-conditional diffusion in image generation; whether it helps when the conditioning is *spatial* (not class-token) is genuinely unclear in the weather-forecasting literature. Worth testing.
- C-Concat doubles the input channel count of the init-conv; tiny overhead.
- Run all three at the same K_in=3, K_out=3, V1 backbone, no physics loss. The winner becomes the default for the rest of Phase 2 and for Phase 3.

---

## 2.6  Ablation matrix (this is the whole Phase-2 contribution)

All rows: same VAE, same backbone scaffolding, 2-RF objective unless noted, train 100 epochs, eval at every 10 with full metric stack (§2.8).

| ID | Ablation | Variants | # runs | Why |
|---|---|---|---|---|
| **A** | Context window (K_in) | (1) single-frame rolled out [= S1], (2) K_in=2, (3) K_in=3 | 2 new + S1 | Justifies the 3-frame conditioning |
| **B** | Temporal mixing | V1 channel-stack, V2 temporal attn | 2 | The 2501.16003-style claim, internalised |
| **C** | Conditioning fusion | C-Add, C-Concat, C-CrossAttn | 3 | Conditioning-design question |
| **G** | Physics loss | OFF, +continuity, +geostrophic, +track, +all | 5 (1 baseline + 4) | The C4 contribution |
| **S2** | Prediction horizon | K_out=1, 2, 3 | 2 new (K_out=3 from A) | Justifies the 9 h horizon |
| **S3** | Reflow depth | 0-RF (OT-CFM), 1-RF, 2-RF | 3 | Which flow variant |
| **S4** | Sampler steps (eval only) | 1, 4, 16, 50 | 0 (re-eval) | Speed/quality curve, internal |
| **S5** | Light hyperparam | base=64 vs 128 (lock-in); latent_dim=4 vs 8 | 3 | Justifies the backbone-128 / latent-4 choice |

Total **new training runs:** 2 + 2 + 3 + 5 + 2 + 3 + 3 = **20 runs**. Many are short (50–80 epochs) — only the headline A6 row uses the full 100. On the RTX 4090, expect ~6–8 h per single-frame run, ~10–14 h per multi-frame V1 run, ~18–22 h per V2 run. Total wall-clock at 24/7: roughly **2–3 weeks** for Phase 2, plus Phase 3 (k-fold + Nath baseline).

**Ablation order (so we can short-circuit if findings are clear early):**

1. **S3 first** — pick the flow variant. If 2-RF wins, all later rows use it; if 1-RF or OT-CFM ties, use the cheaper one.
2. **A second** — pick K_in (we expect K_in=3 to win but the rolled-out single-frame baseline could surprise us).
3. **B third** — V1 vs V2.
4. **C fourth** — conditioning fusion.
5. **S2 fifth** — prediction horizon (sometimes K_out=3 hurts t+1 quality; want to know).
6. **G last** — physics loss, with `use_phys_loss=True/False` flag and per-loss sub-flags.
7. **S5 in parallel** — these are cheap to slip in alongside the others.

After S3/A/B/C/S2 we have a single "best" configuration; that is the row that goes into Phase 3's headline comparison vs Nath et al.

---

## 2.7  Multi-frame dataset wrapper

Add a `MultiFrameCycloneDataset` next to the existing `CycloneDataset` (do *not* replace it — Phase-1 single-frame must remain reproducible).

Spec:

- Walk the same storm-folder structure.
- For each storm, sort timestamps. Emit one sample per valid window `i` with `i ≥ K_in - 1` and `i + K_out < len(timestamps)`. Inputs: `timestamps[i - K_in + 1 : i + 1]`. Targets: `timestamps[i + 1 : i + 1 + K_out]`.
- No padding across storm boundaries.
- Return: `gridsat_in (K_in, 1, H, W)`, `era5_in (K_in, 4, H, W)`, `target_gs (K_out, 1, H, W)`, `target_era5 (K_out, 4, H, W)`, `coords_in (K_in, 2)`, `target_coords (K_out, 2)`, `timestamps_in: list[str]`, `target_timestamps: list[str]`, `storm_name`, `year`.
- Augmentation: random horizontal flip applied *consistently across all K_in + K_out frames* per sample.
- Year-based split — see §2.7.1.

### 2.7.1  Year-based split (replaces the random storm split everywhere)

Hardcoded years (matches `count_storms_per_year.py`'s default):

```python
TRAIN_YEARS = ['2012','2013','2014','2015','2016','2017','2018','2019','2020']
VAL_YEARS   = ['2021']
TEST_YEARS  = ['2022']
```

`get_dataloaders` is changed: removes `num_val_storms`, `random_seed`, and the in-function `random.sample` block; takes `train_years`, `val_years`, `test_years` explicitly; `mode='val'` and `mode='test'` filter on year instead of on a random storm subset.

Before any Phase-2 run, **run `count_storms_per_year.py` once** to confirm 2021 alone has enough storms to make a meaningful val set (≥ 20 storms is the soft target). If not, fall back to `VAL_YEARS = ['2020','2021']` and `TRAIN_YEARS = ['2012'...'2019']`.

### 2.7.2  Latent caching

Encoding ≈ 9 years × hundreds of storms × dozens of timestamps × (K_in + K_out) frame-loads on the VAE is the bottleneck. Cache the per-(storm, timestamp) latent as `{storm}/{timestamp}/latent.npy` on first pass; subsequent epochs skip the encoder. Reduces epoch time roughly 3×. Add `--latent-cache-dir` to configs.

---

## 2.8  Evaluation — per lead time, expanded metric stack

For every Phase-2 model, at every `eval_every` epochs and at the final epoch, compute:

### Pixel/image quality (per channel, per lead time)

- **MAE, MSE, PSNR, SSIM** — existing
- **LPIPS** (VGG-based; new) — perceptual similarity
- **RAPSD** (radially-averaged power spectral density; new) — spatial-scale preservation; logged as the L1 distance between the log-RAPSD of prediction and target

### Distribution / probabilistic

- **CRPS** (new) — sample N=10 from the RF (different ODE-noise seeds) and compute CRPS per pixel, averaged
- **Sliced-Wasserstein-1** between predicted and target value histograms per channel (new)

### Threshold-based (weather-domain)

- **CSI / POD / FAR** at IR brightness-temperature thresholds 220 K and 200 K (new) — does the model put deep convection in the right place?

### TC-specific

- **Steering-flow displacement** (existing, kept) — degrees lat/lon error at t+1, t+2, t+3
- **Track great-circle error** in km (new, derived from steering-flow integration with cos(lat) correction)
- **MSLP error** in hPa, peak-pixel basis (new)
- **Max |U|, |V|** error in m/s (new)

### Multi-frame only

- **FVD** (Fréchet Video Distance; new) — temporal coherence; replaces FID for multi-frame

### Compute (reported once per run)

- Parameter count
- Train time / epoch on the RTX 4090
- Inference time / sample at each sampler-step budget

All values written to `outputs/{run_id}/metrics/per_leadtime.csv` with columns `epoch, lead, channel, mae, mse, psnr, ssim, lpips, rapsd_l1, crps, sw1, csi_220, pod_220, far_220, csi_200, pod_200, far_200, steering_mae_lat_deg, steering_mae_lon_deg, gc_error_km, mslp_err_hpa, u_max_err, v_max_err`. Multi-frame runs also write `outputs/{run_id}/metrics/fvd.csv`.

### Plots produced per run

1. Loss curves (train/val) + per-channel val MSE — already partially in MC-Diff
2. Lead-time × metric curves (PSNR, SSIM, CRPS, track-km) — multi-frame only
3. Steering-flow 3-panel — already in MC-Diff, per lead time in Phase 2
4. RAPSD log-log plot per channel — new
5. Sample-budget × quality scatter (S4) — new
6. Spatial error heatmap, averaged across val samples — new
7. Per-channel ECDF comparison (predicted vs target value distribution) — new
8. Failure-case gallery — 6 worst-PSNR predictions with diagnostics — new

Track-overlay basemap plot and k-fold box plot live in `phase3.md`.

---

## 2.9  Physics loss — flag-driven, default OFF

The model trains **without** physics losses by default. The user has confirmed: stabilise the base model first, then add physics as a documented improvement.

### 2.9.1  Configuration flags

```python
config = {
    ...
    'use_phys_loss':           False,   # master flag; default OFF
    'use_continuity':          False,   # ∂U/∂x + ∂V/∂y ≈ 0 penalty
    'use_geostrophic':         False,   # f·U ≈ -∂P/∂y/ρ, f·V ≈ ∂P/∂x/ρ
    'use_track_consistency':   False,   # predicted U/V integral matches IBTrACS Δposition
    'lambda_continuity':       1e-3,
    'lambda_geostrophic':      1e-4,
    'lambda_track':            1e-2,
    'phys_warmup_epochs':      10,      # ramp each enabled term linearly 0 → λ
}
```

When `use_phys_loss=False`, no decoded-target-from-latent step is needed during training — the loss is pure `‖v_θ − v★‖²`. When `use_phys_loss=True`, the velocity-predicted x1 (decoded) is computed for the physics terms only (no extra optimisation through the VAE — gradients flow through the decoder, the VAE stays frozen).

### 2.9.2  The three losses (only activated when their flag is True)

**Continuity (2-D barotropic approximation):**

```
divergence_xy(U, V) = ∂U/∂x + ∂V/∂y     # finite diff on the decoded 256×256 grid
L_continuity        = mean( divergence_xy(U_pred, V_pred)² )
```

Mask the central 16 × 16 of the storm core (where divergence is physically real and large).

**Geostrophic balance (away from core):**

```
res_x = f · U_pred + (∂P_pred/∂y) / ρ
res_y = f · V_pred - (∂P_pred/∂x) / ρ
L_geo = mean( (res_x² + res_y²) · core_mask )
```

`f = 2Ω sin(lat)` with lat from IBTrACS coords; `ρ = 1.2 kg/m³`. Mask the central 32 × 32 storm core (gradient wind dominates there, not geostrophic).

**Track consistency:**

```
For each lead k ∈ {1..K_out}:
    dlat_pred_k = mean(V_pred_k over central 64×64) · (3 · k h) / 111 km
    dlon_pred_k = mean(U_pred_k over central 64×64) · (3 · k h) / (111 km · cos(lat_t))
L_track     = Σ_k ‖(dlat_pred_k, dlon_pred_k) - (lat_{t+k} - lat_t, lon_{t+k} - lon_t)‖²
```

Always compute the physics terms in **un-normalised physical units** (multiply by per-channel std, add mean) and divide by an empirical scale measured on the first 100 training batches, so the `λ_*` are interpretable across runs.

### 2.9.3  Ablation rows (Ablation G)

| Row | `use_phys_loss` | `use_continuity` | `use_geostrophic` | `use_track_consistency` |
|---|---|---|---|---|
| G1 (baseline) | False | — | — | — |
| G2 | True | True | False | False |
| G3 | True | False | True | False |
| G4 | True | False | False | True |
| G5 | True | True | True | True |

Each row uses the same backbone (the winner of S3 + A + B + C). 80 epochs each is enough; if any single-loss row diverges, report the divergence with the loss curve and move on — that is also a finding.

---

## 2.10  What gets reported as the "Phase-2 result"

Two tables, one figure each:

**Table P2-1: Ablation summary.** Each row = one of A/B/C/S2/S3 winners. Columns: PSNR, SSIM, LPIPS, CRPS, track-km, params, train time / epoch. Bold the winning configuration. This drives the Phase-3 head-to-head.

**Table P2-2: Physics-loss ablation (G).** 5 rows. Same columns as P2-1, plus a "divergence histogram peak" column showing whether continuity actually pulled the predicted wind toward divergence-free.

**Figure P2-1: Lead-time × metric.** PSNR, SSIM, CRPS, track-km vs lead time, with rows for: single-frame rolled out, multi-frame best (no physics), multi-frame best (with all physics), Nath et al. (filled in later in Phase 3).

---

## 2.11  Risks

1. **Multi-frame doesn't beat rolled-out single-frame.** Most likely if K_in=3 is wrong or temporal attention is needed. Mitigation: this is exactly what A and B test. If both lose, drop the multi-frame contribution and re-pitch the paper as "rectified flow for joint IR+ERA5 single-frame forecasting" — still publishable, narrower scope.
2. **Physics losses destabilise.** With the warmup ramp and small `λ` values this is unlikely, but if a row diverges, mask wider (32 → 48 px for continuity) before giving up.
3. **VAE bottleneck caps all rows at the same PSNR.** If every Phase-2 ablation lands within 0.3 dB of each other, the latent VAE is the floor, not the generator. Phase 1's optional VAE touch-up ( §1.6) becomes mandatory.
4. **K_in=3 multi-frame V2 OOM at batch 2 on 16 GB.** Drop to batch 1 + grad-accum 4 + grad-checkpoint all levels. Worst case, downsize V2 to base=96 just for that row and note the asymmetry in the paper.

---

## 2.12  Acceptance criteria

1. `MultiFrameCycloneDataset` exists and a unit test confirms it emits `K_in + K_out` consecutive frames per sample, never crossing storm boundaries.
2. `get_dataloaders` accepts `train_years / val_years / test_years` and no longer does random storm selection.
3. All A/B/C/S2/S3 rows have a `metrics/per_leadtime.csv` with the columns listed in §2.8.
4. The S1 (single-frame rolled-out) baseline row exists and is compared head-to-head with the best multi-frame row in `phase2_report.md`.
5. The physics-loss G rows are produced (1 baseline + 4 variants); `use_phys_loss=False` was the default for all earlier rows.
6. One `phase2_report.md` summarising Tables P2-1, P2-2 and Figure P2-1 exists. Hand off to `phase3.md`.
