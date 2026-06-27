# Satellite Cyclone Forecasting with Latent Rectified Flow

A latent generative model that forecasts tropical-cyclone state — joint **GRIDSAT-B1 infrared satellite imagery + ERA5 reanalysis fields (U-wind, V-wind, temperature, surface pressure)** — at a **9-hour horizon** in a single forward pass.

The headline model is a multi-frame **rectified-flow** generator operating in a 4-channel latent space learned by a multi-channel VAE. It takes 3 past frames at native 3-hour cadence and produces 3 future frames jointly across all 5 physical channels, conditioned on storm best-track coordinates from IBTrACS.

> Undergraduate thesis project · Khulna University of Engineering & Technology (KUET).
> Author: **Meheru Zannat** · [meherujannat@gmail.com](mailto:meherujannat@gmail.com)

---

## TL;DR

| | |
|---|---|
| **Task** | 9-hour cyclone forecast (3 frames × 3 h cadence) of joint IR + ERA5 fields at 256 × 256 |
| **Method** | Multi-frame latent rectified flow over a frozen multi-channel VAE |
| **Conditioning** | Past frames + per-frame IBTrACS lat/lon + temporal embedding |
| **Backbones** | V1 — channel-stacked 2-D UNet · V2 — factorised spatial UNet + 1-D temporal attention |
| **Dataset** | SETCD = GRIDSAT-B1 (NOAA) + ERA5 (ECMWF) + IBTrACS v04r01, 2012–2022 |
| **Split** | Year-based: 2012–2020 train · 2021 val · 2022 test (no random storm splitting) |
| **Hardware** | RTX 4070 Ti SUPER 16 GB / RTX 4070 12 GB, BF16 AMP |

---

## Why this is interesting

Operational cyclone guidance treats satellite imagery and atmospheric state as separate inputs and predicts intensity or track with deterministic regressors. A single generative model that produces **the joint image + field state at future times** lets a forecaster:

- visualise the storm's future IR signature alongside the reanalysis fields that drive it,
- sample multiple plausible futures (rectified flow gives a tractable likelihood / sampling budget),
- and inspect track implicitly via the displacement of the predicted IR centre.

This thesis is the first work I'm aware of to apply **rectified flow matching** to joint multi-channel cyclone forecasting at this horizon and resolution.

---

## Repository layout

```
satellite/
├── README.md                  ← you are here
├── scripts/                   ← model code, training, evaluation (git submodule)
│   ├── assembly.py            ← single-config master orchestrator (VAE → cache → flow → eval)
│   ├── vae.py                 ← MultiChannelVAE (5ch → 4-dim latent, 256→64)
│   ├── multi_frame_flow.py    ← K_in/K_out rectified-flow trainer (V1 + V2)
│   ├── multi_channel_diffusion.py     ← MC-Diff baseline (DDPM in latent space)
│   ├── conditional_flow_matching.py   ← OT-CFM variant
│   ├── rectified_flow_score_matching.py
│   ├── latent_diffusion.py    ← GRIDSAT-only LDM baseline
│   ├── latent_cache.py        ← precompute VAE encodings for faster training
│   ├── multi_frame_dataset.py ← K_in past + K_out future windowed loader
│   ├── eval_metrics_extended.py ← PSNR / SSIM / FID / IS / LPIPS / CRPS / steering-flow
│   ├── physics_losses.py      ← divergence / vorticity / spectral penalties
│   ├── epoch_logger.py        ← per-epoch CSV + .log emitter
│   ├── analysis/              ← aggregation, figures, significance tests
│   └── eda/                   ← dataset stats, storm counts, coord lookup
├── docs/                      ← design plans (phase 0–3), result summaries, system notes
│   └── README.md              ← documentation index
├── dataset_stats/             ← global mean / std, storm-per-year counts, EDA plots
├── paper_results/             ← figures and CSVs that go into the paper
├── report/                    ← LaTeX manuscript + thesis report figures
└── ref_nath/                  ← Nath et al. baseline reproduction (Phase 3 external comparison)
```

Generated artifacts (`checkpoints/`, `outputs/`, `results/`, `logs/`, `cache/`, large zips) are gitignored.

---

## Method

### 1 · Multi-channel VAE (stage 1)

A 5-channel VAE compresses `(GRIDSAT, U, V, T, P)` at 256 × 256 into a 4-channel latent at 64 × 64 with KL-annealed L1 reconstruction. The decoder is unbounded (no `tanh`) so the network does not have to fight the wide dynamic range of pressure. Frozen for the rest of the pipeline.

### 2 · Latent rectified flow (stage 2)

A `DiffusionUNet` with `base_channels=128`, `channel_mults=(1, 2, 3, 4)` and attention at the bottleneck learns the 1-RF velocity field

```
v_θ(z_t, t, ctx) ≈ z_1 − z_0
```

over straight-line interpolants `z_t = (1−t) z_0 + t z_1`. `ctx` carries the K_in past-frame latents, per-frame IBTrACS coordinates, and a temporal embedding. Two backbones are ablated:

- **V1 — channel-stack.** Past and future frames stacked along the channel axis. Cheapest, no cross-frame mixing.
- **V2 — factorised spatial + 1-D temporal attention.** Per-frame UNet with a temporal self-attention layer at the bottleneck attending across the K_in + K_out token sequence.

Sampling uses Euler with a 1 / 4 / 16 / 50 step budget sweep.

### 3 · Evaluation

- **Image fidelity:** PSNR, SSIM, FID, IS, LPIPS per channel and per lead time.
- **Forecast skill:** CRPS over an ensemble of 10 samples.
- **Track skill:** steering-flow displacement (predicted IR centre vs IBTrACS best track) reported in kilometres per lead time.
- **External baseline:** Nath et al. multi-modal multi-scale diffusion (`ref_nath/`).

---

## Quick start

> Note: dataset access (GRIDSAT-B1 NetCDFs, ERA5 hourly fields, IBTrACS v04r01 best track) is required and is **not redistributed** in this repo. The data is licensed for research use through NOAA/ECMWF/IBTrACS and is staged outside the repo via the `THESIS_DATA_ROOT` environment variable.

### Environment

```bash
python -m pip install torch torchvision numpy pandas scipy matplotlib tqdm netcdf4 xarray scikit-image lpips
```

Tested on Python 3.10 / PyTorch 2.x / CUDA 12.

### One-shot training pipeline

```bash
# trains MC-VAE → precomputes latent cache → trains multi-frame 1-RF → evaluates
python scripts/assembly.py
```

All hyperparameters live in the single `CONFIG` dict at the top of [scripts/assembly.py](scripts/assembly.py). Stage toggles (`train_vae`, `precompute_cache`, `train_multi_frame_rf`, `aggregate`) let you run the pipeline incrementally.

### Sanity check (single-frame K=1, K=1)

```bash
python scripts/assembly.py --single-frame
```

### Direct multi-frame flow run (auto-resumes from latest checkpoint)

```bash
python -u scripts/multi_frame_flow.py
```

---

## Design choices worth flagging

- **Year-based split (no random storm split).** Two runs with different seeds would otherwise pick different validation storms. Reviewers reject that. Fixed years are reproducible.
- **Coordinates as conditioning, not VAE input.** Encoding constant lat/lon maps through the VAE wasted capacity; a small `Linear(2 → 64 → 128)` added to the temporal embedding plus a `CoordPredictor` MLP head works better.
- **Dropped 2-RF / reflow.** Reflow needs the initial flow's val v-MSE below ~0.2; ours plateaued around 0.5, so the reflowed model collapsed (val 2.6 while train → 0). The 1-RF model is the headline.
- **BF16 over FP16.** Ada Lovelace handles BF16 cleanly; FP16 grad-scaler quirks were the cause of multiple NaN incidents.

For the full rationale see [docs/](docs/README.md).

---

## Status

- Phase 0 — research plan ✅
- Phase 1 — feature parity + single-frame 1-RF reference ✅
- Phase 2 — multi-frame K_in=3 K_out=3 V2 training 🔄 *in progress*
- Phase 3 — external Nath baseline + paper figures ⏳

---

## License

Research code released as-is for academic review. Dataset files (GRIDSAT-B1, ERA5, IBTrACS) retain their respective upstream licenses and are **not** included.
