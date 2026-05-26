# Satellite Diffusion Model (SDM) — Project Overview

## Research Goal

Build a **latent diffusion model** that, given the current state of a tropical cyclone (satellite imagery + atmospheric reanalysis fields), generates the **next-timestep** state while simultaneously predicting the storm's geographic trajectory.

The model should produce:

1. **GRIDSAT satellite image** (1 channel — infrared brightness temperature)
2. **ERA5 reanalysis fields** (4 channels — U-wind, V-wind, Temperature, Surface Pressure)
3. **Predicted coordinates** (latitude, longitude of the storm center at T+1)

---

## Dataset: SETCD (Satellite-ERA5 Tropical Cyclone Dataset)

| Property | Value |
|----------|-------|
| Source | GRIDSAT-B1 (NOAA) + ERA5 (ECMWF) + IBTrACS v04r01 |
| Resolution | 256×256 pixels per field |
| Training Years | 2006–2012 |
| Test Storm | `2022349N13068` |
| Validation | 35 randomly held-out storms |
| Samples | ~18,609 train / ~930 val |
| Structure | `/{year}/{SID}/{timestamp}/GRIDSAT_data.npy` and `ERA5_data.npy` |

**Data Format per sample:**
- `GRIDSAT_data.npy` — shape `(1, H, W)`, infrared brightness temperature
- `ERA5_data.npy` — shape `(4, H, W)`, channels: [U-wind, V-wind, Temp, Pressure]

**Normalization:** Global z-score (computed from 5,000-sample scan)

| Channel | Mean | Std |
|---------|------|-----|
| GRIDSAT | 169.90 | 122.26 |
| U-wind | -1.17 | 5.39 |
| V-wind | 0.38 | 5.00 |
| Temperature | 298.49 | 4.23 |
| Pressure | 100,976.29 | 521.15 |

**Coordinate Lookup:** Per-timestep lat/lon from IBTrACS best-track data, built by `build_coordinate_lookup.py` and stored in `coordinate_lookup.json`.

---

## Architecture (Current: v2 — Coord-Conditioned)

### Two-Stage Pipeline

```
Stage 1: MultiChannelVAE (5ch)     Stage 2: Diffusion UNet + CoordPredictor
┌─────────────────────────┐        ┌──────────────────────────────────────┐
│ Encoder: 5ch → 4-dim z  │        │ Input: noise z ~ N(0,I)             │
│ Decoder: 4-dim z → 5ch  │        │ Condition: GRIDSAT + ERA5 + coords  │
│ No Tanh (unbounded)     │        │ Output: denoised z → decode → 5ch   │
│ L1 + KL loss            │        │ + CoordPredictor(ctx) → (lat, lon)  │
└─────────────────────────┘        └──────────────────────────────────────┘
```

### Key Components (in `multi_channel_diffusion.py`)

| Component | Class | Description |
|-----------|-------|-------------|
| Dataset | `CycloneDataset` | Loads GRIDSAT+ERA5 pairs, applies z-score norm, looks up coords from JSON |
| VAE | `MultiChannelVAE` | 5-channel VAE, `in=5, latent=4, base_ch=64`, no Tanh |
| VAE Trainer | `MultiChannelVAETrainer` | Stage 1: L1 + KL loss with KL annealing |
| UNet | `DiffusionUNet` | Latent-space noise predictor, attention at mid-level |
| Condition Encoder | `ConditionEncoder` | Encodes GRIDSAT+ERA5+temporal+coords → conditioning |
| Coord Predictor | `CoordPredictor` | MLP: context_emb(256) → 128 → 64 → 2 (lat, lon) |
| Noise Scheduler | `LinearNoiseScheduler` | Linear β schedule, DDIM sampling |
| Diffusion Trainer | `MultiChannelDiffusionTrainer` | Stage 2: noise MSE + coord MSE joint loss |

### Coordinate Integration (v2 Design Decision)

Coordinates are used as **conditioning signals**, NOT as VAE input channels.

- `ConditionEncoder.coord_proj`: Linear(2 → 64 → 128), added to temporal embedding
- `CoordPredictor`: Separate MLP head on top of context embedding, supervised with MSE loss
- **Rationale**: Encoding constant spatial maps through the VAE was wasteful — the VAE struggled to reconstruct them alongside the image channels. Conditioning keeps the VAE focused on image reconstruction while the MLP handles trajectory.

---

## Other Scripts

| File | Purpose |
|------|---------|
| `latent_diffusion.py` | Single-channel (GRIDSAT-only) latent diffusion baseline |
| `diffusion_unet.py` | Standalone UNet architecture (shared building blocks) |
| `build_coordinate_lookup.py` | Downloads IBTrACS, matches to dataset timestamps, outputs JSON |
| `dataloader.py` | Generic cyclone dataloader (used by some scripts) |
| `compute_dataset_stats.py` | Computes global mean/std for normalization |
| `conditional_flow_matching.py` | Alternative generative approach (experimental) |
| `rectified_flow_score_matching.py` | Another alternative approach (experimental) |
| `refinement_gan.py` | Post-hoc GAN refinement (experimental) |

---

## Current Training Config

```python
# VAE
latent_dim = 4, vae_base_channels = 64, vae_epochs = 50
vae_lr = 1e-4, kl_weight = 1e-4, kl_anneal: 0→20 epochs

# Diffusion
base_channels = 64, channel_mults = (1, 2, 4), num_heads = 4
embed_dim = 128, t_emb_dim = 128
num_timesteps = 1000, beta = [1e-4, 0.02]
diffusion_epochs = 100, lr = 1e-4 → 5e-5 (CosineAnnealing, T_max=200)
coord_loss_weight = 1.0
gradient_accumulation_steps = 4, AMP = False

# Evaluation
eval_every = 5 epochs, eval_samples = 200
```
