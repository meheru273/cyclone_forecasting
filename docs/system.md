# Satellite Cyclone Diffusion Model (SDM) — System Documentation

## Overview

A **two-stage Latent Diffusion Model** for cyclone satellite image forecasting. Given the current timestep's GRIDSAT (infrared satellite) and ERA5 (atmospheric reanalysis) data, the model predicts the next timestep's GRIDSAT image.

```
Input: GRIDSAT_t (1ch, 256×256) + ERA5_t (4ch, 256×256)
  ↓
Stage 1: VAE encodes target to latent space (4ch, 64×64)
Stage 2: UNet denoises latent conditioned on inputs
  ↓
Output: Predicted GRIDSAT_{t+1} (1ch, 256×256)
```

---

## Architecture

### Stage 1: Variational Autoencoder (VAE)

Compresses 256×256 single-channel satellite images to a 4×64×64 latent space.

| Component | Details |
|-----------|---------|
| Encoder | 3 Conv layers (stride-2 downsample), GroupNorm, SiLU |
| Latent dim | 4 channels at 64×64 spatial |
| Decoder | Conv + Upsample(nearest) + Conv (artifact-free upsampling) |
| Loss | L1 reconstruction + KL divergence (annealed β from epoch 0→20) |
| Scale factor | ~0.18215 (computed from latent statistics post-training) |

**Key design choice**: Decoder uses `nn.Upsample(nearest) + nn.Conv2d` instead of `nn.ConvTranspose2d` to avoid horizontal line / checkerboard artifacts.

### Stage 2: Conditional Diffusion UNet

Denoises latent representations conditioned on current weather state.

| Component | Details |
|-----------|---------|
| Base channels | 128 |
| Channel multipliers | (1, 2, 3, 4) → [128, 256, 384, 512] |
| Encoder levels | 4 (with downsampling at first 3 levels) |
| Attention | Self-attention at 16×16 resolution (level 2) + bottleneck |
| Attention heads | 8 |
| Time embedding | Sinusoidal → MLP (dim 128 → 512) |
| Condition injection | Spatial concatenation at input + FiLM via time+context embedding |
| Noise schedule | **Cosine** (Nichol & Dhariwal 2021) — 1000 timesteps |
| Sampling | DDIM (50 steps, η=0) |
| Upsampling | `nn.Upsample(nearest) + nn.Conv2d` (artifact-free) |

### Condition Encoder

Encodes the input weather state (GRIDSAT + ERA5) into conditioning features:

1. **Spatial**: GRIDSAT (1ch) + ERA5 (4ch) → Conv layers → 128ch feature maps at 256×256
2. **Temporal**: Timestamp → learned embeddings (year, month, day, hour) → MLP → 256-dim vector
3. **Fusion**: Spatial features + broadcast temporal embedding → 128ch at 256×256
4. **Downsampling**: Condition features downsampled 4× to match latent resolution (64×64)

---

## Training Configuration

### Hyperparameters

```python
config = {
    # Data
    'img_size': 256,
    'batch_size': 4,
    'gradient_accumulation_steps': 4,  # effective batch = 16

    # VAE (Stage 1)
    'latent_dim': 4,
    'vae_base_channels': 64,
    'vae_epochs': 50,
    'vae_lr': 1e-4,
    'kl_weight': 1e-4,
    'recon_loss_type': 'l1',

    # Diffusion (Stage 2)
    'base_channels': 128,
    'channel_mults': (1, 2, 3, 4),
    'num_heads': 8,
    'num_timesteps': 1000,
    'noise_schedule': 'cosine',
    'diffusion_epochs': 200,
    'learning_rate': 2e-4,
    'min_lr': 1e-6,
    'ema_decay': 0.9999,
    'use_amp': True,  # mixed precision
}
```

### Training Pipeline

1. **Stage 1 — VAE Training** (50 epochs)
   - Train VAE on target GRIDSAT images
   - KL weight annealed from 0 → 1e-4 over epochs 0–20
   - Compute `latent_scale_factor` from validation set statistics
   - Save `vae_best.pt` checkpoint

2. **Stage 2 — Diffusion Training** (200 epochs)
   - Freeze VAE encoder/decoder
   - Train UNet to predict noise in latent space
   - Cosine LR schedule: 2e-4 → 1e-6
   - EMA with decay 0.9999 for stable evaluation
   - Evaluation with PSNR, SSIM, FID, IS every 5 epochs

### Noise Schedule

Uses **cosine schedule** (Nichol & Dhariwal 2021) which distributes the learning signal more evenly across timesteps compared to linear schedule. This is particularly important for latent diffusion where the latent space has different statistical properties than pixel space.

```
Cosine: ᾱ(t) = cos²((t/T + s) / (1+s) · π/2),  s=0.008
```

### LR Schedule

```
CosineAnnealingLR: T_max = diffusion_epochs, η_min = 1e-6
```

The learning rate decays from 2e-4 to 1e-6 over the full training run, providing strong initial learning and fine-grained convergence at the end.

---

## Data Format

### Input Data Structure
```
cyclone_dataset/
├── 2006/
│   ├── STORM_ID_1/
│   │   ├── 2006-01-01 00_00_00/
│   │   │   ├── GRIDSAT_data.npy  (H×W or 1×H×W, float32)
│   │   │   └── ERA5_data.npy     (4×H×W, float32: u10, v10, t2m, sp)
│   │   ├── 2006-01-01 06_00_00/
│   │   │   ├── GRIDSAT_data.npy
│   │   │   └── ERA5_data.npy
│   │   └── ...
│   └── STORM_ID_2/
│       └── ...
├── 2007/
└── ...
```

### Normalization (Hardcoded Global Stats)

| Channel | Mean | Std |
|---------|------|-----|
| GRIDSAT | 169.8992 | 122.2612 |
| ERA5 u10 | -1.1655 | 5.3889 |
| ERA5 v10 | 0.3835 | 5.0042 |
| ERA5 t2m | 298.4892 | 4.2326 |
| ERA5 sp | 100976.29 | 521.15 |

### Data Augmentation
- Random horizontal flip (50% probability, training only)
- All inputs resized to 256×256 via bilinear interpolation

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| MSE | Mean Squared Error (noise prediction) | Lower is better |
| MAE | Mean Absolute Error (pixel space) | Lower is better |
| PSNR | Peak Signal-to-Noise Ratio | Higher is better (>25 dB good) |
| SSIM | Structural Similarity Index | Higher is better (>0.7 good) |
| FID | Fréchet Inception Distance | Lower is better (<50 good) |
| IS | Inception Score | Higher is better |

---

## File Structure

```
scripts/
├── latent_diffusion.py     # Main model + training script (VAE + Diffusion)
docs/
├── system.md               # This file
checkpoints/
├── vae/
│   ├── vae_best.pt         # Best VAE checkpoint
│   ├── vae_latest.pt       # Latest VAE checkpoint
│   └── vae_final.pt        # Final VAE with scale factor
└── diffusion/
    ├── diffusion_best.pt   # Best diffusion checkpoint (EMA weights)
    ├── diffusion_latest.pt # Latest diffusion checkpoint
    └── diffusion_epoch_*.pt
outputs/
├── vae/
│   ├── vae_training_curves.png
│   ├── vae_recon_epoch_*.png
│   └── metrics/
└── diffusion/
    ├── diffusion_training_curves.png
    ├── samples_epoch_*.png
    ├── final_eval_metrics.json
    └── metrics/
        └── diffusion_training_metrics.csv
```

---

## Checkpoint Resume

The training pipeline supports automatic checkpoint resumption:
- **VAE**: Resumes from `checkpoints/vae/vae_latest.pt`
- **Diffusion**: Pass `diffusion_checkpoint_path` to resume

> **Important**: After architecture changes (e.g., changing `base_channels`, `channel_mults`), old checkpoints are incompatible. Delete or move old checkpoint files before retraining.

---

## Known Design Decisions

1. **Upsample + Conv over ConvTranspose2d**: Eliminates checkerboard/horizontal line artifacts in both VAE decoder and UNet decoder.
2. **Cosine over Linear noise schedule**: Better gradient flow across all timesteps for latent diffusion.
3. **Unbiased validation**: Validation uses random timesteps matching training distribution, ensuring train/val loss are directly comparable.
4. **EMA for evaluation**: All sample generation and metrics use EMA weights for stability.
5. **Condition concatenation**: Conditions are spatially concatenated at UNet input and injected via FiLM (additive time+context embedding) at every ResNet block.
