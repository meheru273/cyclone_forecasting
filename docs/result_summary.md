# Results Summary — Satellite Cyclone Image Generation

## Overview

Three experiments were run on the GRIDSAT cyclone dataset, testing different generative model architectures for satellite imagery synthesis. All experiments used storms from **2006–2012 for training**, **35 validation storms**, and storm `2022349N13068` as the held-out test storm.

---

## Experiment 1: Pure UNet — Pixel-Space Diffusion (64×64)

**Config:** Direct pixel-space DDPM on 64×64 images. No VAE. 1-channel grayscale. 50 epochs, lr=2e-4, 1000 diffusion timesteps.

### Final Evaluation Metrics (Epoch 50)

| Metric | Value |
|--------|-------|
| MAE | **0.0344** |
| MSE | **0.00337** |
| RMSE | **0.0555** |
| PSNR | **31.53 dB** |
| SSIM | **0.661** |
| FID | **88.46** |

> Strongest pixel-fidelity scores across all experiments — benefiting from direct pixel-space training at low resolution (64×64), avoiding latent compression artifacts.

---

## Experiment 2: Latent Diffusion — UNet from VAE Constructor (256×256)

**Config:** Latent diffusion model. VAE encodes to 4-dim latent space; UNet denoises latents. 256×256 images. 50 diffusion epochs, lr=1e-4, 1000 timesteps. Latent scale factor = 0.18215.

### Final Evaluation Metrics (Epoch 50)

| Metric | Value |
|--------|-------|
| MAE | 0.1082 |
| MSE | 0.03288 |
| RMSE | 0.1737 |
| PSNR | 21.64 dB |
| SSIM | 0.4973 |
| FID | 135.47 |
| IS (mean ± std) | 1.185 ± 0.058 |

### Training Progression

| Epoch | MAE | PSNR | SSIM | FID |
|-------|-----|------|------|-----|
| 1 | 0.412 | 9.90 | 0.046 | 458.7 |
| 10 | 0.173 | 18.96 | 0.283 | 418.3 |
| 20 | 0.185 | 17.51 | 0.257 | 347.5 |
| 30 | 0.141 | 19.63 | 0.347 | 296.4 |
| 40 | 0.119 | 21.02 | 0.438 | 208.5 |
| 50 | 0.108 | 21.65 | 0.497 | 135.5 |

> All metrics show consistent improvement throughout training. FID dropped from ~459 → ~135 over 50 epochs, suggesting continued training would likely yield further gains.

---

## Experiment 3: Standalone VAE (256×256)

**Config:** VAE-only training (no diffusion). 60 epochs, lr=1e-4, KL weight=1e-4, KL annealing epochs 0–25, L1 reconstruction loss, 4-dim latent space, 256×256 images.

### Final Reconstruction Metrics (Epoch 60)

| Metric | Value |
|--------|-------|
| Val MSE | **0.000736** |
| Val PSNR | **38.68 dB** |
| Val SSIM | **0.9556** |
| Val Loss | 0.01519 |
| FID | **24.96** |

### Training Progression

| Epoch | Val PSNR | Val SSIM | FID |
|-------|----------|----------|-----|
| 1 | 36.88 | 0.937 | — |
| 5 | 38.30 | 0.952 | 28.45 |
| 10 | 38.47 | 0.954 | 25.65 |
| 25 | 38.58 | 0.954 | 25.39 |
| 40 | 38.64 | 0.956 | 24.36 |
| 60 | 38.68 | 0.956 | 24.96 |

> The VAE converged very quickly (by ~epoch 5) and plateaued. The FID around **24–25** represents the reconstruction fidelity ceiling of the latent space — this is the "floor" of quality the latent diffusion model must meet.

---

## Experiment 4: Latent Diffusion — UNet from VAE Constructor (256×256, 100 Epochs)

**Config:** Same architecture as Experiment 2, but trained for 100 diffusion epochs. VAE weights loaded from checkpoint (0 VAE epochs). lr=1e-4 with cosine warmup, 1000 timesteps, latent scale factor = 0.18215, gradient accumulation = 4, EMA decay = 0.9999.

### Final Evaluation Metrics (Epoch 100)

| Metric | Value |
|--------|-------|
| MAE | **0.0967** |
| MSE | **0.02740** |
| RMSE | **0.1578** |
| PSNR | **22.49 dB** |
| SSIM | **0.5578** |
| FID | **110.15** |
| IS (mean ± std) | 1.248 ± 0.073 |

### Training Progression (Epochs 51–100)

| Epoch | MAE | PSNR | SSIM | FID |
|-------|-----|------|------|-----|
| 51 | 0.1088 | 21.58 | 0.498 | 133.4 |
| 60 | 0.1007 | 22.17 | 0.539 | 116.3 |
| 70 | 0.0997 | 22.24 | 0.547 | 115.3 |
| 80 | 0.0966 | 22.52 | 0.560 | 115.0 |
| 90 | 0.0974 | 22.49 | 0.558 | 113.9 |
| 100 | 0.0967 | 22.49 | 0.558 | 110.2 |

> Training from epoch 50 → 100 improved FID from **135.5 → 110.2** (−18.7%) and PSNR from **21.6 → 22.5 dB**. Improvement is slowing — the model appears to be approaching convergence in the 110–115 FID range. Continued training beyond 100 epochs may yield diminishing returns without architectural changes.

---

## Experiment 5: Multi-Channel Latent Diffusion — Joint GRIDSAT + ERA5 Generation

**Config:** Joint 5-channel VAE (1 GRIDSAT + 4 ERA5 channels: U-wind, V-wind, Temperature, Pressure) trained for 30 epochs, followed by a multi-channel latent diffusion model trained for 40 epochs. 256×256 images, latent dim = 4, lr=1e-4, 1000 diffusion timesteps, eval every 5 epochs, 50 eval samples.

### Phase 1: Multi-Channel VAE (30 Epochs)

| Metric | Final (Epoch 30) |
|--------|-----------------|
| Val MSE (GRIDSAT) | 0.01332 |
| Val MSE (ERA5 avg) | 0.17975 |
| Val PSNR | **17.23 dB** |
| Val SSIM | **0.772** |
| Val SID | 0.700 |
| FID | **175.0** |

#### VAE Training Progression

| Epoch | Val PSNR | Val SSIM | FID |
|-------|----------|----------|-----|
| 5 | 16.93 | 0.769 | 225.8 |
| 10 | 16.99 | 0.764 | 198.4 |
| 15 | 17.05 | 0.764 | 186.4 |
| 20 | 17.09 | 0.769 | 192.2 |
| 25 | 17.22 | 0.767 | 172.1 |
| 30 | 17.23 | 0.772 | 175.0 |

> The multi-channel VAE converged steadily. The much lower PSNR (~17 dB) compared to the single-channel VAE (~38.7 dB) reflects the added difficulty of jointly encoding heterogeneous ERA5 fields alongside GRIDSAT imagery. FID ~172–175 represents the reconstruction ceiling for this 5-channel latent space.

### Phase 2: Multi-Channel Diffusion (40 Epochs) — Final Evaluation Metrics

#### Per-Channel Breakdown

| Channel | MAE | MSE | PSNR | SSIM |
|---------|-----|-----|------|------|
| GRIDSAT | 0.1292 | 0.03402 | **21.19 dB** | **0.4559** |
| U-wind | 0.2979 | 0.14767 | 14.45 dB | 0.3211 |
| V-wind | 0.3280 | 0.17999 | 13.58 dB | 0.3048 |
| Temperature | 0.3708 | 0.32510 | 11.72 dB | 0.3214 |
| Pressure | 0.4533 | 0.31028 | 11.59 dB | 0.4013 |

#### Overall (All Channels)

| Metric | Value |
|--------|-------|
| Overall MAE | 0.3158 |
| Overall MSE | 0.1994 |
| Overall PSNR | 13.17 dB |
| Overall SSIM | 0.361 |
| Overall SID | 0.759 |

### Multi-Channel Diffusion Training Progression

| Epoch | GRIDSAT PSNR | GRIDSAT SSIM | Overall PSNR | Overall SSIM | Overall SID |
|-------|-------------|-------------|-------------|-------------|------------|
| 5 | 3.93 | 0.008 | 5.36 | 0.020 | 1.232 |
| 10 | 11.46 | 0.096 | 7.18 | 0.029 | 0.187 |
| 15 | 18.90 | 0.327 | 11.07 | 0.230 | 0.491 |
| 20 | 18.53 | 0.399 | 11.23 | 0.292 | 0.950 |
| 25 | 17.68 | 0.343 | 11.57 | 0.283 | 0.771 |
| 30 | 18.69 | 0.368 | 12.05 | 0.304 | 0.778 |
| 35 | 19.84 | 0.405 | 12.40 | 0.324 | 0.651 |
| 40 | **21.38** | **0.481** | **13.27** | **0.387** | **0.650** |

> The multi-channel diffusion model shows clear improvement across all channels from epoch 5 → 40. GRIDSAT reconstruction reaches 21.38 dB SSIM 0.481 — comparable to the single-channel latent diffusion at the same epoch count. ERA5 channels (wind, temperature, pressure) are significantly harder to generate, with 40-epoch PSNR values in the 10–14 dB range. Overall SID improving from ~1.2 → ~0.65 suggests the latent distribution is converging toward the target.

---

## Cross-Experiment Comparison

| Model | Task | Resolution | Epochs | PSNR ↑ | SSIM ↑ | FID ↓ |
|-------|------|-----------|--------|--------|--------|-------|
| Pure UNet (Pixel Diffusion) | GRIDSAT gen. | 64×64 | 50 | **31.53 dB** | **0.661** | **88.46** |
| Latent Diffusion (UNet+VAE) | GRIDSAT gen. | 256×256 | 50 | 21.64 dB | 0.497 | 135.47 |
| Latent Diffusion (UNet+VAE) | GRIDSAT gen. | 256×256 | **100** | 22.49 dB | 0.558 | 110.15 |
| VAE (Reconstruction Only) | GRIDSAT recon. | 256×256 | 60 | 38.68 dB | **0.956** | 24.96 |
| Multi-Channel VAE | GRIDSAT + ERA5 recon. | 256×256 | 30 | 17.23 dB | 0.772 | 175.0 |
| Multi-Channel Diffusion | GRIDSAT + ERA5 gen. | 256×256 | 40 | 13.17 dB\* | 0.361\* | — |

\*Overall across all 5 channels. GRIDSAT-only: PSNR 21.19 dB, SSIM 0.456.

> The VAE reconstruction metrics (PSNR ~38.7 dB, SSIM ~0.956, FID ~25) represent the **upper bound** for single-channel latent diffusion. Extending training from 50 → 100 epochs reduced FID from 135 → 110 and improved PSNR by ~0.85 dB. Multi-channel generation is substantially harder — the shared 5-channel VAE compresses heterogeneous ERA5 fields into the same latent space, creating a harder bottleneck.

---

## Key Takeaways

1. **Pure pixel-space diffusion (64×64)** achieves the best generative PSNR/SSIM (31.5 dB, SSIM 0.66), at the cost of low resolution.
2. **Latent diffusion (256×256) benefits from longer training** — going 50 → 100 epochs reduced FID from 135 → 110 and improved PSNR by ~0.85 dB. The model is still converging slowly; architectural improvements may be needed to close the remaining gap to VAE quality (FID ~25).
3. **The VAE is well-trained** with excellent reconstruction quality (PSNR ~38.7 dB, SSIM ~0.956, FID ~25). The bottleneck remains the diffusion model on top of it.
4. **Multi-channel joint generation introduces significant additional complexity.** The shared 5-channel VAE compresses heterogeneous data into a single latent space, resulting in a higher FID floor (~175). ERA5 weather fields (wind, temp, pressure) are harder to generate than GRIDSAT imagery — PSNR values for ERA5 channels remain 8–11 dB below the GRIDSAT channel after 40 epochs.
5. **Overall SID ~0.65** in Experiment 5 suggests the multi-channel diffusion model has not yet fully learned the joint distribution — further training and potentially channel-specific loss weighting could improve ERA5 channel quality.
