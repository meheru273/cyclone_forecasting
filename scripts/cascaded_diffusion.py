"""
Phase-3 external baseline (R4): Nath et al.-style cascaded pixel-space diffusion,
adapted to *our* setting for an apples-to-apples comparison with the flow models.

Reference: Nath et al., "Forecasting Tropical Cyclones with Cascaded Diffusion
Models" (arXiv:2310.01690). Their released code
(github.com/nathzi1505/forecast-diffmodels) builds a multi-stage pixel-space
DDPM cascade on top of lucidrains/imagen-pytorch: a low-resolution base model
followed by diffusion super-resolution stages, conditioned via classifier-free
guidance.

This module re-implements the *defining architectural idea* — a progressive
pixel-space DDPM cascade (64² → 128² → 256²) with classifier-free guidance —
using this project's own DDPM primitives, so the baseline shares the dataset,
split, channel layout, and evaluation suite with R5/R6/R7. Differences from the
original, all to make the comparison fair (documented in docs/phase3.md §3.3.2):

  * Output is the joint 5-channel state (IR + 4 ERA5), not IR-only.
  * Multi-frame: conditions on K_in past frames, predicts K_out future frames
    (channel-stacked), matching the headline flow rows.
  * Final stage runs at 256², our headline resolution.
  * Trained on `MultiFrameCycloneDataset` with the year-based split.

What stays faithful to Nath et al.:

  * Pixel space (no VAE) — the architectural contrast against our latent flow.
  * ε-prediction DDPM with a cosine noise schedule.
  * Cascaded super-resolution: each stage refines the previous stage's output.
  * Classifier-free guidance (`cond_drop_prob` at train, `cond_scale` at sample).

Outputs, per run_tag, mirror the multi-frame trainer so the same aggregation /
plotting tooling consumes them:

    checkpoints/<run_tag>/cascade_<stage>_best.pt
    outputs/<run_tag>/metrics/<run_tag>_metrics.csv      (per stage, via EpochLogger)
    outputs/<run_tag>/metrics/per_leadtime.csv           (final cascade, 256²)
    outputs/<run_tag>/metrics/sampler_steps.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multi_channel_diffusion import (
    CosineNoiseScheduler,
    DownBlock, MidBlock, UpBlock, get_time_embeddings,
)
from latent_diffusion import EMA
from multi_frame_dataset import get_multi_frame_dataloaders
from eval_metrics_extended import (
    MetricsCalculator,
    lpips_metric, threshold_skill_scores, rapsd_l1,
    mslp_err_hpa, max_wind_err, great_circle_km, denormalise_gridsat,
)
from epoch_logger import EpochLogger


# ============================================================================
# Cascade UNet — plain conditional ε-predictor (conditioning via concatenation)
# ============================================================================

class CascadeUNet(nn.Module):
    """Pixel-space ε-prediction UNet for one cascade stage.

    Conditioning is concatenated to the noisy target on the channel axis. The
    first `context_channels` of the conditioning carry the temporal context
    (the K_in past frames); they are the slice zeroed for the unconditional
    pass under classifier-free guidance. Any remaining conditioning channels
    (the upsampled low-resolution prediction in super-resolution stages) are
    always kept.
    """

    def __init__(self, target_channels: int, cond_channels: int,
                 base_channels: int = 64, channel_mults=(1, 2, 4, 8),
                 t_emb_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        n_levels = len(channel_mults)
        ch_list = [base_channels * m for m in channel_mults]
        emb_dim = t_emb_dim * 4

        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.init_conv = nn.Conv2d(target_channels + cond_channels, ch_list[0], 3, padding=1)

        # Encoder — attention ONLY at the deepest (bottleneck) level. In pixel
        # space the shallower levels run at 64²–256², where full self-attention
        # is quadratic in pixel count and explodes memory. This matches Nath
        # et al.'s super-resolution UNets (layer_attns=(F,F,F,T)).
        self.encoders = nn.ModuleList()
        for i in range(n_levels):
            in_ch = ch_list[i - 1] if i > 0 else ch_list[0]
            out_ch = ch_list[i]
            is_last = (i == n_levels - 1)
            self.encoders.append(DownBlock(
                in_ch, out_ch, emb_dim, down_sample=not is_last,
                use_attention=(i == n_levels - 1), num_heads=num_heads))

        self.mid_block = MidBlock(ch_list[-1], emb_dim, num_heads=num_heads)

        # Dynamic decoder mirror — attention only at the deepest decoder block
        # (i == 0), which runs at the bottleneck resolution.
        decoder_cfgs = []
        for i in range(n_levels):
            if i == 0:
                decoder_cfgs.append((ch_list[-1], ch_list[-1], ch_list[-1], False, True))
            else:
                in_ch_d = ch_list[n_levels - i]
                skip_ch_d = ch_list[n_levels - i - 1]
                out_ch_d = ch_list[n_levels - i - 1]
                decoder_cfgs.append((in_ch_d, skip_ch_d, out_ch_d, True, False))
        self.decoders = nn.ModuleList([
            UpBlock(ic, sc, oc, emb_dim, up_sample=up, use_attention=at, num_heads=num_heads)
            for ic, sc, oc, up, at in decoder_cfgs])

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch_list[0]), nn.SiLU(),
            nn.Conv2d(ch_list[0], target_channels, 3, padding=1))

    def forward(self, noisy, cond, t):
        t_emb = self.t_proj(get_time_embeddings(t, self.t_emb_dim))
        x = torch.cat([noisy, cond], dim=1)
        x = self.init_conv(x)
        skips = []
        for enc in self.encoders:
            x, skip = enc(x, t_emb)
            skips.append(skip)
        x = self.mid_block(x, t_emb)
        for i, dec in enumerate(self.decoders):
            x = dec(x, t_emb, skips[len(skips) - 1 - i])
        return self.final_conv(x)


# ============================================================================
# Trainer
# ============================================================================

class CascadedDiffusionTrainer:
    """Trains the three cascade stages sequentially and evaluates end-to-end."""

    def __init__(self, config: dict):
        self.cfg = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.k_in = int(config['k_in'])
        self.k_out = int(config['k_out'])
        self.n_ch = 5                       # 1 GRIDSAT IR + 4 ERA5
        self.target_channels = self.k_out * self.n_ch
        self.context_channels = self.k_in * self.n_ch

        self.use_amp = config.get('use_amp', True)
        self.amp_dtype = torch.bfloat16 if config.get('amp_dtype', 'bfloat16') == 'bfloat16' \
            else torch.float16

        # Diffusion process (shared across stages).
        self.num_timesteps = int(config.get('num_timesteps', 250))   # Nath: 250
        self.scheduler = CosineNoiseScheduler(self.num_timesteps, device=self.device)

        # Prediction parameterisation. 'v' (Salimans & Ho 2022) reconstructs the
        # target far more reliably than 'eps' when the conditioning is strongly
        # informative (forecasting ≈ persistence) — eps-prediction let the base
        # stage ignore its conditioning and sample near-unconditional noise.
        self.prediction_type = config.get('prediction_type', 'v')
        # Suffix so v-pred checkpoints never collide with / get skipped by the
        # old eps-pred ones.
        self.ckpt_suffix = '' if self.prediction_type == 'eps' else f'_{self.prediction_type}'

        # Classifier-free guidance.
        self.cond_drop_prob = float(config.get('cond_drop_prob', 0.1))   # Nath: 0.1
        self.cond_scale = float(config.get('cond_scale', 3.0))           # Nath: 3.0

        # Cascade stage definitions: resolution and the low-res it refines.
        self.stages = [
            {'name': 'base64', 'res': 64,  'low_res': None},
            {'name': 'sr128',  'res': 128, 'low_res': 64},
            {'name': 'sr256',  'res': 256, 'low_res': 128},
        ]

        # Per-stage optimisation knobs (lists or scalars).
        self.epochs       = config.get('cascade_epochs', [60, 50, 40])
        self.batch_sizes  = config.get('cascade_batch_sizes', [32, 16, 6])
        self.lr           = float(config.get('learning_rate', 1e-4))
        self.min_lr       = float(config.get('min_lr', 5e-5))
        self.warmup_epochs = int(config.get('warmup_epochs', 5))

        base_ch = int(config.get('cascade_base_channels', 64))
        mults = tuple(config.get('cascade_channel_mults', (1, 2, 4, 8)))
        num_heads = int(config.get('num_heads', 4))

        # Build one UNet (+ EMA) per stage.
        self.unets, self.emas = [], []
        for st in self.stages:
            lowres_ch = self.target_channels if st['low_res'] is not None else 0
            cond_ch = self.context_channels + lowres_ch
            unet = CascadeUNet(
                target_channels=self.target_channels, cond_channels=cond_ch,
                base_channels=base_ch, channel_mults=mults,
                t_emb_dim=int(config.get('t_emb_dim', 128)), num_heads=num_heads,
            ).to(self.device)
            self.unets.append(unet)
            self.emas.append(EMA(unet, decay=float(config.get('ema_decay', 0.9999))))

        n_params = sum(sum(p.numel() for p in u.parameters()) for u in self.unets)
        print(f"[Cascade] {len(self.stages)} stages, total UNet params: {n_params:,}, "
              f"K_in={self.k_in}, K_out={self.k_out}, T={self.num_timesteps}, "
              f"pred={self.prediction_type}, CFG(drop={self.cond_drop_prob}, scale={self.cond_scale})")

        # Output dirs.
        self.run_tag = config['run_tag']
        self.ckpt_dir = Path(config['checkpoint_dir']) / self.run_tag
        self.out_dir = Path(config['output_dir']) / self.run_tag
        self.metrics_dir = self.out_dir / 'metrics'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.eval_samples = int(config.get('eval_samples', 200))
        self.eval_every = int(config.get('eval_every', 5))
        self.ddim_steps = int(config.get('eval_sampler_steps', 50))
        self.sampler_step_sweep = list(config.get('sampler_step_sweep', [10, 25, 50, 100]))
        self.crps_n_samples = int(config.get('crps_n_samples', 10))

        # Dataset normalisation stats, filled from the loader on first use.
        self.era5_mean = None
        self.era5_std = None
        self.gs_mean = None
        self.gs_std = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _epochs_for(self, stage_idx):
        e = self.epochs
        return int(e[stage_idx]) if isinstance(e, (list, tuple)) else int(e)

    def _bs_for(self, stage_idx):
        b = self.batch_sizes
        return int(b[stage_idx]) if isinstance(b, (list, tuple)) else int(b)

    def _capture_stats(self, loader):
        if self.era5_mean is not None:
            return
        ds = loader.dataset
        self.gs_mean = float(ds.gs_mean.flatten()[0])
        self.gs_std = float(ds.gs_std.flatten()[0])
        self.era5_mean = ds.era5_mean.flatten().tolist()
        self.era5_std = ds.era5_std.flatten().tolist()

    def _prep_batch(self, batch):
        """Stacks frames into channel-flattened blocks at native 256²."""
        gridsat_in = batch['gridsat_in'].to(self.device)        # (B, K_in, 1, H, W)
        era5_in = batch['era5_in'].to(self.device)              # (B, K_in, 4, H, W)
        target_gs = batch['target_gs'].to(self.device)          # (B, K_out, 1, H, W)
        target_era5 = batch['target_era5'].to(self.device)      # (B, K_out, 4, H, W)

        input_block = torch.cat([gridsat_in, era5_in], dim=2)    # (B, K_in, 5, H, W)
        target_block = torch.cat([target_gs, target_era5], dim=2)  # (B, K_out, 5, H, W)
        B, _, _, H, W = input_block.shape
        input_flat = input_block.reshape(B, self.context_channels, H, W)
        target_flat = target_block.reshape(B, self.target_channels, H, W)

        return {
            'input_flat':   input_flat,        # (B, K_in*5, 256, 256)
            'target_flat':  target_flat,       # (B, K_out*5, 256, 256)
            'target_block': target_block,      # (B, K_out, 5, 256, 256)
            'cur_lat':      batch['raw_lat_in'][:, -1].to(self.device).float(),
            'cur_lon':      batch['raw_lon_in'][:, -1].to(self.device).float(),
        }

    @staticmethod
    def _resize(x, res):
        if x.shape[-1] == res:
            return x
        return F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)

    def _build_cond(self, stage, input_flat, lowres_pred=None):
        """Assemble the conditioning tensor at the stage resolution.

        Layout: [context (K_in*5) | lowres (K_out*5, SR stages only)].
        Context is always first so the unconditional pass can zero exactly
        cond[:, :context_channels].
        """
        res = stage['res']
        ctx = self._resize(input_flat, res)
        if stage['low_res'] is None:
            return ctx
        # lowres_pred is at the stage's low_res; upsample to stage res.
        low = self._resize(lowres_pred, res)
        return torch.cat([ctx, low], dim=1)

    def _drop_context(self, cond):
        """Return a copy of `cond` with the temporal-context channels zeroed
        (the unconditional input for classifier-free guidance)."""
        u = cond.clone()
        u[:, :self.context_channels] = 0.0
        return u

    # ------------------------------------------------------------------
    # Per-stage training
    # ------------------------------------------------------------------

    def _stage_targets(self, stage, prepped):
        """Ground-truth target (at stage res) and the GT low-res conditioning
        (downsampled target from the previous stage's resolution)."""
        res = stage['res']
        target = self._resize(prepped['target_flat'], res)
        lowres_gt = None
        if stage['low_res'] is not None:
            lowres_gt = self._resize(prepped['target_flat'], stage['low_res'])
        return target, lowres_gt

    def _stage_train_step(self, stage_idx, prepped):
        stage = self.stages[stage_idx]
        unet = self.unets[stage_idx]
        target, lowres_gt = self._stage_targets(stage, prepped)
        cond = self._build_cond(stage, prepped['input_flat'], lowres_pred=lowres_gt)

        B = target.shape[0]
        # Classifier-free guidance: drop temporal context for a random subset.
        if self.cond_drop_prob > 0:
            drop = (torch.rand(B, device=self.device) < self.cond_drop_prob)
            if drop.any():
                cond = cond.clone()
                cond[drop, :self.context_channels] = 0.0

        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        noise = torch.randn_like(target)
        noisy = self.scheduler.add_noise(target, noise, t)
        model_target = self._model_target(target, noise, t)

        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            pred = unet(noisy, cond, t)
            loss = F.mse_loss(pred, model_target)
        return loss

    # ----- v / eps parameterisation helpers -----

    def _sched_coef(self, t):
        """sqrt(alpha_cumprod) and sqrt(1-alpha_cumprod) at t, shaped to broadcast.
        Accepts a (B,) long tensor or a python int."""
        if isinstance(t, int):
            sa = self.scheduler.sqrt_alpha_cumprod[t]
            soma = self.scheduler.sqrt_one_minus_alpha_cumprod[t]
            return sa, soma
        sa = self.scheduler.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        soma = self.scheduler.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sa, soma

    def _model_target(self, x0, noise, t):
        """Regression target for the chosen parameterisation."""
        if self.prediction_type == 'eps':
            return noise
        sa, soma = self._sched_coef(t)        # v = sqrt_acp*noise - sqrt_omacp*x0
        return sa * noise - soma * x0

    def _to_eps(self, model_out, x_t, t):
        """Convert the model output to an ε estimate so the existing DDIM step
        (which consumes ε) can be reused regardless of parameterisation."""
        if self.prediction_type == 'eps':
            return model_out
        sa, soma = self._sched_coef(t)        # eps = sqrt_acp*v + sqrt_omacp*x_t
        return sa * model_out + soma * x_t

    def train_stage(self, stage_idx, train_loader, val_loader):
        stage = self.stages[stage_idx]
        unet = self.unets[stage_idx]
        ema = self.emas[stage_idx]
        n_epochs = self._epochs_for(stage_idx)

        best_ckpt = self.ckpt_dir / f"cascade_{stage['name']}{self.ckpt_suffix}_best.pt"
        latest_ckpt = self.ckpt_dir / f"cascade_{stage['name']}{self.ckpt_suffix}_latest.pt"
        if best_ckpt.exists():
            print(f"[Cascade] stage {stage['name']}: loading existing {best_ckpt.name} (skip train)")
            self._load_stage(stage_idx, best_ckpt)
            return

        optimizer = AdamW(unet.parameters(), lr=self.lr,
                          weight_decay=float(self.cfg.get('weight_decay', 1e-5)))
        scheduler = CosineAnnealingLR(optimizer, T_max=max(n_epochs - self.warmup_epochs, 1),
                                      eta_min=self.min_lr)
        logger = EpochLogger(self.metrics_dir, run_tag=f"{self.run_tag}_{stage['name']}")

        print(f"\n{'-' * 60}\n[Cascade] training stage {stage_idx+1}/3 "
              f"'{stage['name']}' @ {stage['res']}²  ({n_epochs} epochs)\n{'-' * 60}")
        best_val = float('inf')
        for epoch in range(n_epochs):
            # LR warmup then cosine.
            if epoch < self.warmup_epochs:
                warm = self.lr * (epoch + 1) / max(self.warmup_epochs, 1)
                for g in optimizer.param_groups:
                    g['lr'] = warm

            unet.train()
            logger.start_epoch()
            pbar = tqdm(train_loader, desc=f"{stage['name']} Ep {epoch+1}")
            for batch in pbar:
                prepped = self._prep_batch(batch)
                optimizer.zero_grad()
                loss = self._stage_train_step(stage_idx, prepped)
                if not torch.isfinite(loss):
                    logger.log_nan_skip()
                    optimizer.zero_grad()
                    continue
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                if not torch.isfinite(gnorm):
                    logger.log_nan_skip()
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                ema.update(unet)
                logger.log_batch(loss=loss.item(), grad_norm=gnorm.item())
                pbar.set_postfix({'mse': f"{loss.item():.4f}"})

            if epoch >= self.warmup_epochs:
                scheduler.step()

            val_loss = self._stage_validate(stage_idx, val_loader)
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                self._save_stage(stage_idx, best_ckpt, epoch, val_loss)
            self._save_stage(stage_idx, latest_ckpt, epoch, val_loss)

            logger.end_epoch(
                epoch + 1, phase=stage['name'],
                lr=optimizer.param_groups[0]['lr'], is_best=is_best,
                eval_metrics={'val_loss': val_loss},
                val_loss=val_loss,
            )
            print(f"  [{stage['name']}] epoch {epoch+1}/{n_epochs}  "
                  f"val_eps_mse={val_loss:.4f}  best={best_val:.4f}")

        # Make sure the in-memory weights are the best ones for the cascade eval.
        if best_ckpt.exists():
            self._load_stage(stage_idx, best_ckpt)

    @torch.no_grad()
    def _stage_validate(self, stage_idx, val_loader, max_batches=40):
        unet = self.unets[stage_idx]
        unet.eval()
        losses = []
        for bi, batch in enumerate(val_loader):
            if bi >= max_batches:
                break
            prepped = self._prep_batch(batch)
            loss = self._stage_train_step(stage_idx, prepped)
            if torch.isfinite(loss):
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float('nan')

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _ddim_timesteps(self, n_steps):
        n_steps = min(n_steps, self.num_timesteps)
        ts = torch.linspace(self.num_timesteps - 1, 0, n_steps).round().long().tolist()
        return ts

    @torch.no_grad()
    def _sample_stage(self, stage_idx, cond, res, n_steps):
        """DDIM sampling for one stage with classifier-free guidance."""
        unet = self.unets[stage_idx]
        unet.eval()
        B = cond.shape[0]
        x = torch.randn(B, self.target_channels, res, res, device=self.device)
        cond_u = self._drop_context(cond) if self.cond_scale != 1.0 else None
        ts = self._ddim_timesteps(n_steps)
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if (i + 1) < len(ts) else -1
            t_b = torch.full((B,), t, device=self.device, dtype=torch.long)
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                out_c = self._to_eps(unet(x, cond, t_b).float(), x, t)
                if cond_u is not None:
                    out_u = self._to_eps(unet(x, cond_u, t_b).float(), x, t)
                    eps = out_u + self.cond_scale * (out_c - out_u)
                else:
                    eps = out_c
            x = self.scheduler.sample_prev_timestep_ddim(
                x, eps.float(), t, t_prev=t_prev, eta=0.0, clamp_range=(-5.0, 5.0))
        return x

    @torch.no_grad()
    def sample_cascade(self, prepped, n_steps=None):
        """Run the full 64²→128²→256² cascade. Returns (B, K_out, 5, 256, 256)."""
        n_steps = n_steps or self.ddim_steps
        for ema, unet in zip(self.emas, self.unets):
            ema.apply(unet)
        try:
            lowres_pred = None
            for stage_idx, stage in enumerate(self.stages):
                cond = self._build_cond(stage, prepped['input_flat'], lowres_pred=lowres_pred)
                x = self._sample_stage(stage_idx, cond, stage['res'], n_steps)
                lowres_pred = x      # feeds the next stage
            final = x                # (B, K_out*5, 256, 256)
        finally:
            for ema, unet in zip(self.emas, self.unets):
                ema.restore(unet)
        B = final.shape[0]
        return final.reshape(B, self.k_out, self.n_ch, final.shape[-2], final.shape[-1])

    # ------------------------------------------------------------------
    # Per-leadtime evaluation (matches multi_frame_flow output format)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_base_only(self, prepped, n_steps=None):
        """Diagnostic: run ONLY the base 64² stage, then bilinearly upsample its
        output to 256². Isolates the base stage's quality from the SR cascade.
        Returns (B, K_out, 5, 256, 256)."""
        n_steps = n_steps or self.ddim_steps
        ema, unet = self.emas[0], self.unets[0]
        ema.apply(unet)
        try:
            cond = self._build_cond(self.stages[0], prepped['input_flat'], lowres_pred=None)
            x = self._sample_stage(0, cond, self.stages[0]['res'], n_steps)   # (B, K_out*5, 64, 64)
            x = self._resize(x, 256)
        finally:
            ema.restore(unet)
        B = x.shape[0]
        return x.reshape(B, self.k_out, self.n_ch, 256, 256)

    @torch.no_grad()
    def evaluate_per_leadtime(self, loader, num_samples=None, n_steps=None,
                              sample_fn=None, out_name='per_leadtime.csv'):
        self._capture_stats(loader)
        num_samples = num_samples or self.eval_samples
        n_steps = n_steps or self.ddim_steps
        sample_fn = sample_fn or self.sample_cascade
        ch_names = ['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']
        calc = MetricsCalculator()

        rows = []
        crps_buffer = {k + 1: [] for k in range(self.k_out)}
        n_seen = 0
        for batch in loader:
            if n_seen >= num_samples:
                break
            prepped = self._prep_batch(batch)
            gen_block = sample_fn(prepped, n_steps=n_steps)   # (B,K_out,5,256,256)
            tgt_block = prepped['target_block']

            for k in range(self.k_out):
                gen_k = gen_block[:, k]
                tgt_k = tgt_block[:, k]
                ch_rows = {}
                for ci, cn in enumerate(ch_names):
                    g = gen_k[:, ci:ci+1]; t = tgt_k[:, ci:ci+1]
                    ch_rows[f'{cn}_mse'] = float(F.mse_loss(g, t).item())
                    ch_rows[f'{cn}_mae'] = float(torch.mean(torch.abs(g - t)).item())
                    ch_rows[f'{cn}_psnr'] = calc.psnr(g, t)
                    ch_rows[f'{cn}_ssim'] = calc.ssim(g, t)
                ch_rows['overall_mse'] = calc.mse(gen_k, tgt_k)
                ch_rows['overall_mae'] = calc.mae(gen_k, tgt_k)
                ch_rows['overall_psnr'] = calc.psnr(gen_k, tgt_k)
                ch_rows['overall_ssim'] = calc.ssim(gen_k, tgt_k)
                ch_rows['gridsat_lpips'] = lpips_metric(gen_k[:, 0:1], tgt_k[:, 0:1])
                ch_rows['gridsat_rapsd_l1'] = rapsd_l1(gen_k[:, 0:1], tgt_k[:, 0:1])
                gs_pred_K = denormalise_gridsat(gen_k[:, 0:1], self.gs_mean, self.gs_std)
                gs_tgt_K = denormalise_gridsat(tgt_k[:, 0:1], self.gs_mean, self.gs_std)
                sk = threshold_skill_scores(gs_pred_K, gs_tgt_K, threshold=220.0, comparison='less')
                ch_rows['csi_220'] = sk['csi']; ch_rows['pod_220'] = sk['pod']; ch_rows['far_220'] = sk['far']
                ch_rows['mslp_err_hpa'] = mslp_err_hpa(gen_k, tgt_k, self.era5_mean, self.era5_std)
                ch_rows.update(max_wind_err(gen_k, tgt_k, self.era5_mean, self.era5_std))
                ch_rows['track_km'] = self._steering_track_km(
                    gen_k, tgt_k, prepped['cur_lat'], prepped['cur_lon'], lead_k=k + 1)
                rows.append({'lead': k + 1, **ch_rows})
            n_seen += gen_block.shape[0]

        crps_rows = self._crps_pass(loader, n_steps=n_steps,
                                    num_samples=min(num_samples, 40),
                                    sample_fn=sample_fn)
        df = pd.DataFrame(rows)
        out = df.groupby('lead').mean(numeric_only=True).reset_index()
        for k_idx in range(1, self.k_out + 1):
            out.loc[out['lead'] == k_idx, 'crps'] = crps_rows.get(k_idx, float('nan'))
        out.to_csv(self.metrics_dir / out_name, index=False)
        print(f"[Cascade] {out_name} written ({len(out)} leads, {n_seen} samples)")
        return out

    @torch.no_grad()
    def _crps_pass(self, loader, n_steps=50, num_samples=40, sample_fn=None):
        """CRPS over an ensemble of stochastic cascade rollouts."""
        sample_fn = sample_fn or self.sample_cascade
        calc_crps = {k + 1: [] for k in range(self.k_out)}
        n_seen = 0
        for batch in loader:
            if n_seen >= num_samples:
                break
            prepped = self._prep_batch(batch)
            ens = torch.stack([
                sample_fn(prepped, n_steps=n_steps)
                for _ in range(self.crps_n_samples)], dim=0)   # (N,B,K_out,5,H,W)
            tgt = prepped['target_block']
            for k in range(self.k_out):
                # Mean absolute error of ensemble mean as a CRPS proxy on the
                # GRIDSAT channel (matches the cheap CRPS used elsewhere).
                e = ens[:, :, k, 0]            # (N,B,H,W)
                t = tgt[:, k, 0]               # (B,H,W)
                mae_term = (e - t.unsqueeze(0)).abs().mean()
                spread = (e.unsqueeze(0) - e.unsqueeze(1)).abs().mean()
                calc_crps[k + 1].append(float((mae_term - 0.5 * spread).item()))
            n_seen += tgt.shape[0]
        return {k: (float(np.mean(v)) if v else float('nan')) for k, v in calc_crps.items()}

    def _steering_track_km(self, gen_5ch, tgt_5ch, cur_lat, cur_lon, lead_k=1, roi=64):
        B, C, H, W = gen_5ch.shape
        y0 = (H - roi) // 2; x0 = (W - roi) // 2
        u_m, u_s = float(self.era5_mean[0]), float(self.era5_std[0])
        v_m, v_s = float(self.era5_mean[1]), float(self.era5_std[1])
        gen_u = gen_5ch[:, 1, y0:y0+roi, x0:x0+roi].cpu().numpy() * u_s + u_m
        gen_v = gen_5ch[:, 2, y0:y0+roi, x0:x0+roi].cpu().numpy() * v_s + v_m
        tgt_u = tgt_5ch[:, 1, y0:y0+roi, x0:x0+roi].cpu().numpy() * u_s + u_m
        tgt_v = tgt_5ch[:, 2, y0:y0+roi, x0:x0+roi].cpu().numpy() * v_s + v_m
        dt_sec = lead_k * 3.0 * 3600.0
        lat_np = cur_lat.cpu().numpy(); lon_np = cur_lon.cpu().numpy()
        kms = []
        for b in range(B):
            cl = max(np.cos(np.radians(lat_np[b])), 0.1)
            dlat_gen = float(gen_v[b].mean()) * dt_sec / 111_000.0
            dlon_gen = float(gen_u[b].mean()) * dt_sec / (111_000.0 * cl)
            dlat_gt = float(tgt_v[b].mean()) * dt_sec / 111_000.0
            dlon_gt = float(tgt_u[b].mean()) * dt_sec / (111_000.0 * cl)
            kms.append(great_circle_km(
                lat_np[b] + dlat_gt, lon_np[b] + dlon_gt,
                lat_np[b] + dlat_gen, lon_np[b] + dlon_gen))
        return float(np.mean(kms)) if kms else 0.0

    @torch.no_grad()
    def run_sampler_step_sweep(self, loader, steps_list=None, num_samples=60):
        self._capture_stats(loader)
        steps_list = steps_list or self.sampler_step_sweep
        calc = MetricsCalculator()
        rows = []
        for n_steps in steps_list:
            psnrs, ssims, mses = [], [], []
            t0 = time.time(); count = 0
            for batch in loader:
                if count >= num_samples:
                    break
                prepped = self._prep_batch(batch)
                gen = self.sample_cascade(prepped, n_steps=n_steps)
                tgt = prepped['target_block']
                psnrs.append(calc.psnr(gen.flatten(0, 1), tgt.flatten(0, 1)))
                ssims.append(calc.ssim(gen.flatten(0, 1), tgt.flatten(0, 1)))
                mses.append(calc.mse(gen.flatten(0, 1), tgt.flatten(0, 1)))
                count += gen.shape[0]
            elapsed = time.time() - t0
            ms = (elapsed / max(count, 1)) * 1000
            rows.append({
                'n_steps': n_steps,
                'overall_psnr': float(np.mean(psnrs)) if psnrs else 0.0,
                'overall_ssim': float(np.mean(ssims)) if ssims else 0.0,
                'overall_mse': float(np.mean(mses)) if mses else 0.0,
                'wallclock_ms_per_sample': ms,
            })
            print(f"  [cascade sweep] {n_steps:>3} steps: PSNR={rows[-1]['overall_psnr']:.2f}, "
                  f"SSIM={rows[-1]['overall_ssim']:.4f}, {ms:.1f} ms/sample")
        pd.DataFrame(rows).to_csv(self.metrics_dir / 'sampler_steps.csv', index=False)
        return rows

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_stage(self, stage_idx, path, epoch, val_loss):
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'unet': self.unets[stage_idx].state_dict(),
            'ema': self.emas[stage_idx].state_dict(),
            'stage': self.stages[stage_idx]['name'],
        }, path)

    def _load_stage(self, stage_idx, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.unets[stage_idx].load_state_dict(ckpt['unet'])
        self.emas[stage_idx].load_state_dict(ckpt['ema'])

    def load_best_checkpoints(self):
        """Load every stage's `cascade_<name>_best.pt` for eval-only runs."""
        for stage_idx, stage in enumerate(self.stages):
            p = self.ckpt_dir / f"cascade_{stage['name']}{self.ckpt_suffix}_best.pt"
            if not p.exists():
                raise FileNotFoundError(
                    f"[Cascade] eval-only needs {p} — train the cascade first.")
            self._load_stage(stage_idx, p)
            print(f"  [load] {stage['name']} ← {p.name}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def eval_only(self, root_dir, train_years, val_years, test_years,
                  num_workers=8, coord_lookup_path=None, num_samples=None):
        """Diagnostic eval on the existing best checkpoints — no training.

        Produces three CSVs in the metrics dir so the failure can be localised:
          per_leadtime_base64only.csv  — base 64² stage alone (upsampled to 256²)
          per_leadtime_cs{scale}.csv   — full cascade at the current cond_scale
          sampler_steps.csv            — step/quality/latency sweep (full cascade)

        Compare base64only vs the full cascade: if base-only is decent but the
        cascade is poor, the super-resolution stages are the problem (missing
        low-res noise augmentation). Re-run with --cond-scale 1.0 to test CFG.
        """
        self.load_best_checkpoints()
        _, val_loader, _ = get_multi_frame_dataloaders(
            root_dir=root_dir, train_years=train_years,
            val_years=val_years, test_years=test_years,
            k_in=self.k_in, k_out=self.k_out,
            batch_size=self._bs_for(2), num_workers=num_workers,
            img_size=256, coord_lookup_path=coord_lookup_path,
        )
        num_samples = num_samples or self.eval_samples

        print(f"\n[Cascade eval-only] base-64 stage ONLY ({self.ddim_steps} DDIM steps)")
        base = self.evaluate_per_leadtime(
            val_loader, num_samples=num_samples, sample_fn=self._sample_base_only,
            out_name='per_leadtime_base64only.csv')

        cs_tag = f"cs{self.cond_scale:g}".replace('.', 'p')
        print(f"\n[Cascade eval-only] full cascade (cond_scale={self.cond_scale}, "
              f"{self.ddim_steps} DDIM steps)")
        full = self.evaluate_per_leadtime(
            val_loader, num_samples=num_samples, sample_fn=self.sample_cascade,
            out_name=f'per_leadtime_{cs_tag}.csv')

        # Side-by-side headline for the lead-1 row.
        def _row1(df):
            r = df[df['lead'] == 1].iloc[0]
            return r['overall_psnr'], r['overall_ssim'], r['track_km']
        bp, bs, bt = _row1(base)
        fp, fs, ft = _row1(full)
        print(f"\n{'='*60}\n[Cascade eval-only] lead-1 comparison\n{'='*60}")
        print(f"  {'variant':<22}{'PSNR':>8}{'SSIM':>9}{'track_km':>11}")
        print(f"  {'base-64 only':<22}{bp:>8.2f}{bs:>9.3f}{bt:>11.1f}")
        print(f"  {'full cascade cs='+format(self.cond_scale,'g'):<22}{fp:>8.2f}{fs:>9.3f}{ft:>11.1f}")
        print(f"{'='*60}")
        if bp - fp > 2.0:
            print("  → base stage is markedly better than the cascade: the SR stages "
                  "are degrading it (missing low-res noise augmentation).")
        elif bp < 10.0:
            print("  → base stage itself is weak: the problem is upstream of the SR "
                  "cascade (base capacity / CFG / sampler).")
        return base, full

    def train(self, root_dir, train_years, val_years, test_years,
              num_workers=8, coord_lookup_path=None):
        """Train all three stages, each with its own resolution-tuned batch size,
        then evaluate the full cascade and write the paper CSVs."""
        for stage_idx, stage in enumerate(self.stages):
            bs = self._bs_for(stage_idx)
            train_loader, val_loader, _ = get_multi_frame_dataloaders(
                root_dir=root_dir, train_years=train_years,
                val_years=val_years, test_years=test_years,
                k_in=self.k_in, k_out=self.k_out,
                batch_size=bs, num_workers=num_workers,
                img_size=256, coord_lookup_path=coord_lookup_path,
            )
            self._capture_stats(train_loader)
            self.train_stage(stage_idx, train_loader, val_loader)

        # Final end-to-end evaluation on the val split (256², full cascade).
        _, val_loader, _ = get_multi_frame_dataloaders(
            root_dir=root_dir, train_years=train_years,
            val_years=val_years, test_years=test_years,
            k_in=self.k_in, k_out=self.k_out,
            batch_size=self._bs_for(2), num_workers=num_workers,
            img_size=256, coord_lookup_path=coord_lookup_path,
        )
        print(f"\n[Cascade] final end-to-end evaluation ({self.ddim_steps} DDIM steps)")
        self.evaluate_per_leadtime(val_loader)
        self.run_sampler_step_sweep(val_loader)
        print(f"[Cascade] done → {self.metrics_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True, help='SETCD root_dir')
    p.add_argument('--train-years', nargs='+',
                   default=['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'])
    p.add_argument('--val-years', nargs='+', default=['2021'])
    p.add_argument('--test-years', nargs='+', default=['2022'])
    p.add_argument('--coord-lookup-path', default=None)
    p.add_argument('--k-in', type=int, default=3)
    p.add_argument('--k-out', type=int, default=3)
    p.add_argument('--run-tag', default='cascaded_diffusion_R4')
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--num-timesteps', type=int, default=250)
    p.add_argument('--prediction-type', choices=['v', 'eps'], default='v',
                   help="Diffusion parameterisation. 'v' (default) is far more "
                        "sample-robust than 'eps' for this conditional forecasting task.")
    p.add_argument('--cond-scale', type=float, default=3.0)
    p.add_argument('--epochs', type=int, nargs='+', default=[60, 50, 40],
                   help='Per-stage epochs (base64, sr128, sr256).')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[32, 16, 6],
                   help='Per-stage batch sizes.')
    p.add_argument('--eval-samples', type=int, default=200)
    p.add_argument('--eval-sampler-steps', type=int, default=50)
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training; load best checkpoints and run the '
                        'base-only vs full-cascade diagnostic eval.')
    args = p.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    cfg = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_amp': True, 'amp_dtype': 'bfloat16',
        'k_in': args.k_in, 'k_out': args.k_out,
        'run_tag': args.run_tag,
        'checkpoint_dir': args.checkpoint_dir,
        'output_dir': args.output_dir,
        'num_timesteps': args.num_timesteps,
        'prediction_type': args.prediction_type,
        'cond_scale': args.cond_scale,
        'cascade_epochs': args.epochs,
        'cascade_batch_sizes': args.batch_sizes,
        'eval_samples': args.eval_samples,
        'eval_sampler_steps': args.eval_sampler_steps,
    }
    trainer = CascadedDiffusionTrainer(cfg)
    if args.eval_only:
        trainer.eval_only(
            root_dir=args.root, train_years=args.train_years,
            val_years=args.val_years, test_years=args.test_years,
            num_workers=args.num_workers, coord_lookup_path=args.coord_lookup_path,
        )
    else:
        trainer.train(
            root_dir=args.root, train_years=args.train_years,
            val_years=args.val_years, test_years=args.test_years,
            num_workers=args.num_workers, coord_lookup_path=args.coord_lookup_path,
        )


if __name__ == '__main__':
    main()
