"""
Phase-2 headline experiment: multi-frame rectified-flow forecasting.

Architecture: V1 channel-stacked 2-D UNet (FlowMatchingUNet with widened
in/out channels). Conditioning: per-input-frame `ConditionEncoder` outputs
*summed* before the time embedding; per-frame coords stacked and projected.

Default: K_in = 3, K_out = 3 at 3-hour cadence (9-hour horizon).

Training: 2-Rectified Flow (initial flow + one reflow pass), optimised via
straight-line interpolation (`x_t = (1-t)*x0 + t*x1`, `v = x1 - x0`).
Physics losses (continuity / geostrophic / track) are flag-driven, default OFF.

Evaluation: per-leadtime PSNR/SSIM/LPIPS/MSE per channel + overall, CRPS over
N=10 stochastic samples, RAPSD L1, steering-flow displacement, track km error.
Sampler-step sweep at the final epoch.

This script is the **Phase-2 main runner**. See docs/phase2.md for the
ablation matrix. The defaults below produce row "Ours — multi-frame 2-RF
without physics loss" (R6 of phase3.md Table 1).
"""

from __future__ import annotations

import copy
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

from latent_diffusion import (
    EMA, ConditionEncoder,
    DownBlock, MidBlock, UpBlock,
)
from vae import MultiChannelVAE
from conditional_flow_matching import FlowMatchingUNet, ODESampler
from multi_frame_dataset import MultiFrameCycloneDataset, get_multi_frame_dataloaders
from physics_losses import PhysicsLossDriver
from eval_metrics_extended import (
    MetricsCalculator,
    lpips_metric, crps_ensemble, threshold_skill_scores, rapsd_l1,
    mslp_err_hpa, max_wind_err, great_circle_km, denormalise_gridsat,
)
from epoch_logger import EpochLogger


# ============================================================================
# V1 BACKBONE — channel-stacked multi-frame FlowMatchingUNet
# ============================================================================

class MultiFrameFlowUNet(nn.Module):
    """Wraps FlowMatchingUNet for multi-frame I/O via channel-stacking.

    In : x_t        (B, K_out * latent_dim, h_lat, w_lat)
         gridsat_in (B, K_in, 1, H, W)
         era5_in    (B, K_in, 4, H, W)
         t_cont     (B,) in [0,1]
         coords_in  (B, K_in, 2) per-frame normalised lat/lon

    Out: velocity   (B, K_out * latent_dim, h_lat, w_lat)
    """

    def __init__(self, latent_dim=4, k_in=3, k_out=3,
                 base_channels=128, channel_mults=(1, 2, 3, 4),
                 t_emb_dim=128, num_heads=8,
                 gridsat_channels=1, era5_channels=4,
                 img_size_latent=64, condition_embed_dim=128,
                 condition_img_size=256):
        super().__init__()
        self.k_in  = k_in
        self.k_out = k_out
        self.latent_dim = latent_dim

        # Underlying flow UNet that operates on channel-stacked latents
        self.unet = FlowMatchingUNet(
            in_channels=k_out * latent_dim,
            out_channels=k_out * latent_dim,
            base_channels=base_channels,
            channel_mults=tuple(channel_mults),
            t_emb_dim=t_emb_dim,
            num_heads=num_heads,
            gridsat_channels=gridsat_channels,
            era5_channels=era5_channels,
            img_size=img_size_latent,
            condition_embed_dim=condition_embed_dim,
            condition_img_size=condition_img_size,
        )
        # Replace the underlying coord proj to take K_in * 2 inputs.
        emb_dim = condition_embed_dim
        self.unet.condition_encoder.coord_proj = nn.Sequential(
            nn.Linear(k_in * 2, emb_dim // 2), nn.SiLU(),
            nn.Linear(emb_dim // 2, emb_dim),
        )

        # Learned per-frame slot embedding added into the time embedding.
        # Lets the UNet distinguish output slots in the channel-stack.
        self.frame_slot_emb = nn.Embedding(k_out, t_emb_dim * 4)

    def encode_conditioning(self, gridsat_in, era5_in, coords_in_flat,
                            timestamps_repr):
        """Mean-aggregates per-frame conditioning over the K_in axis."""
        B = gridsat_in.shape[0]
        # gridsat_in: (B, K_in, 1, H, W) — flatten K_in across batch, run encoder, then mean
        gs = gridsat_in.view(B * self.k_in, *gridsat_in.shape[2:])
        e5 = era5_in.view(B * self.k_in, *era5_in.shape[2:])
        ts = []  # ConditionEncoder expects a list of strings of len B*K_in
        for tlist in timestamps_repr:
            ts.extend(tlist)
        # Coords come in already flattened to (B, K_in * 2) — the coord_proj inside
        # condition_encoder uses that shape. But we still need a per-frame call for
        # the spatial conditioning, so we feed a dummy coords=None there and add
        # the coord embedding to the *aggregated* context vector below.
        encoded_image_feats, context_emb_per_frame = self.unet.condition_encoder(
            gs, e5, ts, coords=None)
        # Both have batch dim B*K_in. Aggregate by mean over K_in.
        encoded_image_feats = encoded_image_feats.view(
            B, self.k_in, *encoded_image_feats.shape[1:]).mean(dim=1)
        context_emb = context_emb_per_frame.view(
            B, self.k_in, -1).mean(dim=1)
        # Add the K_in-flattened coord embedding
        coord_emb = self.unet.condition_encoder.coord_proj(coords_in_flat)
        # context_projection inside ConditionEncoder operates on (B, embed_dim) —
        # but we already took the post-projection mean. Add coord emb into context.
        # context_emb is in (B, embed_dim*2) space (post context_projection).
        # We project coord_emb (B, embed_dim) up to that space using a copy of the
        # first linear in context_projection — simplest: tile-and-add.
        coord_emb_up = torch.cat([coord_emb, coord_emb], dim=-1)
        context_emb = context_emb + coord_emb_up
        return encoded_image_feats, context_emb

    def forward(self, x_t, gridsat_in, era5_in, timestamps_in, t_continuous,
                coords_in_flat,
                encoded_condition=None, context_emb=None):
        if encoded_condition is None or context_emb is None:
            encoded_condition, context_emb = self.encode_conditioning(
                gridsat_in, era5_in, coords_in_flat, timestamps_in)

        # Reuse the underlying UNet's forward by passing the already-encoded
        # conditioning. The UNet only needs *some* gridsat/era5/timestamps
        # placeholders for its `condition_encoder` skip path (which we bypass).
        # We pass the *first* frame as a dummy — it is unused because
        # encoded_condition/context_emb are provided.
        gs_dummy = gridsat_in[:, 0]                   # (B, 1, H, W)
        e5_dummy = era5_in[:, 0]                       # (B, 4, H, W)
        ts_dummy = [t[0] for t in timestamps_in]      # list[str] of len B

        # Add the per-output-frame slot embedding into the time embedding by
        # broadcasting: we call the underlying UNet K_out times? No — channel
        # stack lets us do this in one pass. The slot signal is added by
        # adjusting context_emb with a learned bias per output channel-slot.
        # We average slot embeddings here (so the model sees a single slot
        # bias), then leave the spatial UNet to assign channels.
        slot_idx = torch.arange(self.k_out, device=x_t.device)
        slot_bias = self.frame_slot_emb(slot_idx).mean(dim=0)   # (t_emb_dim*4,)

        # Use FlowMatchingUNet's existing pipeline by injecting via t_emb.
        # We call the inner UNet pipeline manually to avoid recomputing
        # condition_encoder.
        t_emb = self.unet._continuous_time_embedding(t_continuous)
        t_emb = self.unet.t_proj(t_emb)
        ctx_emb = self.unet.context_proj(context_emb)
        comb = t_emb + ctx_emb + slot_bias.unsqueeze(0)

        x = torch.cat([x_t, encoded_condition], dim=1)
        x = self.unet.init_conv(x)
        skips = []
        for enc in self.unet.encoders:
            x, skip = enc(x, comb)
            skips.append(skip)
        x = self.unet.mid_block(x, comb)
        for i, dec in enumerate(self.unet.decoders):
            x = dec(x, comb, skips[len(skips) - 1 - i])
        return self.unet.final_conv(x)


# ============================================================================
# V2 BACKBONE — factorised spatio-temporal (Ablation B)
# ============================================================================

class TemporalAttention(nn.Module):
    """Self-attention across the K_out frame (time) axis at each spatial
    location. This is the mechanism that lets later output frames (t+2, t+3)
    draw on the representations of earlier ones (t+1) — the thing V1's
    channel-stack cannot do explicitly."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x, k_out):
        # x: (B*k_out, C, h, w) — attend across the k_out axis per (h, w).
        BK, C, h, w = x.shape
        B = BK // k_out
        xn = self.norm(x).view(B, k_out, C, h, w)
        xn = xn.permute(0, 3, 4, 1, 2).reshape(B * h * w, k_out, C)   # (B*h*w, T, C)
        out, _ = self.attn(xn, xn, xn)
        out = out.reshape(B, h, w, k_out, C).permute(0, 3, 4, 1, 2).reshape(BK, C, h, w)
        return x + out


class MultiFrameFlowUNetV2(nn.Module):
    """Factorised spatio-temporal variant (Phase-2 Ablation B).

    Where V1 channel-stacks the K_out output frames and processes them with a
    single 2-D UNet (frames mixed, slot embedding averaged away), V2:

      * processes each output frame through a **shared** 2-D UNet (frames kept
        separable, folded into the batch axis), and
      * mixes information **across** the K_out frames with 1-D temporal
        self-attention at the bottleneck, and
      * adds a **per-frame** slot embedding so each lead-time is explicit.

    This directly targets the steep quality/track degradation at the longer
    leads (t+2 = 6 h, t+3 = 9 h). Same I/O contract as MultiFrameFlowUNet, so
    the trainer is unchanged.
    """

    def __init__(self, latent_dim=4, k_in=3, k_out=3,
                 base_channels=128, channel_mults=(1, 2, 3, 4),
                 t_emb_dim=128, num_heads=8,
                 gridsat_channels=1, era5_channels=4,
                 img_size_latent=64, condition_embed_dim=128,
                 condition_img_size=256):
        super().__init__()
        self.k_in, self.k_out, self.latent_dim = k_in, k_out, latent_dim

        # Per-frame UNet: single-frame in/out channels (vs V1's k_out*latent_dim).
        self.unet = FlowMatchingUNet(
            in_channels=latent_dim, out_channels=latent_dim,
            base_channels=base_channels, channel_mults=tuple(channel_mults),
            t_emb_dim=t_emb_dim, num_heads=num_heads,
            gridsat_channels=gridsat_channels, era5_channels=era5_channels,
            img_size=img_size_latent, condition_embed_dim=condition_embed_dim,
            condition_img_size=condition_img_size,
        )
        emb_dim = condition_embed_dim
        self.unet.condition_encoder.coord_proj = nn.Sequential(
            nn.Linear(k_in * 2, emb_dim // 2), nn.SiLU(),
            nn.Linear(emb_dim // 2, emb_dim),
        )
        # Per-output-frame slot embedding (each lead distinct, NOT averaged).
        self.frame_slot_emb = nn.Embedding(k_out, t_emb_dim * 4)
        # Temporal attention at the bottleneck (deepest channel count).
        bottleneck_ch = base_channels * tuple(channel_mults)[-1]
        self.temporal_attn = TemporalAttention(bottleneck_ch, num_heads=num_heads)

    # Conditioning aggregation is identical to V1 (mean over the K_in inputs).
    encode_conditioning = MultiFrameFlowUNet.encode_conditioning

    def forward(self, x_t, gridsat_in, era5_in, timestamps_in, t_continuous,
                coords_in_flat, encoded_condition=None, context_emb=None):
        if encoded_condition is None or context_emb is None:
            encoded_condition, context_emb = self.encode_conditioning(
                gridsat_in, era5_in, coords_in_flat, timestamps_in)
        B = x_t.shape[0]
        K, ld = self.k_out, self.latent_dim
        sp = x_t.shape[2:]   # (h, w)

        # Time + context embedding, then per-frame slot bias → (B*K, emb_dim).
        t_emb = self.unet._continuous_time_embedding(t_continuous)
        t_emb = self.unet.t_proj(t_emb)
        ctx_emb = self.unet.context_proj(context_emb)
        comb = t_emb + ctx_emb                                   # (B, emb_dim)
        slot = self.frame_slot_emb(torch.arange(K, device=x_t.device))  # (K, emb_dim)
        comb = (comb.unsqueeze(1) + slot.unsqueeze(0)).reshape(B * K, -1)

        # Fold the K_out frames into the batch axis; broadcast conditioning.
        xf = x_t.view(B, K, ld, *sp).reshape(B * K, ld, *sp)
        cond = encoded_condition.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(
            B * K, *encoded_condition.shape[1:])

        x = torch.cat([xf, cond], dim=1)
        x = self.unet.init_conv(x)
        skips = []
        for enc in self.unet.encoders:
            x, skip = enc(x, comb)
            skips.append(skip)
        x = self.unet.mid_block(x, comb)
        x = self.temporal_attn(x, K)                     # << cross-lead mixing
        for i, dec in enumerate(self.unet.decoders):
            x = dec(x, comb, skips[len(skips) - 1 - i])
        v = self.unet.final_conv(x)                      # (B*K, ld, h, w)
        return v.view(B, K, ld, *v.shape[2:]).reshape(B, K * ld, *v.shape[2:])


# ============================================================================
# MULTI-FRAME RECTIFIED-FLOW TRAINER
# ============================================================================

class MultiFrameRFTrainer:
    """Two-phase training: initial 1-RF, then one reflow pass producing 2-RF.

    Operates in the latent space of a frozen MC-VAE. The K_out output frames
    are channel-stacked into a single latent of shape (B, K_out*4, 64, 64).
    """

    def __init__(self, config, mc_vae_checkpoint_path, resume_path=None):
        self.cfg = config
        self.device = torch.device(config['device'])
        self.k_in  = int(config.get('k_in',  3))
        self.k_out = int(config.get('k_out', 3))
        self.latent_dim = int(config.get('latent_dim', 4))

        # Output dirs
        run_tag = config.get('run_tag', f"mf_kin{self.k_in}_kout{self.k_out}")
        self.checkpoint_dir = Path(config['checkpoint_dir']) / run_tag
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config['output_dir']) / run_tag
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # --- VAE ---
        print(f"[MF-RF] Loading MC-VAE from {mc_vae_checkpoint_path}")
        ckpt = torch.load(mc_vae_checkpoint_path,
                          map_location=self.device, weights_only=False)
        self.vae = MultiChannelVAE(
            in_channels=5, latent_dim=self.latent_dim,
            base_channels=config.get('vae_base_channels', 64),
        ).to(self.device)
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.latent_scale_factor = ckpt.get('latent_scale_factor', None)
        self.latent_clamp_range  = (-5.0, 5.0)

        # --- UNet ---  (V1 channel-stack vs V2 factorised spatio-temporal)
        img_size = config['img_size']
        latent_size = img_size // 4
        self.variant = str(config.get('variant', 'v1')).lower()
        unet_cls = MultiFrameFlowUNetV2 if self.variant == 'v2' else MultiFrameFlowUNet
        self.unet = unet_cls(
            latent_dim=self.latent_dim,
            k_in=self.k_in, k_out=self.k_out,
            base_channels=config['base_channels'],
            channel_mults=tuple(config['channel_mults']),
            t_emb_dim=config['t_emb_dim'],
            num_heads=config['num_heads'],
            gridsat_channels=1, era5_channels=4,
            img_size_latent=latent_size,
            condition_embed_dim=config['embed_dim'],
            condition_img_size=img_size,
        ).to(self.device)

        # condition_downsample (matches single-frame trainers)
        self.condition_downsample = nn.Sequential(
            nn.Conv2d(config['base_channels'], config['base_channels'], 3, 2, 1),
            nn.GroupNorm(8, config['base_channels']), nn.SiLU(),
            nn.Conv2d(config['base_channels'], config['base_channels'], 3, 2, 1),
            nn.GroupNorm(8, config['base_channels']), nn.SiLU(),
        ).to(self.device)

        self.optimizer = AdamW(
            list(self.unet.parameters()) + list(self.condition_downsample.parameters()),
            lr=config['learning_rate'], weight_decay=config['weight_decay'],
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.get('diffusion_epochs', 100),
            eta_min=config.get('min_lr', 5e-5),
        )
        self.ema = EMA(self.unet, decay=config.get('ema_decay', 0.9999))
        self.use_amp = config.get('use_amp', True)
        self.scaler = (torch.amp.GradScaler('cuda', init_scale=1024, growth_interval=2000)
                       if self.use_amp else None)
        self.grad_accum = int(config.get('gradient_accumulation_steps', 1))
        self.warmup_epochs = int(config.get('warmup_epochs', 5))
        self.base_lr = float(config['learning_rate'])
        self.reflow_iterations = int(config.get('reflow_iterations', 1))
        self.reflow_epochs = int(config.get('reflow_epochs', 40))
        self.reflow_sample_steps = int(config.get('reflow_sample_steps', 20))
        self.sampler_step_sweep = list(config.get('sampler_step_sweep', [1, 4, 16, 50]))
        self.crps_n_samples = int(config.get('crps_n_samples', 10))

        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.train_losses, self.val_losses = [], []
        self.metrics_history = []

        # Rich per-epoch metric logger (CSV + .log)
        self.logger = EpochLogger(self.metrics_dir, run_tag=f'mf_{run_tag}')

        # Physics losses
        ds_means = MultiFrameCycloneDataset.ERA5_MEAN.flatten().tolist()
        ds_stds  = MultiFrameCycloneDataset.ERA5_STD.flatten().tolist()
        self.physics = PhysicsLossDriver(config, ds_means, ds_stds)

        if resume_path:
            self._load_checkpoint(resume_path)

        n_params = sum(p.numel() for p in self.unet.parameters())
        print(f"[MF-RF] UNet params: {n_params:,}, "
              f"K_in={self.k_in}, K_out={self.k_out}, variant={self.variant}, "
              f"physics={'ON' if self.physics.enabled else 'OFF'}")

    # --------------------------------------------------------------
    # Latent helpers
    # --------------------------------------------------------------

    @torch.no_grad()
    def _encode_target_block(self, target_5ch_block=None, cached_latents=None):
        """Channel-stack K_out target latents, scaled by latent_scale_factor.

        Two paths:
          - If `cached_latents` is provided (B, K_out, latent_dim, h, w) from
            disk cache, skip the VAE entirely — just rescale and reshape.
          - Otherwise encode `target_5ch_block` (B, K_out, 5, H, W) through
            the VAE.

        The cache stores raw mu (un-scaled); both paths apply
        `self.latent_scale_factor` here.
        """
        if cached_latents is not None:
            z = cached_latents.to(self.device, non_blocking=True)
            z = z * self.latent_scale_factor
            B, K, latent_dim, h, w = z.shape
            return z.reshape(B, K * latent_dim, h, w)

        # Fallback: live VAE encoding
        B, K, C, H, W = target_5ch_block.shape
        flat = target_5ch_block.view(B * K, C, H, W)
        mu, _ = self.vae.encode(flat)
        z = mu * self.latent_scale_factor
        latent_dim, h, w = z.shape[1], z.shape[2], z.shape[3]
        return z.view(B, K, latent_dim, h, w).reshape(B, K * latent_dim, h, w)

    @torch.no_grad()
    def _decode_target_block(self, z_stacked):
        """z_stacked: (B, K_out * latent_dim, h, w) → (B, K_out, 5, H, W)."""
        B, total_C, h, w = z_stacked.shape
        latent_dim = self.latent_dim
        K = total_C // latent_dim
        z = z_stacked.view(B, K, latent_dim, h, w).reshape(B * K, latent_dim, h, w)
        x5 = self.vae.decode(z / self.latent_scale_factor)
        x5 = x5.view(B, K, 5, x5.shape[2], x5.shape[3])
        return x5

    @torch.no_grad()
    def compute_latent_scale_factor(self, loader, num_batches=50):
        """Pre-training: estimate 1/std(mu) for normalising the latent space.
        Uses cached mu when available (no VAE forward); falls back to
        live encoding otherwise."""
        all_z = []
        n_cached, n_encoded = 0, 0
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            cached = batch.get('target_latents')
            if cached is not None:
                # cached mu is un-scaled — exactly what we want for std estimation.
                # shape (B, K_out, latent_dim, h, w) → flatten batch+K dims
                B, K = cached.shape[:2]
                all_z.append(cached.reshape(B * K, *cached.shape[2:]).cpu())
                n_cached += B * K
            else:
                t5 = torch.cat([batch['target_gs'], batch['target_era5']], dim=2)
                t5 = t5.to(self.device)
                B, K, C, H, W = t5.shape
                flat = t5.view(B * K, C, H, W)
                mu, _ = self.vae.encode(flat)
                all_z.append(mu.cpu())
                n_encoded += B * K
        z_all = torch.cat(all_z, 0)
        z_std = float(z_all.std().item())
        self.latent_scale_factor = 1.0 / max(z_std, 1e-6)
        src = (f"{n_cached} cached"
               + (f" + {n_encoded} live-encoded" if n_encoded else ""))
        print(f"  ✓ latent_scale_factor = {self.latent_scale_factor:.5f} "
              f"(z_std={z_std:.4f}, source: {src})")

    @torch.no_grad()
    def compute_latent_clamp_range(self, loader, num_batches=50):
        """Pre-training: compute the empirical range of scaled latents,
        used to clamp ODE-integrated outputs at sampling time."""
        mins, maxs = [], []
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            cached = batch.get('target_latents')
            if cached is not None:
                z = self._encode_target_block(cached_latents=cached.to(self.device))
            else:
                t5 = torch.cat([batch['target_gs'], batch['target_era5']], dim=2).to(self.device)
                z = self._encode_target_block(target_5ch_block=t5)
            mins.append(float(z.min().item()))
            maxs.append(float(z.max().item()))
        zmin, zmax = min(mins), max(maxs)
        margin = (zmax - zmin) * 0.1
        self.latent_clamp_range = (zmin - margin, zmax + margin)
        print(f"  ✓ latent_clamp_range = [{self.latent_clamp_range[0]:.2f}, "
              f"{self.latent_clamp_range[1]:.2f}]")

    # --------------------------------------------------------------
    # Forward helpers
    # --------------------------------------------------------------

    def _prep_batch(self, batch):
        gridsat_in = batch['gridsat_in'].to(self.device)              # (B, K_in, 1, H, W)
        era5_in    = batch['era5_in'].to(self.device)                  # (B, K_in, 4, H, W)
        target_gs  = batch['target_gs'].to(self.device)                # (B, K_out, 1, H, W)
        target_era5 = batch['target_era5'].to(self.device)             # (B, K_out, 4, H, W)
        target_5ch_block = torch.cat([target_gs, target_era5], dim=2)  # (B, K_out, 5, H, W)
        coords_in = batch['coords_in'].to(self.device)                  # (B, K_in, 2)
        coords_in_flat = coords_in.view(coords_in.shape[0], -1)         # (B, K_in*2)
        timestamps_in = list(zip(*[batch['timestamps_in'][k] for k in range(self.k_in)])) \
            if isinstance(batch['timestamps_in'], list) else batch['timestamps_in']
        # collate quirk: torch returns list-of-lists. Convert to list of per-sample tuples.
        # Each entry below should be a list[str] of length k_in for that sample.
        if isinstance(timestamps_in, list) and len(timestamps_in) > 0 \
           and not isinstance(timestamps_in[0], (list, tuple)):
            # already correct
            pass
        # Capture pre-encoded target latents if the dataset's latent cache fired.
        # The trainer can skip the per-batch VAE forward when this is present.
        target_latents = batch.get('target_latents')
        if target_latents is not None:
            target_latents = target_latents.to(self.device, non_blocking=True)

        return {
            'gridsat_in':       gridsat_in,
            'era5_in':          era5_in,
            'target_5ch_block': target_5ch_block,
            'target_latents':   target_latents,   # None if no cache hit
            'coords_in_flat':   coords_in_flat,
            'timestamps_in':    timestamps_in,
            'cur_lat':          batch['raw_lat_in'][:, -1].to(self.device).float(),  # latest input frame
            'cur_lon':          batch['raw_lon_in'][:, -1].to(self.device).float(),
            'target_lat':       batch['raw_target_lat'].to(self.device).float(),     # (B, K_out)
            'target_lon':       batch['raw_target_lon'].to(self.device).float(),
        }

    def _model_forward(self, x_t, t, prepped, enc_cond=None, ctx=None):
        """If enc_cond/ctx are None, compute them here (via encode_conditioning
        + condition_downsample) so callers like validate() don't have to.

        The bare UNet computes encoded_condition at full image resolution
        (256x256); concatenating that with the latent x_t (64x64) blows up.
        We always downsample to latent resolution before passing in.
        """
        if enc_cond is None or ctx is None:
            enc_cond_full, ctx_new = self.unet.encode_conditioning(
                prepped['gridsat_in'], prepped['era5_in'],
                prepped['coords_in_flat'], prepped['timestamps_in'])
            enc_cond = self.condition_downsample(enc_cond_full)
            if ctx is None:
                ctx = ctx_new
        return self.unet(
            x_t,
            prepped['gridsat_in'], prepped['era5_in'],
            prepped['timestamps_in'], t,
            coords_in_flat=prepped['coords_in_flat'],
            encoded_condition=enc_cond, context_emb=ctx,
        )

    # --------------------------------------------------------------
    # Train / val
    # --------------------------------------------------------------

    def _train_step(self, prepped, use_reflow_pairs=False, epoch=0):
        # Use cached latents when available (skips VAE encode).
        x1_gt = self._encode_target_block(
            target_5ch_block=prepped['target_5ch_block'],
            cached_latents=prepped.get('target_latents'),
        )
        bs = x1_gt.shape[0]
        x0 = torch.randn_like(x1_gt)

        # Conditioning (once per batch)
        enc_cond_full, ctx = self.unet.encode_conditioning(
            prepped['gridsat_in'], prepped['era5_in'],
            prepped['coords_in_flat'], prepped['timestamps_in'])
        enc_cond = self.condition_downsample(enc_cond_full)

        if use_reflow_pairs:
            with torch.no_grad():
                # Build x1 by integrating from x0 with current weights, using
                # the SAME conditioning as the loss step (the reflow bug fix).
                x = x0.clone()
                num_steps = self.reflow_sample_steps
                dt = 1.0 / num_steps
                for i in range(num_steps):
                    tt = torch.full((bs,), i * dt, device=self.device)
                    v = self._model_forward(x, tt, prepped, enc_cond=enc_cond, ctx=ctx)
                    x = x + v * dt
                x1 = x
        else:
            x1 = x1_gt

        t = torch.rand(bs, device=self.device)
        t_sp = t[:, None, None, None]
        x_t = (1 - t_sp) * x0 + t_sp * x1
        v_target = x1 - x0

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            v_pred = self._model_forward(x_t, t, prepped, enc_cond=enc_cond, ctx=ctx)
            loss_v = F.mse_loss(v_pred, v_target)
            loss = loss_v

            # Physics: predict the implied x1 by extrapolation, decode, evaluate.
            # Cheap: predicted x1_hat = x_t + (1 - t_sp) * v_pred is *not* the
            # endpoint; the simpler clean signal is to just use x1_gt's decoded
            # form (when use_reflow_pairs=False). For physics we need the
            # *predicted* trajectory endpoint to encourage physically-plausible
            # outputs. We use one-Euler-step rollout from x_t to t=1 as a cheap
            # estimate of x1_hat.
            if self.physics.enabled:
                dt_to_1 = (1.0 - t_sp)
                x1_hat = x_t + v_pred * dt_to_1
                x1_hat = torch.clamp(x1_hat,
                                     self.latent_clamp_range[0],
                                     self.latent_clamp_range[1])
                # Physics must run in fp32: the pressure denormalisation adds
                # ~101325 Pa, which overflows fp16 (max 65504) → inf → NaN.
                with torch.amp.autocast('cuda', enabled=False):
                    pred_5ch_block = self._decode_target_block(x1_hat.float())
                    phys = self.physics.compute(
                        pred_5ch=pred_5ch_block,
                        cur_lat=prepped['cur_lat'].float(),
                        cur_lon=prepped['cur_lon'].float(),
                        target_lat=prepped['target_lat'].float(),
                        target_lon=prepped['target_lon'].float(),
                        epoch=epoch,
                    )
                loss = loss + phys['total']

        return loss, loss_v

    def train_epoch(self, loader, epoch, use_reflow=False):
        self.unet.train(); self.condition_downsample.train(); self.vae.eval()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc=f"{'Reflow' if use_reflow else 'MF-RF'} Ep {epoch}")
        for bi, batch in enumerate(pbar):
            if bi % self.grad_accum == 0:
                self.optimizer.zero_grad()
            prepped = self._prep_batch(batch)
            loss, loss_v = self._train_step(prepped, use_reflow_pairs=use_reflow, epoch=epoch)
            loss = loss / self.grad_accum
            if not torch.isfinite(loss):
                print(f"  ⚠ NaN/Inf loss at batch {bi}, skipping")
                self.logger.log_nan_skip()
                self.optimizer.zero_grad()
                # Do NOT call scaler.update() here — scaler.scale() hasn't run
                # for this batch, so there are no inf checks to consume.
                # Calling update() would assert "No inf checks were recorded".
                continue
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            last_grad_norm = None
            if (bi + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    g = torch.nn.utils.clip_grad_norm_(
                        list(self.unet.parameters())
                        + list(self.condition_downsample.parameters()), 1.0)
                    if not torch.isfinite(g):
                        print(f"  ⚠ NaN grad at batch {bi}, skip step")
                        self.logger.log_nan_skip()
                        self.optimizer.zero_grad(); self.scaler.update()
                    else:
                        self.scaler.step(self.optimizer); self.scaler.update()
                        self.ema.update(self.unet)
                        last_grad_norm = float(g.item())
                else:
                    g = torch.nn.utils.clip_grad_norm_(
                        list(self.unet.parameters())
                        + list(self.condition_downsample.parameters()), 1.0)
                    self.optimizer.step()
                    self.ema.update(self.unet)
                    last_grad_norm = float(g.item())
            total += float(loss_v.item()) * self.grad_accum
            n += 1
            self.logger.log_batch(loss=float(loss_v.item()),
                                  grad_norm=last_grad_norm)
            pbar.set_postfix(loss_v=f"{loss_v.item():.4f}")
            if bi % 50 == 0:
                torch.cuda.empty_cache()
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.unet.eval(); self.vae.eval()
        total, n = 0.0, 0
        for batch in tqdm(loader, desc='MF-RF Val'):
            prepped = self._prep_batch(batch)
            x1_gt = self._encode_target_block(
                target_5ch_block=prepped['target_5ch_block'],
                cached_latents=prepped.get('target_latents'),
            )
            bs = x1_gt.shape[0]
            x0 = torch.randn_like(x1_gt)
            t = torch.rand(bs, device=self.device)
            t_sp = t[:, None, None, None]
            x_t = (1 - t_sp) * x0 + t_sp * x1_gt
            v_target = x1_gt - x0
            v_pred = self._model_forward(x_t, t, prepped)
            total += float(F.mse_loss(v_pred, v_target).item()); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def sample(self, prepped, num_steps=4):
        """Euler integration from x0~N(0,I) to t=1, return decoded
        (B, K_out, 5, H, W)."""
        self.unet.eval(); self.vae.eval()
        bs = prepped['gridsat_in'].shape[0]
        lat_size = self.cfg['img_size'] // 4
        total_C = self.k_out * self.latent_dim
        x = torch.randn(bs, total_C, lat_size, lat_size, device=self.device)
        enc_cond_full, ctx = self.unet.encode_conditioning(
            prepped['gridsat_in'], prepped['era5_in'],
            prepped['coords_in_flat'], prepped['timestamps_in'])
        enc_cond = self.condition_downsample(enc_cond_full)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tt = torch.full((bs,), i * dt, device=self.device)
            v = self._model_forward(x, tt, prepped, enc_cond=enc_cond, ctx=ctx)
            x = x + v * dt
        x = torch.clamp(x, self.latent_clamp_range[0], self.latent_clamp_range[1])
        return self._decode_target_block(x)

    # --------------------------------------------------------------
    # Per-leadtime evaluation
    # --------------------------------------------------------------

    @torch.no_grad()
    def evaluate_per_leadtime(self, loader, num_samples=200, num_steps=4):
        """Builds the per-leadtime metrics CSV row(s)."""
        self.ema.apply(self.unet)
        self.unet.eval(); self.vae.eval()
        ch_names = ['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']
        calc = MetricsCalculator()
        ds = loader.dataset
        gs_mean = float(ds.gs_mean.flatten()[0]); gs_std = float(ds.gs_std.flatten()[0])
        era5_mean = ds.era5_mean.flatten().tolist()
        era5_std  = ds.era5_std.flatten().tolist()

        rows = []
        ensemble_buffer_for_crps = []   # list of (gen_5ch_block, target) per batch
        # Per-lead steering displacements (degrees), accumulated across all batches
        # for the steering-flow diagnostic plot. Keyed by lead k (1..K_out).
        disp_per_lead = {k + 1: {'gt': [], 'gen': []} for k in range(self.k_out)}
        n_seen = 0
        for batch in loader:
            if n_seen >= num_samples:
                break
            prepped = self._prep_batch(batch)
            gen_5ch_block = self.sample(prepped, num_steps=num_steps)
            tgt_block = prepped['target_5ch_block']

            # Per lead-time metrics
            for k in range(self.k_out):
                gen_k = gen_5ch_block[:, k]       # (B, 5, H, W)
                tgt_k = tgt_block[:, k]
                # Per-channel
                ch_rows = {}
                for ci, cn in enumerate(ch_names):
                    g = gen_k[:, ci:ci+1]; t = tgt_k[:, ci:ci+1]
                    ch_rows[f'{cn}_mse']  = float(F.mse_loss(g, t).item())
                    ch_rows[f'{cn}_mae']  = float(torch.mean(torch.abs(g - t)).item())
                    ch_rows[f'{cn}_psnr'] = calc.psnr(g, t)
                    ch_rows[f'{cn}_ssim'] = calc.ssim(g, t)
                ch_rows['overall_mse']  = calc.mse(gen_k, tgt_k)
                ch_rows['overall_mae']  = calc.mae(gen_k, tgt_k)
                ch_rows['overall_psnr'] = calc.psnr(gen_k, tgt_k)
                ch_rows['overall_ssim'] = calc.ssim(gen_k, tgt_k)
                # LPIPS on GRIDSAT (most natural-image-like)
                ch_rows['gridsat_lpips'] = lpips_metric(gen_k[:, 0:1], tgt_k[:, 0:1])
                # RAPSD log-L1, on GRIDSAT
                ch_rows['gridsat_rapsd_l1'] = rapsd_l1(gen_k[:, 0:1], tgt_k[:, 0:1])
                # CSI/POD/FAR at IR 220K (deep convection) — needs un-normalised GRIDSAT
                gs_pred_K = denormalise_gridsat(gen_k[:, 0:1], gs_mean, gs_std)
                gs_tgt_K  = denormalise_gridsat(tgt_k[:, 0:1], gs_mean, gs_std)
                sk = threshold_skill_scores(gs_pred_K, gs_tgt_K, threshold=220.0, comparison='less')
                ch_rows['csi_220'] = sk['csi']; ch_rows['pod_220'] = sk['pod']; ch_rows['far_220'] = sk['far']
                # MSLP / wind errors
                ch_rows['mslp_err_hpa'] = mslp_err_hpa(gen_k, tgt_k, era5_mean, era5_std)
                wind = max_wind_err(gen_k, tgt_k, era5_mean, era5_std)
                ch_rows.update(wind)
                # Steering-flow displacement → track km + accumulate (dlat, dlon)
                # tuples for the steering-flow scatter/quiver plot below.
                track_km, gt_d, gen_d = self._steering_track_km(
                    gen_k, tgt_k,
                    prepped['cur_lat'], prepped['cur_lon'],
                    era5_mean, era5_std, lead_k=k + 1,
                    return_displacements=True)
                ch_rows['track_km'] = track_km
                disp_per_lead[k + 1]['gt'].extend(gt_d)
                disp_per_lead[k + 1]['gen'].extend(gen_d)

                rows.append({'lead': k + 1, **ch_rows})
            ensemble_buffer_for_crps.append((gen_5ch_block.cpu(), tgt_block.cpu()))
            n_seen += gen_5ch_block.shape[0]

        # CRPS over N=crps_n_samples — re-sample with different noises for the same conditioning
        crps_rows = self._crps_pass(loader, num_steps=num_steps,
                                    num_samples=min(num_samples, 50))
        self.ema.restore(self.unet)

        # Steering-flow plot from accumulated displacements
        try:
            self._plot_steering_flow_multi_lead(disp_per_lead)
        except Exception as e:
            print(f"  Steering-flow plot failed: {e}")

        df = pd.DataFrame(rows)
        out = df.groupby('lead').mean(numeric_only=True).reset_index()
        for k_idx in range(1, self.k_out + 1):
            out.loc[out['lead'] == k_idx, 'crps'] = crps_rows.get(k_idx, float('nan'))
        return out

    def _plot_steering_flow_multi_lead(self, disp_per_lead):
        """3 panels per lead-time × K_out leads: GT vs predicted steering vectors
        and per-lead error histogram. Single PNG → mf_steering_flow.png.
        """
        if not disp_per_lead or all(len(d['gt']) == 0 for d in disp_per_lead.values()):
            return
        leads = sorted(disp_per_lead.keys())
        n_leads = len(leads)
        fig, axes = plt.subplots(n_leads, 3, figsize=(15, 4 * n_leads),
                                  squeeze=False)
        for r, k in enumerate(leads):
            gt  = np.array(disp_per_lead[k]['gt'])    # (N, 2) — (dlat, dlon)
            gen = np.array(disp_per_lead[k]['gen'])
            if len(gt) == 0:
                continue

            # (r, 0) Quiver — GT and predicted
            N = min(len(gt), 150)
            idx = (np.random.default_rng(0).choice(len(gt), N, replace=False)
                   if len(gt) > N else np.arange(len(gt)))
            origins = np.zeros(N)
            axes[r, 0].quiver(origins, origins, gt[idx, 1], gt[idx, 0],
                              color='#4CAF50', alpha=0.4, scale=5, label='GT')
            axes[r, 0].quiver(origins, origins, gen[idx, 1], gen[idx, 0],
                              color='#F44336', alpha=0.4, scale=5, label='Predicted')
            axes[r, 0].set_xlabel('dlon (deg)'); axes[r, 0].set_ylabel('dlat (deg)')
            axes[r, 0].set_title(f't+{k} steering vectors')
            axes[r, 0].legend(fontsize=8); axes[r, 0].grid(True, alpha=0.3)
            axes[r, 0].set_aspect('equal')

            # (r, 1) Scatter — GT vs pred per component
            axes[r, 1].scatter(gt[:, 0], gen[:, 0], alpha=0.4, s=12,
                                c='#2196F3', label='dlat')
            axes[r, 1].scatter(gt[:, 1], gen[:, 1], alpha=0.4, s=12,
                                c='#FF9800', label='dlon')
            lim = max(np.abs(gt).max(), np.abs(gen).max()) * 1.2
            axes[r, 1].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='Perfect')
            mae_lat = float(np.mean(np.abs(gt[:, 0] - gen[:, 0])))
            mae_lon = float(np.mean(np.abs(gt[:, 1] - gen[:, 1])))
            axes[r, 1].set_xlabel('GT (deg)'); axes[r, 1].set_ylabel('Pred (deg)')
            axes[r, 1].set_title(f't+{k}: MAE lat={mae_lat:.3f}, lon={mae_lon:.3f}')
            axes[r, 1].legend(fontsize=8); axes[r, 1].grid(True, alpha=0.3)

            # (r, 2) Error histogram (great-circle deg)
            err = np.sqrt((gt[:, 0] - gen[:, 0]) ** 2 + (gt[:, 1] - gen[:, 1]) ** 2)
            axes[r, 2].hist(err, bins=25, color='#9C27B0', alpha=0.7, edgecolor='white')
            axes[r, 2].axvline(np.mean(err),   color='r',     linestyle='--',
                                label=f'Mean={np.mean(err):.3f}')
            axes[r, 2].axvline(np.median(err), color='orange', linestyle='--',
                                label=f'Median={np.median(err):.3f}')
            axes[r, 2].set_xlabel(f't+{k} displacement error (deg)')
            axes[r, 2].set_ylabel('Count')
            axes[r, 2].set_title(f't+{k} error distribution (N={len(gt)})')
            axes[r, 2].legend(fontsize=8); axes[r, 2].grid(True, alpha=0.3)

        plt.suptitle('Multi-frame steering-flow analysis (per lead time)',
                     fontsize=13, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mf_steering_flow.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def _crps_pass(self, loader, num_steps=4, num_samples=50):
        """Generate N=crps_n_samples ensemble members per batch and compute CRPS."""
        results = {}
        n_seen = 0
        per_lead_crps = {k: [] for k in range(1, self.k_out + 1)}
        for batch in loader:
            if n_seen >= num_samples:
                break
            prepped = self._prep_batch(batch)
            ensemble = []
            for _ in range(self.crps_n_samples):
                ens = self.sample(prepped, num_steps=num_steps)   # (B, K_out, 5, H, W)
                ensemble.append(ens.cpu())
            ens_stack = torch.stack(ensemble, dim=0)   # (N, B, K_out, 5, H, W)
            for k in range(self.k_out):
                samples_k = ens_stack[:, :, k]                              # (N, B, 5, H, W)
                target_k  = prepped['target_5ch_block'][:, k].cpu()         # (B, 5, H, W)
                c = crps_ensemble(samples_k, target_k)
                per_lead_crps[k + 1].append(c)
            n_seen += prepped['gridsat_in'].shape[0]
        for k, vals in per_lead_crps.items():
            results[k] = float(np.mean(vals)) if vals else float('nan')
        return results

    @torch.no_grad()
    def _steering_track_km(self, gen_5ch, tgt_5ch, cur_lat, cur_lon,
                           era5_mean, era5_std, lead_k=1, roi=64,
                           return_displacements=False):
        """Mean great-circle distance (km) between predicted and GT lead-k endpoints
        derived from steering-flow integration over the central ROI.

        If return_displacements=True, also returns two per-sample lists of
        (dlat, dlon) tuples — for the steering-flow scatter/quiver plot.
        """
        B, C, H, W = gen_5ch.shape
        y0 = (H - roi) // 2; x0 = (W - roi) // 2
        u_m, u_s = float(era5_mean[0]), float(era5_std[0])
        v_m, v_s = float(era5_mean[1]), float(era5_std[1])
        gen_u = gen_5ch[:, 1, y0:y0+roi, x0:x0+roi].cpu().numpy() * u_s + u_m
        gen_v = gen_5ch[:, 2, y0:y0+roi, x0:x0+roi].cpu().numpy() * v_s + v_m
        tgt_u = tgt_5ch[:, 1, y0:y0+roi, x0:x0+roi].cpu().numpy() * u_s + u_m
        tgt_v = tgt_5ch[:, 2, y0:y0+roi, x0:x0+roi].cpu().numpy() * v_s + v_m
        dt_sec = lead_k * 3.0 * 3600.0
        kms, gt_d, gen_d = [], [], []
        lat_np = cur_lat.cpu().numpy()
        lon_np = cur_lon.cpu().numpy()
        for b in range(B):
            cl = max(np.cos(np.radians(lat_np[b])), 0.1)
            dlat_gen = float(gen_v[b].mean()) * dt_sec / 111_000.0
            dlon_gen = float(gen_u[b].mean()) * dt_sec / (111_000.0 * cl)
            dlat_gt  = float(tgt_v[b].mean()) * dt_sec / 111_000.0
            dlon_gt  = float(tgt_u[b].mean()) * dt_sec / (111_000.0 * cl)
            kms.append(great_circle_km(
                lat_np[b] + dlat_gt,  lon_np[b] + dlon_gt,
                lat_np[b] + dlat_gen, lon_np[b] + dlon_gen))
            gt_d.append((dlat_gt, dlon_gt))
            gen_d.append((dlat_gen, dlon_gen))
        mean_km = float(np.mean(kms)) if kms else 0.0
        if return_displacements:
            return mean_km, gt_d, gen_d
        return mean_km

    # --------------------------------------------------------------
    # Sampler-step sweep
    # --------------------------------------------------------------

    @torch.no_grad()
    def run_sampler_step_sweep(self, loader, steps_list=(1, 4, 16, 50), num_samples=100):
        self.ema.apply(self.unet)
        self.unet.eval(); self.vae.eval()
        calc = MetricsCalculator()
        rows = []
        for n_steps in steps_list:
            psnrs, ssims, mses = [], [], []
            t0 = time.time(); count = 0
            for batch in loader:
                if count >= num_samples:
                    break
                prepped = self._prep_batch(batch)
                gen = self.sample(prepped, num_steps=n_steps)
                # Average over lead-times for the headline number
                psnrs.append(calc.psnr(gen.flatten(0, 1), prepped['target_5ch_block'].flatten(0, 1)))
                ssims.append(calc.ssim(gen.flatten(0, 1), prepped['target_5ch_block'].flatten(0, 1)))
                mses.append(calc.mse(gen.flatten(0, 1), prepped['target_5ch_block'].flatten(0, 1)))
                count += gen.shape[0]
            elapsed = time.time() - t0
            ms = (elapsed / max(count, 1)) * 1000
            rows.append({
                'n_steps': n_steps,
                'overall_psnr': float(np.mean(psnrs)) if psnrs else 0.0,
                'overall_ssim': float(np.mean(ssims)) if ssims else 0.0,
                'overall_mse':  float(np.mean(mses))  if mses  else 0.0,
                'wallclock_ms_per_sample': ms,
            })
            print(f"  [MF sampler sweep] {n_steps:>3} steps: PSNR={rows[-1]['overall_psnr']:.2f}, "
                  f"SSIM={rows[-1]['overall_ssim']:.4f}, {ms:.1f} ms/sample")
        self.ema.restore(self.unet)
        pd.DataFrame(rows).to_csv(self.metrics_dir / 'sampler_steps.csv', index=False)
        return rows

    # --------------------------------------------------------------
    # Diagnostic plots (per-epoch visual inspection + paper figures)
    # --------------------------------------------------------------

    @torch.no_grad()
    def save_samples(self, loader, epoch, phase='initial', n_samples=4, num_steps=4):
        """Per-epoch visual: for each of `n_samples` storms, plot
            row 0      : last input frame  (t)         — GRIDSAT
            rows 1..K  : ground-truth      (t+1..t+K)  — GRIDSAT
            rows K+1.. : prediction        (t+1..t+K)  — GRIDSAT

        One PNG per call → outputs/<run>/mf_samples_{phase}_ep{epoch}.png
        """
        self.unet.eval(); self.vae.eval()
        self.ema.apply(self.unet)
        try:
            batch = next(iter(loader))
            prepped = self._prep_batch(batch)
            # truncate to n_samples for the plot
            for k in ('gridsat_in', 'era5_in', 'target_5ch_block',
                      'coords_in_flat', 'cur_lat', 'cur_lon',
                      'target_lat', 'target_lon'):
                if k in prepped and torch.is_tensor(prepped[k]):
                    prepped[k] = prepped[k][:n_samples]
            prepped['timestamps_in'] = prepped['timestamps_in'][:n_samples] \
                if isinstance(prepped['timestamps_in'], list) else prepped['timestamps_in']
            if prepped.get('target_latents') is not None:
                prepped['target_latents'] = prepped['target_latents'][:n_samples]

            gen5_block = self.sample(prepped, num_steps=num_steps)   # (B, K_out, 5, H, W)
            B = gen5_block.shape[0]
            K = self.k_out

            # GRIDSAT-only summary (channel 0)
            inp_last_gs = prepped['gridsat_in'][:, -1, 0].cpu().numpy()  # (B, H, W) last input frame
            tgt_gs      = prepped['target_5ch_block'][:, :, 0].cpu().numpy()  # (B, K, H, W)
            gen_gs      = gen5_block[:, :, 0].cpu().numpy()

            n_rows = 1 + 2 * K
            fig, axes = plt.subplots(n_rows, B, figsize=(2.4 * B, 2.4 * n_rows))
            if B == 1:
                axes = axes.reshape(-1, 1)

            for s in range(B):
                axes[0, s].imshow(inp_last_gs[s], cmap='gray')
                axes[0, s].set_xticks([]); axes[0, s].set_yticks([])
                for k in range(K):
                    rt, rp = 1 + 2 * k, 2 + 2 * k
                    axes[rt, s].imshow(tgt_gs[s, k], cmap='gray')
                    axes[rp, s].imshow(gen_gs[s, k], cmap='gray')
                    axes[rt, s].set_xticks([]); axes[rt, s].set_yticks([])
                    axes[rp, s].set_xticks([]); axes[rp, s].set_yticks([])
            axes[0, 0].set_ylabel('Input (t)', fontsize=8, fontweight='bold')
            for k in range(K):
                axes[1 + 2 * k, 0].set_ylabel(f'Target t+{k+1}', fontsize=8, fontweight='bold')
                axes[2 + 2 * k, 0].set_ylabel(f'Gen.   t+{k+1}', fontsize=8, fontweight='bold')
            plt.suptitle(f'MF-RF Samples — {phase} — Epoch {epoch} (GRIDSAT IR)',
                         fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'mf_samples_{phase}_ep{epoch}.png',
                        dpi=130, bbox_inches='tight')
            plt.close()

            # All-5-channel figure (sample 0): rows = channels, columns =
            # [Input t | Target/Gen for each lead]. Confirms the model emits the
            # full GRIDSAT + ERA5 state, not just IR.
            try:
                self._save_samples_all_channels(prepped, gen5_block, epoch, phase)
            except Exception as e:
                print(f"  all-channel sample plot failed at epoch {epoch}: {e}")
        finally:
            self.ema.restore(self.unet)

    def _save_samples_all_channels(self, prepped, gen5_block, epoch, phase):
        """For sample 0, plot every channel (GRIDSAT IR + 4 ERA5) across the
        input frame, ground truth, and prediction for all K_out lead times.
        → outputs/<run>/mf_samples_allch_{phase}_ep{epoch}.png
        """
        ch_names = ['GRIDSAT IR', 'U-wind', 'V-wind', 'Temp', 'Pressure']
        cmaps    = ['gray', 'RdBu_r', 'RdBu_r', 'inferno', 'viridis']
        K = self.k_out

        # Last input frame (5ch) and target/gen blocks for sample 0.
        input_5ch = torch.cat([prepped['gridsat_in'][:, -1],
                               prepped['era5_in'][:, -1]], dim=1).cpu().numpy()  # (B,5,H,W)
        tgt = prepped['target_5ch_block'].cpu().numpy()   # (B,K,5,H,W)
        gen = gen5_block.cpu().numpy()                     # (B,K,5,H,W)
        s = 0

        n_cols = 1 + 2 * K
        fig, axes = plt.subplots(5, n_cols, figsize=(2.2 * n_cols, 2.4 * 5))
        if n_cols == 1:
            axes = axes.reshape(5, 1)

        col_titles = ['Input (t)']
        for k in range(K):
            col_titles += [f'Target t+{k+1}', f'Gen t+{k+1}']

        for ci in range(5):
            # Shared colour scale per channel (robust 1–99 percentile of GT).
            ref = np.concatenate([input_5ch[s, ci].ravel(), tgt[s, :, ci].ravel()])
            vmin, vmax = np.percentile(ref, 1), np.percentile(ref, 99)
            axes[ci, 0].imshow(input_5ch[s, ci], cmap=cmaps[ci], vmin=vmin, vmax=vmax)
            for k in range(K):
                axes[ci, 1 + 2 * k].imshow(tgt[s, k, ci], cmap=cmaps[ci], vmin=vmin, vmax=vmax)
                axes[ci, 2 + 2 * k].imshow(gen[s, k, ci], cmap=cmaps[ci], vmin=vmin, vmax=vmax)
            for c in range(n_cols):
                axes[ci, c].set_xticks([]); axes[ci, c].set_yticks([])
            axes[ci, 0].set_ylabel(ch_names[ci], fontsize=9, fontweight='bold')
        for c in range(n_cols):
            axes[0, c].set_title(col_titles[c], fontsize=8)

        plt.suptitle(f'MF-RF Samples — {phase} — Epoch {epoch} (all 5 channels)',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'mf_samples_allch_{phase}_ep{epoch}.png',
                    dpi=130, bbox_inches='tight')
        plt.close()

    def _plot_training_curves(self):
        """End-of-training: 4-panel diagnostic from the EpochLogger CSV."""
        csv_path = self.logger.csv_path
        if not csv_path.exists():
            return
        df = pd.read_csv(csv_path)
        if df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # (0,0) Train / Val loss
        ax = axes[0, 0]
        if 'train_loss_mean' in df.columns:
            ax.plot(df['epoch'], df['train_loss_mean'], label='Train v-MSE (mean)', linewidth=2)
            if 'train_loss_std' in df.columns:
                lo = df['train_loss_mean'] - df['train_loss_std']
                hi = df['train_loss_mean'] + df['train_loss_std']
                ax.fill_between(df['epoch'], lo, hi, alpha=0.2, label='±1σ (batch)')
        if 'val_loss' in df.columns:
            ax.plot(df['epoch'], df['val_loss'], label='Val v-MSE', linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_yscale('log')
        ax.set_title('Velocity MSE'); ax.legend(); ax.grid(True, alpha=0.3)

        # (0,1) Gradient norm
        ax = axes[0, 1]
        if 'grad_norm_mean' in df.columns:
            ax.plot(df['epoch'], df['grad_norm_mean'], label='Mean', linewidth=2)
        if 'grad_norm_max' in df.columns:
            ax.plot(df['epoch'], df['grad_norm_max'], '--', label='Max', alpha=0.7)
        ax.set_xlabel('Epoch'); ax.set_ylabel('||grad||')
        ax.set_title('Gradient norm (after clip)'); ax.legend(); ax.grid(True, alpha=0.3)

        # (1,0) Learning rate
        ax = axes[1, 0]
        if 'learning_rate' in df.columns:
            ax.plot(df['epoch'], df['learning_rate'], color='purple', linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('LR'); ax.set_yscale('log')
        ax.set_title('Learning rate'); ax.grid(True, alpha=0.3)

        # (1,1) Per-lead PSNR (if available)
        ax = axes[1, 1]
        plotted = False
        for k in range(1, self.k_out + 1):
            col = f'lead{k}_psnr'
            if col in df.columns:
                sub = df[df[col].notna()]
                if not sub.empty:
                    ax.plot(sub['epoch'], sub[col], '-o', label=f't+{k} PSNR', markersize=4)
                    plotted = True
        if plotted:
            ax.set_xlabel('Epoch'); ax.set_ylabel('PSNR (dB)')
            ax.set_title('Per-lead-time PSNR (val)'); ax.legend(); ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No per-lead-time eval data yet\n(runs every eval_every epochs)',
                    ha='center', va='center', transform=ax.transAxes, alpha=0.5)
            ax.set_xticks([]); ax.set_yticks([])

        plt.suptitle(f'MF-RF Training Diagnostics — {self.cfg.get("run_tag", "")}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mf_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_per_leadtime_curves(self):
        """End-of-training: lead-time × metric curves from per_leadtime.csv.

        Shows how prediction quality degrades with forecast horizon — the
        single most important diagnostic for the multi-frame claim.
        """
        per_lead_path = self.metrics_dir / 'per_leadtime.csv'
        if not per_lead_path.exists():
            return
        df = pd.read_csv(per_lead_path)
        if df.empty or 'lead' not in df.columns:
            return

        # Take the LAST eval block (one row per lead)
        last_rows = df.groupby('lead').tail(1).sort_values('lead')

        metrics = [
            ('overall_psnr',  'PSNR (dB) ↑'),
            ('overall_ssim',  'SSIM ↑'),
            ('crps',          'CRPS ↓'),
            ('track_km',      'Track error (km) ↓'),
        ]
        present = [(c, lbl) for c, lbl in metrics if c in last_rows.columns]
        if not present:
            return
        fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4))
        if len(present) == 1:
            axes = [axes]
        for ax, (col, label) in zip(axes, present):
            sub = last_rows[last_rows[col].notna()]
            if not sub.empty:
                ax.plot(sub['lead'], sub[col], 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Lead time (× 3 h)')
                ax.set_ylabel(label)
                ax.set_title(label)
                ax.set_xticks(sorted(sub['lead'].unique()))
                ax.grid(True, alpha=0.3)
        plt.suptitle('Per-lead-time metrics (final eval)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mf_per_leadtime_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    # --------------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------------

    def save_checkpoint(self, epoch, val_loss, phase='initial'):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        ckpt = {
            'epoch': epoch, 'phase': phase,
            'unet_state_dict': self.unet.state_dict(),
            'condition_downsample_state_dict': self.condition_downsample.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema.state_dict(),
            'best_val_loss': self.best_val_loss,
            'latent_scale_factor': self.latent_scale_factor,
            'latent_clamp_range': self.latent_clamp_range,
            'metrics_history': self.metrics_history,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'k_in': self.k_in, 'k_out': self.k_out,
            'config': self.cfg,
        }
        torch.save(ckpt, self.checkpoint_dir / f'mf_{phase}_latest.pt')
        if is_best:
            self.ema.apply(self.unet)
            ckpt['unet_state_dict'] = self.unet.state_dict()
            torch.save(ckpt, self.checkpoint_dir / f'mf_{phase}_best.pt')
            self.ema.restore(self.unet)
            print(f"  ✓ New best MF-{phase} (val_loss: {val_loss:.6f})")

    def _load_checkpoint(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.unet.load_state_dict(ckpt['unet_state_dict'])
            self.condition_downsample.load_state_dict(ckpt['condition_downsample_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.scaler and ckpt.get('scaler_state_dict'):
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.start_epoch = ckpt.get('epoch', 0) + 1

            # LR scheduler restore — honour p2_fresh_lr_schedule. When True, we
            # discard the saved scheduler state AND reset the optimizer LR back
            # to config['learning_rate'], then rebuild the scheduler with T_max
            # covering only the remaining epochs. Use this when the previous
            # cosine schedule already decayed to min_lr (no headroom for more
            # learning) but we want to keep model + optimizer state.
            fresh_schedule = bool(self.cfg.get('fresh_lr_schedule', False))
            if fresh_schedule:
                remaining = max(1, int(self.cfg.get('diffusion_epochs', 100))
                                - self.start_epoch + 1)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.base_lr
                self.lr_scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=remaining,
                    eta_min=self.cfg.get('min_lr', 5e-5),
                )
                print(f"  ✓ Fresh LR schedule: lr={self.base_lr:.2e}, "
                      f"T_max={remaining} epochs (covers remaining)")
            elif ckpt.get('lr_scheduler_state_dict'):
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])

            if 'ema_state_dict' in ckpt:
                self.ema.load_state_dict(ckpt['ema_state_dict'])
            if 'metrics_history' in ckpt:
                self.metrics_history = ckpt['metrics_history']
            if 'train_losses' in ckpt:
                self.train_losses = ckpt['train_losses']; self.val_losses = ckpt['val_losses']
            if ckpt.get('latent_clamp_range'):
                self.latent_clamp_range = ckpt['latent_clamp_range']
            print(f"  ✓ Resumed MF-RF from epoch {self.start_epoch - 1}")
        except Exception as e:
            print(f"  ✗ Checkpoint load failed: {e}")
            self.start_epoch = 1

    # --------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------

    def train(self, train_loader, val_loader):
        num_epochs = int(self.cfg.get('diffusion_epochs', 100))

        # Calibrate scale factor and clamp range from TRAIN data — this is the
        # distribution the model will see during training, so the normalisation
        # should match it (not val, which would leak val statistics).
        if self.latent_scale_factor is None:
            self.compute_latent_scale_factor(train_loader)
        self.compute_latent_clamp_range(train_loader)

        print(f"\n{'='*60}\nPHASE 2 — MULTI-FRAME RECTIFIED FLOW\n{'='*60}")
        print(f"K_in={self.k_in}, K_out={self.k_out}, "
              f"physics={'ON' if self.physics.enabled else 'OFF'}, "
              f"reflow_iterations={self.reflow_iterations}")

        # Phase 1: initial flow
        print(f"\n--- Initial flow ({num_epochs} epochs) ---")
        for epoch in range(self.start_epoch, num_epochs + 1):
            if epoch <= self.warmup_epochs:
                f = 0.1 + 0.9 * (epoch / self.warmup_epochs)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.base_lr * f

            self.logger.start_epoch()
            tl = self.train_epoch(train_loader, epoch, use_reflow=False)
            vl = self.validate(val_loader)
            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            self.train_losses.append(tl); self.val_losses.append(vl)
            print(f"Ep {epoch}/{num_epochs} | Train v-MSE: {tl:.6f} | Val v-MSE: {vl:.6f} | LR: {lr:.2e}")

            # Aggregate per-lead-time eval metrics into mean-across-leads for the
            # one-row-per-epoch CSV. Full per-lead detail still goes to per_leadtime.csv.
            eval_aggregate: dict = {}
            if epoch % self.cfg.get('eval_every', 5) == 0 or epoch == num_epochs:
                try:
                    per_lead = self.evaluate_per_leadtime(
                        val_loader,
                        num_samples=self.cfg.get('eval_samples', 200),
                        num_steps=self.cfg.get('eval_sampler_steps', 4))
                    per_lead.to_csv(self.metrics_dir / 'per_leadtime.csv',
                                    mode='a',
                                    header=not (self.metrics_dir / 'per_leadtime.csv').exists(),
                                    index=False)
                    # Per-lead breakdown (lead1/lead2/lead3 PSNR/SSIM/CRPS/track_km)
                    for _, row in per_lead.iterrows():
                        k = int(row['lead'])
                        eval_aggregate[f"lead{k}_psnr"]     = float(row['overall_psnr'])
                        eval_aggregate[f"lead{k}_ssim"]     = float(row['overall_ssim'])
                        eval_aggregate[f"lead{k}_crps"]     = float(row.get('crps', float('nan')))
                        eval_aggregate[f"lead{k}_track_km"] = float(row.get('track_km', float('nan')))
                    # Mean-across-leads headline columns (matches LD metrics-csv naming)
                    eval_aggregate['mae']  = float(per_lead['overall_mae' ].mean()) if 'overall_mae'  in per_lead else float('nan')
                    eval_aggregate['mse']  = float(per_lead['overall_mse' ].mean()) if 'overall_mse'  in per_lead else float('nan')
                    eval_aggregate['rmse'] = float(eval_aggregate['mse'] ** 0.5) if eval_aggregate.get('mse', float('nan')) == eval_aggregate.get('mse', float('nan')) else float('nan')
                    eval_aggregate['psnr'] = float(per_lead['overall_psnr'].mean())
                    eval_aggregate['ssim'] = float(per_lead['overall_ssim'].mean())
                except Exception as e:
                    print(f"  Per-leadtime eval failed: {e}")

            is_best = vl < self.best_val_loss
            self.logger.end_epoch(
                epoch,
                learning_rate=lr,
                phase='initial',
                is_best=is_best,
                latent_scale_factor=self.latent_scale_factor,
                train_loss=tl,
                val_loss=vl,
                eval_metrics=eval_aggregate,
            )
            self.metrics_history = list(self.logger.history)
            self.save_checkpoint(epoch, vl, phase='initial')

            # Per-epoch visual sample dump every 10 epochs (and at the last epoch)
            if epoch % 10 == 0 or epoch == num_epochs:
                try:
                    self.save_samples(val_loader, epoch, phase='initial')
                except Exception as e:
                    print(f"  save_samples failed at epoch {epoch}: {e}")

        # Final sampler sweep for the initial flow (1-RF row of S3)
        try:
            self.run_sampler_step_sweep(val_loader, steps_list=self.sampler_step_sweep)
        except Exception as e:
            print(f"  Sampler sweep (initial) failed: {e}")

        # Phase 2: reflow iterations
        for ri in range(1, self.reflow_iterations + 1):
            print(f"\n--- Reflow iter {ri}/{self.reflow_iterations} ({self.reflow_epochs} ep) ---")
            self.optimizer = AdamW(
                list(self.unet.parameters()) + list(self.condition_downsample.parameters()),
                lr=self.base_lr * 0.5,
                weight_decay=self.cfg['weight_decay'])
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.reflow_epochs,
                eta_min=self.cfg.get('min_lr', 5e-5))
            for ep in range(1, self.reflow_epochs + 1):
                self.logger.start_epoch()
                tl = self.train_epoch(train_loader, ep, use_reflow=True)
                vl = self.validate(val_loader)
                self.lr_scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Reflow{ri} Ep {ep}/{self.reflow_epochs} | Train: {tl:.6f} | Val: {vl:.6f}")
                is_best = vl < self.best_val_loss
                self.logger.end_epoch(
                    ep,
                    learning_rate=lr,
                    phase=f'reflow_{ri}',
                    is_best=is_best,
                    latent_scale_factor=self.latent_scale_factor,
                    train_loss=tl,
                    val_loss=vl,
                )
                self.metrics_history = list(self.logger.history)
                self.save_checkpoint(ep, vl, phase=f'reflow_{ri}')
                if ep % 10 == 0 or ep == self.reflow_epochs:
                    try:
                        self.save_samples(val_loader, ep, phase=f'reflow_{ri}')
                    except Exception as e:
                        print(f"  save_samples failed at reflow {ri} ep {ep}: {e}")
            try:
                self.run_sampler_step_sweep(val_loader, steps_list=self.sampler_step_sweep)
            except Exception as e:
                print(f"  Sampler sweep (reflow {ri}) failed: {e}")

        # End-of-training paper figures
        try:
            self._plot_training_curves()
            self._plot_per_leadtime_curves()
        except Exception as e:
            print(f"  End-of-training plots failed: {e}")

        print(f"\n{'='*60}\nPHASE 2 COMPLETE — best val: {self.best_val_loss:.6f}\n{'='*60}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Thin entry-point. The canonical config lives in scripts/assembly.py.

    Mirrors assembly.stage_train_multi_frame_rf: when p2_auto_resume is True,
    look for mf_initial_latest.pt in the run_tag's checkpoint folder and pass
    it to the trainer as resume_path.
    """
    from assembly import CONFIG, _p2_config
    from pathlib import Path
    import torch as _torch

    cfg = _p2_config(CONFIG)

    print("\n=== Phase 2: Multi-Frame Rectified Flow ===\n")
    train_loader, val_loader, _ = get_multi_frame_dataloaders(
        cfg['root_dir'],
        cfg['train_years'], cfg['val_years'], cfg['test_years'],
        k_in=cfg['k_in'], k_out=cfg['k_out'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        img_size=cfg['img_size'],
        coord_lookup_path=cfg.get('coord_lookup_path'),
        latent_cache_dir=cfg.get('latent_cache_dir'),
    )

    # Auto-resume: pick up mf_initial_latest.pt if it exists and is below target.
    resume_path = None
    if CONFIG.get('p2_auto_resume', False):
        ckpt_dir = Path(cfg['checkpoint_dir']) / CONFIG.get('run_tag', 'mf_default')
        latest = ckpt_dir / 'mf_initial_latest.pt'
        if latest.exists():
            try:
                meta = _torch.load(str(latest), map_location='cpu', weights_only=False)
                saved_epoch = int(meta.get('epoch', 0))
                if saved_epoch >= int(cfg.get('diffusion_epochs', 100)):
                    print(f"[main] SKIP — checkpoint at epoch {saved_epoch} "
                          f"already reached target {cfg.get('diffusion_epochs')}")
                    return
                resume_path = str(latest)
                print(f"[main] resuming from epoch {saved_epoch} → "
                      f"target epoch {cfg.get('diffusion_epochs')}")
            except Exception as e:
                print(f"[main] couldn't read {latest}: {e}; starting fresh")
                resume_path = None
        else:
            print(f"[main] no checkpoint at {latest}; starting fresh")

    trainer = MultiFrameRFTrainer(cfg, cfg['mc_vae_checkpoint'],
                                   resume_path=resume_path)
    trainer.train(train_loader, val_loader)

    print("\n=== Phase 2 complete ===")


if __name__ == '__main__':
    main()
