"""
Strategy 2: Conditional Flow Matching (OT-CFM)
================================================
Learns a time-dependent velocity field v(x_t, t) that transports
samples from noise (t=0) to data (t=1) via optimal-transport
conditional flows.

Key differences from DDPM diffusion:
 - No noise schedule — uses simple linear interpolation x_t = (1-t)*x0 + t*x1
 - Target is the velocity v = x1 - x0, not noise
 - Sampling via ODE integration (Euler / Heun), not iterative denoising
 - Naturally handles continuous data (ERA5 fields)

Generates 5-channel output (1 GRIDSAT + 4 ERA5) jointly.
"""

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from latent_diffusion import (
    EMA,
    CycloneDataset,
    get_all_storms,
    get_dataloaders,
    SpatialPositionalEncoding,
    TemporalEncoding,
    ConditionEncoder,
    get_time_embeddings,
    ResnetBlock,
    SelfAttention2D,
    DownBlock,
    MidBlock,
    UpBlock,
)
from eval_metrics_extended import MetricsCalculator
from vae import MultiChannelVAE, MultiChannelVAETrainer
from epoch_logger import EpochLogger


# ============================================================================
# FLOW MATCHING UNET  (velocity field predictor)
# ============================================================================

class FlowMatchingUNet(nn.Module):
    """
    UNet that predicts the velocity field v(x_t, t, condition).
    Architecture mirrors DiffusionUNet but the output semantics differ:
      - Input: interpolated state x_t  (latent or pixel), condition features
      - Output: velocity v (same shape as x_t)
    The time embedding now encodes a *continuous* t ∈ [0, 1] instead of
    discrete diffusion timesteps.
    """

    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 channel_mults=(1, 2, 4), t_emb_dim=128, num_heads=4,
                 gridsat_channels=1, era5_channels=4, img_size=64,
                 condition_embed_dim=128, condition_img_size=256):
        super().__init__()
        n_levels = len(channel_mults)
        ch_list = [base_channels * m for m in channel_mults]
        emb_dim = t_emb_dim * 4
        cond_emb_dim = condition_embed_dim * 2

        # Continuous time embedding  (t ∈ [0,1] → sinusoidal → MLP)
        self.t_emb_dim = t_emb_dim
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.context_proj = nn.Sequential(
            nn.Linear(cond_emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim))

        self.condition_encoder = ConditionEncoder(
            gridsat_channels=gridsat_channels, era5_channels=era5_channels,
            img_size=condition_img_size, embed_dim=condition_embed_dim,
            output_channels=base_channels)

        self.init_conv = nn.Conv2d(in_channels + base_channels, ch_list[0], 3, padding=1)

        self.encoders = nn.ModuleList()
        for i in range(n_levels):
            in_ch = ch_list[i - 1] if i > 0 else ch_list[0]
            out_ch = ch_list[i]
            is_last = (i == n_levels - 1)
            self.encoders.append(DownBlock(
                in_ch, out_ch, emb_dim, down_sample=not is_last,
                use_attention=(i == n_levels - 2), num_heads=num_heads))

        self.mid_block = MidBlock(ch_list[-1], emb_dim, num_heads=num_heads)

        # Dynamic decoder configs — supports arbitrary number of levels.
        # (Was hardcoded for n_levels=3, broke at n_levels=4.)
        decoder_cfgs = []
        for i in range(n_levels):
            if i == 0:
                # First decoder level mirrors the last encoder (no upsampling)
                decoder_cfgs.append((ch_list[-1], ch_list[-1], ch_list[-1], False, False))
            else:
                in_ch_d   = ch_list[n_levels - i]
                skip_ch_d = ch_list[n_levels - i - 1]
                out_ch_d  = ch_list[n_levels - i - 1]
                use_attn  = (i == 1)   # attention at the first upsampling level
                decoder_cfgs.append((in_ch_d, skip_ch_d, out_ch_d, True, use_attn))
        self.decoders = nn.ModuleList([
            UpBlock(in_ch, skip_ch, out_ch, emb_dim, up_sample=up, use_attention=attn, num_heads=num_heads)
            for in_ch, skip_ch, out_ch, up, attn in decoder_cfgs])

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch_list[0]), nn.SiLU(),
            nn.Conv2d(ch_list[0], out_channels, 3, padding=1))

    def _continuous_time_embedding(self, t_continuous):
        """Convert continuous t ∈ [0,1] to sinusoidal embedding.
        t_continuous: (B,) float tensor in [0, 1]
        """
        half_dim = self.t_emb_dim // 2
        factor = 10000 ** (torch.arange(0, half_dim, dtype=torch.float32,
                                         device=t_continuous.device) / half_dim)
        t_scaled = t_continuous[:, None].float() * 1000.0  # scale to ~[0, 1000] like diffusion
        t_emb = t_scaled / factor[None, :]
        return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

    def forward(self, x_t, gridsat, era5, timestamps, t_continuous,
                coords=None, encoded_condition=None, context_emb=None):
        """
        Args:
            x_t: interpolated state (B, C, H, W)
            t_continuous: (B,) float tensor in [0, 1]
            coords: (B, 2) per-frame IBTrACS lat/lon (normalised); None falls back
                to image+temporal-only conditioning.
        Returns:
            velocity field v (B, C, H, W)
        """
        t_emb = self._continuous_time_embedding(t_continuous)
        t_emb = self.t_proj(t_emb)

        if encoded_condition is None or context_emb is None:
            encoded_condition, context_emb = self.condition_encoder(
                gridsat, era5, timestamps, coords=coords)
        ctx_emb = self.context_proj(context_emb)
        comb = t_emb + ctx_emb

        x = torch.cat([x_t, encoded_condition], dim=1)
        x = self.init_conv(x)

        skips = []
        for enc in self.encoders:
            x, skip = enc(x, comb)
            skips.append(skip)

        x = self.mid_block(x, comb)

        for i, dec in enumerate(self.decoders):
            x = dec(x, comb, skips[len(skips) - 1 - i])

        return self.final_conv(x)


# ============================================================================
# OT CONDITIONAL FLOW MATCHER
# ============================================================================

class OTFlowMatcher:
    """
    Implements Optimal-Transport Conditional Flow Matching.

    Training:
      x0 ~ N(0, I)   (noise)
      x1 ~ data
      t  ~ U(0, 1)
      x_t = (1 - t) * x0 + t * x1    (straight-line interpolation)
      v_target = x1 - x0              (constant velocity along path)
      loss = MSE(v_pred, v_target)

    Sampling:
      Start from x0 ~ N(0, I), integrate ODE  dx = v(x, t) dt  from t=0 to t=1
    """

    @staticmethod
    def sample_path(x0, x1, t):
        """Interpolate between noise x0 and data x1 at time t.
        t: (B, 1, 1, 1) shaped tensor for broadcasting
        """
        return (1 - t) * x0 + t * x1

    @staticmethod
    def target_velocity(x0, x1):
        """The target velocity field is simply x1 - x0 (straight path)."""
        return x1 - x0

    @staticmethod
    def compute_loss(v_pred, v_target):
        """MSE loss between predicted and target velocity."""
        return F.mse_loss(v_pred, v_target)


# ============================================================================
# ODE SAMPLER (Euler & Heun)
# ============================================================================

class ODESampler:
    """Integrate dx = v(x, t)dt from t=0 to t=1."""

    @staticmethod
    @torch.no_grad()
    def euler(model, x0, gridsat, era5, timestamps,
              encoded_cond, context_emb, num_steps=50, coords=None):
        """Simple Euler integration."""
        dt = 1.0 / num_steps
        x = x0.clone()
        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=torch.float32)
            v = model(x, gridsat, era5, timestamps, t, coords=coords,
                      encoded_condition=encoded_cond, context_emb=context_emb)
            x = x + v * dt
        return x

    @staticmethod
    @torch.no_grad()
    def heun(model, x0, gridsat, era5, timestamps,
             encoded_cond, context_emb, num_steps=50, coords=None):
        """Heun's method (2nd-order) for better accuracy."""
        dt = 1.0 / num_steps
        x = x0.clone()
        for i in range(num_steps):
            t_i = i * dt
            t_ip1 = min((i + 1) * dt, 1.0)

            t_batch = torch.full((x.shape[0],), t_i, device=x.device, dtype=torch.float32)
            v1 = model(x, gridsat, era5, timestamps, t_batch, coords=coords,
                       encoded_condition=encoded_cond, context_emb=context_emb)
            x_euler = x + v1 * dt

            t_batch2 = torch.full((x.shape[0],), t_ip1, device=x.device, dtype=torch.float32)
            v2 = model(x_euler, gridsat, era5, timestamps, t_batch2, coords=coords,
                       encoded_condition=encoded_cond, context_emb=context_emb)

            x = x + 0.5 * (v1 + v2) * dt
        return x


# ============================================================================
# FLOW MATCHING TRAINER
# ============================================================================

class FlowMatchingTrainer:
    """
    Trains FlowMatchingUNet with OT-CFM objective.
    Operates in latent space of a pre-trained 5-channel VAE.
    """

    def __init__(self, config, mc_vae_checkpoint_path, fm_checkpoint_path=None):
        self.config = config
        self.device = torch.device(config['device'])

        self.checkpoint_dir = Path(config['checkpoint_dir']) / 'flow_matching'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config['output_dir']) / 'flow_matching'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load & freeze MC-VAE ----
        print(f"Loading MC-VAE from: {mc_vae_checkpoint_path}")
        ckpt = torch.load(mc_vae_checkpoint_path, map_location=self.device, weights_only=False)
        self.vae = MultiChannelVAE(
            in_channels=5,
            latent_dim=config.get('latent_dim', 4),
            base_channels=config.get('vae_base_channels', 64),
        ).to(self.device)
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        # G2: lazy scale factor. Recomputed from val data in train() if missing.
        self.latent_scale_factor = ckpt.get('latent_scale_factor', None)
        if self.latent_scale_factor is None:
            print("  ⚠ latent_scale_factor missing from checkpoint — will recompute in train()")
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        # G3: latent clamp range (recomputed from data in train())
        self.latent_clamp_range = (-5.0, 5.0)
        print(f"  ✓ MC-VAE frozen, scale={self.latent_scale_factor}")

        # ---- Flow Matching UNet ----
        latent_size = config['img_size'] // 4
        self.unet = FlowMatchingUNet(
            in_channels=config.get('latent_dim', 4),
            out_channels=config.get('latent_dim', 4),
            base_channels=config['base_channels'],
            channel_mults=tuple(config['channel_mults']),
            t_emb_dim=config['t_emb_dim'],
            num_heads=config['num_heads'],
            gridsat_channels=1,
            era5_channels=4,
            img_size=latent_size,
            condition_embed_dim=config['embed_dim'],
            condition_img_size=config['img_size'],
        ).to(self.device)

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
            eta_min=config.get('min_lr', 1e-6),
        )

        self.ema = EMA(self.unet, decay=config.get('ema_decay', 0.9999))
        self.use_amp = config.get('use_amp', True)
        # Conservative init_scale (matches MC-Diff) for float16 stability;
        # ignored under BF16 because GradScaler is unused there.
        self.scaler = (torch.amp.GradScaler('cuda', init_scale=1024, growth_interval=2000)
                       if self.use_amp else None)
        self.grad_accum = config.get('gradient_accumulation_steps', 1)
        self.flow_matcher = OTFlowMatcher()
        self.sampler_method = config.get('flow_sampler', 'heun')  # 'euler' or 'heun'
        self.num_sample_steps = config.get('flow_sample_steps', 50)

        # G4: LR warmup
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.base_lr = config['learning_rate']

        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.train_losses, self.val_losses = [], []
        self.metrics_history = []

        # Rich per-epoch logger (CSV + .log)
        self.logger = EpochLogger(self.metrics_dir, run_tag='fm')

        if fm_checkpoint_path:
            self._load_checkpoint(fm_checkpoint_path)

        n_params = sum(p.numel() for p in self.unet.parameters())
        print(f"FlowMatching Trainer ready. UNet params: {n_params:,}, sampler: {self.sampler_method}")

    def _load_checkpoint(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.unet.load_state_dict(ckpt['unet_state_dict'])
            self.condition_downsample.load_state_dict(ckpt['condition_downsample_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.scaler and 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict']:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.start_epoch = ckpt.get('epoch', 0) + 1
            if 'lr_scheduler_state_dict' in ckpt and ckpt['lr_scheduler_state_dict']:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            if 'ema_state_dict' in ckpt:
                self.ema.load_state_dict(ckpt['ema_state_dict'])
            # G7: restore training-curve history
            if 'metrics_history' in ckpt:
                self.metrics_history = ckpt['metrics_history']
                print(f"  ✓ Metrics history restored ({len(self.metrics_history)} entries)")
            if 'train_losses' in ckpt:
                self.train_losses = ckpt['train_losses']
                self.val_losses  = ckpt['val_losses']
                print(f"  ✓ Loss history restored ({len(self.train_losses)} epochs)")
            # G3: restore the data-derived clamp range if saved
            if ckpt.get('latent_clamp_range'):
                self.latent_clamp_range = ckpt['latent_clamp_range']
            print(f"  ✓ Resumed FlowMatching from epoch {self.start_epoch - 1}")
        except Exception as e:
            print(f"  ✗ Checkpoint load failed: {e} — starting fresh")
            self.start_epoch = 1

    # ---- Latent helpers ----
    @torch.no_grad()
    def encode_to_latent(self, x_5ch):
        mu, logvar = self.vae.encode(x_5ch)
        z = self.vae.reparameterize(mu, logvar)
        return z * self.latent_scale_factor

    @torch.no_grad()
    def decode_from_latent(self, z_scaled):
        return self.vae.decode(z_scaled / self.latent_scale_factor)

    @torch.no_grad()
    def compute_latent_scale_factor(self, dataloader, num_batches=50):
        """G2: recompute latent_scale_factor = 1/std(z) from data.
        Called once in train() if the VAE checkpoint did not save it."""
        self.vae.eval()
        all_z = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            tgt_gs = batch['target'].to(self.device)
            tgt_era5 = batch['target_era5'].to(self.device)
            x5 = torch.cat([tgt_gs, tgt_era5], dim=1)
            mu, logvar = self.vae.encode(x5)
            z = self.vae.reparameterize(mu, logvar)
            all_z.append(z.cpu())
        z_all = torch.cat(all_z, 0)
        z_std = z_all.std().item()
        self.latent_scale_factor = 1.0 / max(z_std, 1e-6)
        print(f"  ✓ Recomputed latent_scale_factor = {self.latent_scale_factor:.5f} (z_std={z_std:.4f})")
        return self.latent_scale_factor

    @torch.no_grad()
    def compute_latent_clamp_range(self, dataloader, num_batches=50):
        """G3: dynamic latent clamp range from data (scaled-latent space).
        Used to clamp the ODE-integrated latent before decoding (prevents drift)."""
        self.vae.eval()
        all_mins, all_maxs = [], []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            tgt_gs = batch['target'].to(self.device)
            tgt_era5 = batch['target_era5'].to(self.device)
            x5 = torch.cat([tgt_gs, tgt_era5], dim=1)
            z_scaled = self.encode_to_latent(x5)
            all_mins.append(z_scaled.min().item())
            all_maxs.append(z_scaled.max().item())
        z_min, z_max = min(all_mins), max(all_maxs)
        margin = (z_max - z_min) * 0.1
        self.latent_clamp_range = (z_min - margin, z_max + margin)
        print(f"  ✓ Latent clamp range = [{self.latent_clamp_range[0]:.2f}, "
              f"{self.latent_clamp_range[1]:.2f}] (raw [{z_min:.3f}, {z_max:.3f}])")
        return self.latent_clamp_range

    # ---- Training ----
    def train_epoch(self, loader, epoch):
        self.unet.train(); self.condition_downsample.train(); self.vae.eval()
        total_loss, n = 0, 0
        pbar = tqdm(loader, desc=f'FM Train Ep {epoch}')

        for bi, batch in enumerate(pbar):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            cur_coords = batch['current_coords'].to(self.device)   # (B, 2) — G1
            target_gs = batch['target'].to(self.device)
            target_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([target_gs, target_era5], dim=1)
            bs = target_5ch.shape[0]

            # Encode data to latent x1
            with torch.no_grad():
                x1 = self.encode_to_latent(target_5ch)  # data latent

            # Sample noise x0
            x0 = torch.randn_like(x1)

            # Sample t ~ U(0, 1)
            t = torch.rand(bs, device=self.device)
            t_spatial = t[:, None, None, None]  # (B, 1, 1, 1) for broadcasting

            # Interpolate & target velocity
            x_t = self.flow_matcher.sample_path(x0, x1, t_spatial)
            v_target = self.flow_matcher.target_velocity(x0, x1)

            # Condition encoding (G1: coords)
            enc_cond, ctx = self.unet.condition_encoder(
                gridsat, era5, timestamps, coords=cur_coords)
            enc_cond = self.condition_downsample(enc_cond)

            if bi % self.grad_accum == 0:
                self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                v_pred = self.unet(x_t, gridsat, era5, timestamps, t,
                                   coords=cur_coords,
                                   encoded_condition=enc_cond, context_emb=ctx)
                loss = self.flow_matcher.compute_loss(v_pred, v_target) / self.grad_accum

            # G4: NaN/Inf guard on loss
            if not torch.isfinite(loss):
                print(f"  ⚠ NaN/Inf loss at batch {bi}, skipping")
                self.logger.log_nan_skip()
                self.optimizer.zero_grad()
                # Do NOT call scaler.update() — scaler.scale() hasn't run for
                # this batch, so no inf checks have been recorded.
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            last_grad_norm = None
            if (bi + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.unet.parameters()) + list(self.condition_downsample.parameters()), 1.0)
                    if not torch.isfinite(grad_norm):
                        print(f"  ⚠ NaN gradients at batch {bi}, skipping step")
                        self.logger.log_nan_skip()
                        self.optimizer.zero_grad()
                        self.scaler.update()
                    else:
                        self.scaler.step(self.optimizer); self.scaler.update()
                        self.ema.update(self.unet)
                        last_grad_norm = float(grad_norm.item())
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.unet.parameters()) + list(self.condition_downsample.parameters()), 1.0)
                    self.optimizer.step()
                    self.ema.update(self.unet)
                    last_grad_norm = float(grad_norm.item())

            total_loss += loss.item() * self.grad_accum; n += 1
            self.logger.log_batch(loss=float(loss.item() * self.grad_accum),
                                  grad_norm=last_grad_norm)
            pbar.set_postfix(loss=f"{loss.item()*self.grad_accum:.4f}")
            if bi % 50 == 0:
                torch.cuda.empty_cache()

        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.unet.eval(); self.vae.eval()
        total_loss, n = 0, 0
        for batch in tqdm(loader, desc='FM Val'):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            cur_coords = batch['current_coords'].to(self.device)   # G1
            target_gs = batch['target'].to(self.device)
            target_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([target_gs, target_era5], dim=1)
            bs = target_5ch.shape[0]

            x1 = self.encode_to_latent(target_5ch)
            x0 = torch.randn_like(x1)
            t = torch.rand(bs, device=self.device)
            t_spatial = t[:, None, None, None]

            x_t = self.flow_matcher.sample_path(x0, x1, t_spatial)
            v_target = self.flow_matcher.target_velocity(x0, x1)

            enc_cond, ctx = self.unet.condition_encoder(
                gridsat, era5, timestamps, coords=cur_coords)
            enc_cond = self.condition_downsample(enc_cond)

            v_pred = self.unet(x_t, gridsat, era5, timestamps, t,
                               coords=cur_coords,
                               encoded_condition=enc_cond, context_emb=ctx)
            total_loss += self.flow_matcher.compute_loss(v_pred, v_target).item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def sample(self, gridsat, era5, timestamps, coords=None, num_steps=None):
        """Generate 5-channel output via ODE integration.
        coords: (B, 2) per-frame IBTrACS lat/lon for conditioning (G1)."""
        self.unet.eval(); self.vae.eval()
        bs = gridsat.shape[0]
        lat_size = self.config['img_size'] // 4
        lat_dim = self.config.get('latent_dim', 4)
        x0 = torch.randn(bs, lat_dim, lat_size, lat_size, device=self.device)

        enc_cond, ctx = self.unet.condition_encoder(
            gridsat, era5, timestamps, coords=coords)
        enc_cond = self.condition_downsample(enc_cond)

        steps = num_steps or self.num_sample_steps
        if self.sampler_method == 'heun':
            z = ODESampler.heun(self.unet, x0, gridsat, era5, timestamps,
                                enc_cond, ctx, num_steps=steps, coords=coords)
        else:
            z = ODESampler.euler(self.unet, x0, gridsat, era5, timestamps,
                                 enc_cond, ctx, num_steps=steps, coords=coords)

        # G3: clamp integrated latent before decoding to stop ODE drift
        if hasattr(self, 'latent_clamp_range') and self.latent_clamp_range is not None:
            z = torch.clamp(z, self.latent_clamp_range[0], self.latent_clamp_range[1])

        return self.decode_from_latent(z)

    # ---- Checkpoint / Samples / Eval ----
    def save_checkpoint(self, epoch, val_loss):
        """G7: persist metrics_history, train_losses, val_losses for resume."""
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        ckpt = {
            'epoch': epoch,
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
            'config': self.config,
        }
        torch.save(ckpt, self.checkpoint_dir / 'fm_latest.pt')
        if is_best:
            self.ema.apply(self.unet)
            ckpt['unet_state_dict'] = self.unet.state_dict()
            torch.save(ckpt, self.checkpoint_dir / 'fm_best.pt')
            self.ema.restore(self.unet)
            print(f"  ✓ New best FM model (val_loss: {val_loss:.6f})")

    def save_samples(self, loader, epoch):
        """G6: 3-row Input/Target/Generated layout per channel."""
        batch = next(iter(loader))
        ns = min(6, len(batch['gridsat']))
        gridsat = batch['gridsat'][:ns].to(self.device)
        era5_in = batch['era5'][:ns].to(self.device)
        timestamps = batch['timestamp'][:ns]
        cur_coords = batch['current_coords'][:ns].to(self.device)
        target_gs = batch['target'][:ns]
        target_era5 = batch['target_era5'][:ns]

        gen5 = self.sample(gridsat, era5_in, timestamps, coords=cur_coords)
        gen_gs = gen5[:, 0:1].cpu().numpy()
        gen_era5 = gen5[:, 1:5].cpu().numpy()
        tgt_gs = target_gs.numpy()
        tgt_era5 = target_era5.numpy()
        inp_gs = batch['gridsat'][:ns].numpy()
        inp_era5 = batch['era5'][:ns].numpy()

        ch_names = ['GRIDSAT', 'U-component', 'V-component', 'Temperature', 'Pressure']
        rows_per_ch = 3   # Input T / Target T+1 / Generated
        n_rows = rows_per_ch * len(ch_names)
        fig, axes = plt.subplots(n_rows, ns, figsize=(2.5 * ns, 2 * n_rows))
        if ns == 1:
            axes = axes.reshape(-1, 1)

        for ch_i, ch_name in enumerate(ch_names):
            r_t   = ch_i * rows_per_ch
            r_t1  = ch_i * rows_per_ch + 1
            r_gen = ch_i * rows_per_ch + 2
            cmap = 'gray' if ch_i == 0 else 'viridis'

            for s in range(ns):
                if ch_i == 0:
                    axes[r_t,   s].imshow(inp_gs[s, 0],    cmap=cmap)
                    axes[r_t1,  s].imshow(tgt_gs[s, 0],    cmap=cmap)
                    axes[r_gen, s].imshow(gen_gs[s, 0],    cmap=cmap)
                else:
                    axes[r_t,   s].imshow(inp_era5[s, ch_i - 1], cmap=cmap)
                    axes[r_t1,  s].imshow(tgt_era5[s, ch_i - 1], cmap=cmap)
                    axes[r_gen, s].imshow(gen_era5[s, ch_i - 1], cmap=cmap)
                for r in (r_t, r_t1, r_gen):
                    axes[r, s].set_xticks([]); axes[r, s].set_yticks([])

            axes[r_t,   0].set_ylabel(f'Input {ch_name}',  fontsize=7, fontweight='bold')
            axes[r_t1,  0].set_ylabel(f'Target {ch_name}', fontsize=7, fontweight='bold')
            axes[r_gen, 0].set_ylabel(f'Gen. {ch_name}',   fontsize=7, fontweight='bold')

        plt.suptitle(f'Flow Matching Samples — Epoch {epoch}', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'fm_samples_ep{epoch}.png', dpi=150)
        plt.close()

    @torch.no_grad()
    def evaluate_era5(self, loader, num_samples=200):
        """G5 + G8: per-channel MSE/MAE/PSNR/SSIM + overall + steering-flow
        displacement, with EMA weights applied throughout."""
        self.ema.apply(self.unet)
        self.unet.eval(); self.vae.eval()
        ch_names = ['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']
        ch_mse  = {c: [] for c in ch_names}
        ch_mae  = {c: [] for c in ch_names}
        ch_psnr = {c: [] for c in ch_names}
        ch_ssim = {c: [] for c in ch_names}
        overall_mse, overall_mae, overall_psnr, overall_ssim, overall_sid = [], [], [], [], []
        gt_displacements, gen_displacements = [], []
        n = 0
        calc = MetricsCalculator()

        ds = loader.dataset
        era5_mean = ds.era5_mean.numpy().flatten()
        era5_std  = ds.era5_std.numpy().flatten()

        for batch in loader:
            if n >= num_samples:
                break
            gridsat = batch['gridsat'].to(self.device)
            era5_in = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            cur_coords = batch['current_coords'].to(self.device)
            tgt_gs = batch['target'].to(self.device)
            tgt_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([tgt_gs, tgt_era5], dim=1)

            gen5 = self.sample(gridsat, era5_in, timestamps, coords=cur_coords)

            for ci, cn in enumerate(ch_names):
                gen_ch = gen5[:, ci:ci+1]
                tgt_ch = target_5ch[:, ci:ci+1]
                ch_mse[cn].append(F.mse_loss(gen_ch, tgt_ch).item())
                ch_mae[cn].append(torch.mean(torch.abs(gen_ch - tgt_ch)).item())
                ch_psnr[cn].append(calc.psnr(gen_ch, tgt_ch))
                ch_ssim[cn].append(calc.ssim(gen_ch, tgt_ch))

            overall_mse.append(calc.mse(gen5, target_5ch))
            overall_mae.append(calc.mae(gen5, target_5ch))
            overall_psnr.append(calc.psnr(gen5, target_5ch))
            overall_ssim.append(calc.ssim(gen5, target_5ch))
            overall_sid.append(calc.sid(gen5, target_5ch))

            # Steering flow displacement (3-hour) from ERA5 U/V — central 64x64 ROI
            roi = 64
            h, w = gen5.shape[2], gen5.shape[3]
            y0, x0 = (h - roi) // 2, (w - roi) // 2
            gt_u  = target_5ch[:, 1, y0:y0+roi, x0:x0+roi].cpu().numpy() * era5_std[0] + era5_mean[0]
            gt_v  = target_5ch[:, 2, y0:y0+roi, x0:x0+roi].cpu().numpy() * era5_std[1] + era5_mean[1]
            gen_u = gen5[:, 1, y0:y0+roi, x0:x0+roi].cpu().numpy() * era5_std[0] + era5_mean[0]
            gen_v = gen5[:, 2, y0:y0+roi, x0:x0+roi].cpu().numpy() * era5_std[1] + era5_mean[1]
            dt_sec = 3.0 * 3600
            for b in range(gt_u.shape[0]):
                lat_val = batch['raw_lat'][b]
                lat_val = float(lat_val) if not isinstance(lat_val, (int, float)) else lat_val
                cos_lat = max(np.cos(np.radians(lat_val)), 0.1)
                gt_displacements.append((gt_v[b].mean() * dt_sec / 111000.0,
                                          gt_u[b].mean() * dt_sec / (111000.0 * cos_lat)))
                gen_displacements.append((gen_v[b].mean() * dt_sec / 111000.0,
                                           gen_u[b].mean() * dt_sec / (111000.0 * cos_lat)))
            n += gridsat.shape[0]

        self.ema.restore(self.unet)

        results = {}
        for cn in ch_names:
            results[f'{cn}_mse']  = float(np.mean(ch_mse[cn]))  if ch_mse[cn]  else 0.0
            results[f'{cn}_mae']  = float(np.mean(ch_mae[cn]))  if ch_mae[cn]  else 0.0
            results[f'{cn}_psnr'] = float(np.mean(ch_psnr[cn])) if ch_psnr[cn] else 0.0
            results[f'{cn}_ssim'] = float(np.mean(ch_ssim[cn])) if ch_ssim[cn] else 0.0
        results['overall_mse']  = float(np.mean(overall_mse))  if overall_mse  else 0.0
        results['overall_mae']  = float(np.mean(overall_mae))  if overall_mae  else 0.0
        results['overall_psnr'] = float(np.mean(overall_psnr)) if overall_psnr else 0.0
        results['overall_ssim'] = float(np.mean(overall_ssim)) if overall_ssim else 0.0
        results['overall_sid']  = float(np.mean(overall_sid))  if overall_sid  else 0.0

        if gt_displacements:
            gt_d  = np.array(gt_displacements)
            gen_d = np.array(gen_displacements)
            results['steering_mae_lat_deg']   = float(np.mean(np.abs(gt_d[:, 0] - gen_d[:, 0])))
            results['steering_mae_lon_deg']   = float(np.mean(np.abs(gt_d[:, 1] - gen_d[:, 1])))
            results['steering_mae_total_deg'] = float(np.mean(np.sqrt(
                (gt_d[:, 0] - gen_d[:, 0]) ** 2 + (gt_d[:, 1] - gen_d[:, 1]) ** 2)))
            self._plot_steering_flow(gt_d, gen_d)

        return results

    def _plot_steering_flow(self, gt_d, gen_d):
        """3-panel steering-flow analysis: quiver, scatter, error histogram."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        N = min(len(gt_d), 200)
        idx = (np.random.choice(len(gt_d), N, replace=False)
               if len(gt_d) > N else np.arange(len(gt_d)))
        origins = np.zeros(N)
        axes[0].quiver(origins, origins, gt_d[idx, 1], gt_d[idx, 0],
                       color='#4CAF50', alpha=0.4, scale=5, label='GT')
        axes[0].quiver(origins, origins, gen_d[idx, 1], gen_d[idx, 0],
                       color='#F44336', alpha=0.4, scale=5, label='Gen')
        axes[0].set_xlabel('dlon (deg)'); axes[0].set_ylabel('dlat (deg)')
        axes[0].set_title('Steering Vectors (3h displacement)')
        axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_aspect('equal')

        axes[1].scatter(gt_d[:, 0], gen_d[:, 0], alpha=0.4, s=15, c='#2196F3', label='dlat')
        axes[1].scatter(gt_d[:, 1], gen_d[:, 1], alpha=0.4, s=15, c='#FF9800', label='dlon')
        lim = max(np.abs(gt_d).max(), np.abs(gen_d).max()) * 1.2
        axes[1].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='Perfect')
        mae_lat = np.mean(np.abs(gt_d[:, 0] - gen_d[:, 0]))
        mae_lon = np.mean(np.abs(gt_d[:, 1] - gen_d[:, 1]))
        axes[1].set_xlabel('GT (deg)'); axes[1].set_ylabel('Gen (deg)')
        axes[1].set_title(f'Displacement Scatter (MAE lat={mae_lat:.3f}, lon={mae_lon:.3f})')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        errors = np.sqrt((gt_d[:, 0] - gen_d[:, 0]) ** 2
                         + (gt_d[:, 1] - gen_d[:, 1]) ** 2)
        axes[2].hist(errors, bins=30, color='#9C27B0', alpha=0.7, edgecolor='white')
        axes[2].axvline(np.mean(errors),   color='r',      linestyle='--', label=f'Mean={np.mean(errors):.3f}')
        axes[2].axvline(np.median(errors), color='orange', linestyle='--', label=f'Median={np.median(errors):.3f}')
        axes[2].set_xlabel('Displacement Error (deg)'); axes[2].set_ylabel('Count')
        axes[2].set_title('Steering Flow Error Distribution')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'Steering Flow Analysis (N={len(gt_d)}, 3h from ERA5 U/V)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fm_steering_flow.png', dpi=150)
        plt.close()

    @torch.no_grad()
    def run_sampler_step_sweep(self, loader, steps_list=(1, 4, 16, 50), num_samples=100):
        """G9: at the final epoch, sweep ODE step counts and record quality + wallclock.
        Writes outputs/.../metrics/sampler_steps.csv with one row per step count."""
        self.ema.apply(self.unet)
        self.unet.eval(); self.vae.eval()
        ch_names = ['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']
        calc = MetricsCalculator()
        rows = []
        for n_steps in steps_list:
            psnrs, ssims, mses = [], [], []
            t0 = time.time()
            count = 0
            for batch in loader:
                if count >= num_samples:
                    break
                gridsat = batch['gridsat'].to(self.device)
                era5_in = batch['era5'].to(self.device)
                timestamps = batch['timestamp']
                cur_coords = batch['current_coords'].to(self.device)
                tgt_gs = batch['target'].to(self.device)
                tgt_era5 = batch['target_era5'].to(self.device)
                tgt5 = torch.cat([tgt_gs, tgt_era5], dim=1)

                gen5 = self.sample(gridsat, era5_in, timestamps,
                                   coords=cur_coords, num_steps=n_steps)
                psnrs.append(calc.psnr(gen5, tgt5))
                ssims.append(calc.ssim(gen5, tgt5))
                mses.append(calc.mse(gen5, tgt5))
                count += gridsat.shape[0]
            elapsed = time.time() - t0
            ms_per_sample = (elapsed / max(count, 1)) * 1000
            rows.append({
                'n_steps': n_steps,
                'overall_psnr': float(np.mean(psnrs)) if psnrs else 0.0,
                'overall_ssim': float(np.mean(ssims)) if ssims else 0.0,
                'overall_mse':  float(np.mean(mses))  if mses  else 0.0,
                'wallclock_ms_per_sample': ms_per_sample,
                'sampler': self.sampler_method,
            })
            print(f"  [Sampler sweep] {n_steps:>3} steps: "
                  f"PSNR={rows[-1]['overall_psnr']:.2f}, "
                  f"SSIM={rows[-1]['overall_ssim']:.4f}, "
                  f"{ms_per_sample:.1f} ms/sample")
        self.ema.restore(self.unet)
        pd.DataFrame(rows).to_csv(self.metrics_dir / 'sampler_steps.csv', index=False)
        return rows

    # ---- Main train loop ----
    def train(self, train_loader, val_loader):
        num_epochs = self.config.get('diffusion_epochs', 100)

        # G2: lazy-compute scale factor from data if VAE checkpoint didn't save it
        if self.latent_scale_factor is None:
            print("  Recomputing latent_scale_factor from validation data...")
            self.compute_latent_scale_factor(val_loader, num_batches=50)

        # G3: dynamic latent clamp range from data
        self.compute_latent_clamp_range(val_loader, num_batches=50)

        print(f"\n{'='*60}\nSTRATEGY 2: CONDITIONAL FLOW MATCHING\n{'='*60}")
        print(f"Epochs: {self.start_epoch} → {num_epochs}, sampler: {self.sampler_method}, "
              f"steps: {self.num_sample_steps}")
        print(f"Latent scale factor: {self.latent_scale_factor:.5f}")

        for epoch in range(self.start_epoch, num_epochs + 1):
            # G4: LR warmup
            if epoch <= self.warmup_epochs:
                warmup_factor = 0.1 + 0.9 * (epoch / self.warmup_epochs)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.base_lr * warmup_factor
                print(f"  Warmup LR: {self.base_lr * warmup_factor:.2e}")

            self.logger.start_epoch()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"Ep {epoch}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.1e}")

            eval_aggregate: dict = {}
            if epoch % self.config.get('eval_every', 5) == 0 or epoch == num_epochs:
                try:
                    ch_metrics = self.evaluate_era5(val_loader,
                                                     num_samples=self.config.get('eval_samples', 200))
                    eval_aggregate.update(ch_metrics)
                    print(f"  ERA5 Metrics: " +
                          ", ".join(f"{k}={v:.4f}" for k, v in ch_metrics.items()))
                except Exception as e:
                    print(f"  Eval failed: {e}")

            is_best = val_loss < self.best_val_loss
            self.logger.end_epoch(
                epoch,
                learning_rate=lr,
                phase='cfm',
                is_best=is_best,
                latent_scale_factor=self.latent_scale_factor,
                train_loss=train_loss,
                val_loss=val_loss,
                eval_metrics=eval_aggregate,
            )
            self.metrics_history = list(self.logger.history)
            self.save_checkpoint(epoch, val_loss)

            if epoch % 10 == 0:
                self.ema.apply(self.unet)
                self.save_samples(val_loader, epoch)
                self.ema.restore(self.unet)

        # G9: sampler-step sweep at final epoch
        try:
            sweep = self.config.get('sampler_step_sweep', [1, 4, 16, 50])
            self.run_sampler_step_sweep(val_loader, steps_list=sweep)
        except Exception as e:
            print(f"  Sampler-step sweep failed: {e}")

        # Final plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train'); plt.plot(self.val_losses, label='Val')
        plt.xlabel('Epoch'); plt.ylabel('Velocity MSE'); plt.title('Flow Matching Training')
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(self.output_dir / 'fm_training_curves.png', dpi=150); plt.close()

        print(f"\n{'='*60}\nFLOW MATCHING COMPLETE — best val: {self.best_val_loss:.6f}\n{'='*60}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Thin entry-point. The canonical config lives in scripts/assembly.py.

    Equivalent to: python scripts/assembly.py --stage train_single_frame_rf
    (but with the CFM objective rather than 2-RF).
    """
    from assembly import CONFIG, _p1_config
    from multi_channel_diffusion import get_dataloaders

    cfg = _p1_config(CONFIG)
    cfg["flow_sampler"]      = CONFIG.get("p1_flow_sampler", "heun")
    cfg["flow_sample_steps"] = CONFIG.get("p1_flow_sample_steps", 50)

    print("\n=== Strategy 2: Conditional Flow Matching ===\n")
    train_loader, val_loader, _ = get_dataloaders(
        root_dir=cfg["root_dir"],
        train_years=cfg["train_years"],
        val_years=cfg["val_years"],
        test_years=cfg["test_years"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        img_size=cfg["img_size"],
        coord_lookup_path=cfg.get("coord_lookup_path"),
    )

    trainer = FlowMatchingTrainer(cfg, cfg["mc_vae_checkpoint"])
    trainer.train(train_loader, val_loader)

    print("\n=== Strategy 2 Complete ===")


if __name__ == '__main__':
    main()
