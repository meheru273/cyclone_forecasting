"""
Strategy 3: Rectified Flows + Score Matching
=============================================
Two complementary approaches in one script:

3a — Rectified Flows:
  - Uses straight-line interpolation like flow matching
  - Adds iterative "reflow" to straighten transport paths
  - Straightened paths → better few-step / one-step generation

3b — Denoising Score Matching:
  - Learns ∇_x log p_σ(x)  (score function) at multiple noise levels
  - Sampling via annealed Langevin dynamics
  - Good for continuous data distributions

Both generate 5-channel output (1 GRIDSAT + 4 ERA5) in latent space.
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
    ResnetBlock,
    SelfAttention2D,
    DownBlock,
    MidBlock,
    UpBlock,
)
from eval_metrics_extended import MetricsCalculator
from epoch_logger import EpochLogger
from vae import MultiChannelVAE, MultiChannelVAETrainer
from conditional_flow_matching import FlowMatchingUNet, ODESampler


# ============================================================================
# SCORE NETWORK  (predicts score ∇_x log p_σ(x))
# ============================================================================

class ScoreUNet(nn.Module):
    """
    UNet that predicts the score function ∇_x log p_σ(x).
    Uses noise level σ as conditioning (encoded via log-σ embedding).
    Architecture mirrors FlowMatchingUNet.
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

        # Noise-level embedding (log σ → sinusoidal → MLP)
        self.t_emb_dim = t_emb_dim
        self.sigma_proj = nn.Sequential(
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

        # Dynamic decoder — supports arbitrary n_levels (was hardcoded to 3)
        decoder_cfgs = []
        for i in range(n_levels):
            if i == 0:
                decoder_cfgs.append((ch_list[-1], ch_list[-1], ch_list[-1], False, False))
            else:
                in_ch_d   = ch_list[n_levels - i]
                skip_ch_d = ch_list[n_levels - i - 1]
                out_ch_d  = ch_list[n_levels - i - 1]
                use_attn  = (i == 1)
                decoder_cfgs.append((in_ch_d, skip_ch_d, out_ch_d, True, use_attn))
        self.decoders = nn.ModuleList([
            UpBlock(in_ch, skip_ch, out_ch, emb_dim, up_sample=up, use_attention=attn, num_heads=num_heads)
            for in_ch, skip_ch, out_ch, up, attn in decoder_cfgs])

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch_list[0]), nn.SiLU(),
            nn.Conv2d(ch_list[0], out_channels, 3, padding=1))

    def _sigma_embedding(self, sigma):
        """Convert noise level σ to embedding via log(σ)."""
        half_dim = self.t_emb_dim // 2
        factor = 10000 ** (torch.arange(0, half_dim, dtype=torch.float32,
                                         device=sigma.device) / half_dim)
        log_sigma = torch.log(sigma.clamp(min=1e-5))[:, None] * 100.0
        emb = log_sigma / factor[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x_noisy, gridsat, era5, timestamps, sigma,
                coords=None, encoded_condition=None, context_emb=None):
        """
        Args:
            x_noisy: noisy data (B, C, H, W)
            sigma: noise level (B,) float tensor
            coords: (B, 2) per-frame IBTrACS lat/lon (G1)
        Returns:
            score ∇_x log p_σ(x) with shape (B, C, H, W)
        """
        s_emb = self._sigma_embedding(sigma)
        s_emb = self.sigma_proj(s_emb)

        if encoded_condition is None or context_emb is None:
            encoded_condition, context_emb = self.condition_encoder(
                gridsat, era5, timestamps, coords=coords)
        ctx_emb = self.context_proj(context_emb)
        comb = s_emb + ctx_emb

        x = torch.cat([x_noisy, encoded_condition], dim=1)
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
# RECTIFIED FLOW TRAINER
# ============================================================================

class RectifiedFlowTrainer:
    """
    Rectified Flows: train a flow model with straight-line interpolation,
    then optionally "reflow" to straighten transport paths for faster sampling.

    Training phase 1 (initial flow):
      Same as OT-CFM  —  x_t = (1-t)*x0 + t*x1,  v_target = x1 - x0

    Training phase 2 (reflow, optional):
      Use the trained model to generate pairs (x0, x̂1) by running the ODE
      from noise x0 to get x̂1 = ODE(x0, 0→1). Then retrain with these
      straightened pairs: v_target = x̂1 - x0.

    This results in straighter paths → better 1-step or few-step generation.
    """

    def __init__(self, config, mc_vae_checkpoint_path, rf_checkpoint_path=None):
        self.config = config
        self.device = torch.device(config['device'])

        self.checkpoint_dir = Path(config['checkpoint_dir']) / 'rectified_flow'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config['output_dir']) / 'rectified_flow'
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
        # G2: lazy scale factor
        self.latent_scale_factor = ckpt.get('latent_scale_factor', None)
        if self.latent_scale_factor is None:
            print("  ⚠ latent_scale_factor missing from checkpoint — will recompute in train()")
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        # G3: latent clamp range (recomputed from data in train())
        self.latent_clamp_range = (-5.0, 5.0)
        print(f"  ✓ MC-VAE frozen, scale={self.latent_scale_factor}")

        # ---- Flow UNet (same architecture as flow matching) ----
        latent_size = config['img_size'] // 4
        self.unet = FlowMatchingUNet(
            in_channels=config.get('latent_dim', 4),
            out_channels=config.get('latent_dim', 4),
            base_channels=config['base_channels'],
            channel_mults=tuple(config['channel_mults']),
            t_emb_dim=config['t_emb_dim'],
            num_heads=config['num_heads'],
            gridsat_channels=1, era5_channels=4,
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
        self.scaler = (torch.amp.GradScaler('cuda', init_scale=1024, growth_interval=2000)
                       if self.use_amp else None)
        self.grad_accum = config.get('gradient_accumulation_steps', 1)

        self.reflow_iterations = config.get('reflow_iterations', 1)   # 0 = no reflow
        self.reflow_sample_steps = config.get('reflow_sample_steps', 20)

        # G4: LR warmup
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.base_lr = config['learning_rate']

        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.train_losses, self.val_losses = [], []
        self.metrics_history = []

        # Rich per-epoch logger (CSV + .log)
        self.logger = EpochLogger(self.metrics_dir, run_tag='rf')

        if rf_checkpoint_path:
            self._load_checkpoint(rf_checkpoint_path)

        print(f"RectifiedFlow Trainer ready. Reflow iterations: {self.reflow_iterations}")

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
            # G7: restore history
            if 'metrics_history' in ckpt:
                self.metrics_history = ckpt['metrics_history']
                print(f"  ✓ Metrics history restored ({len(self.metrics_history)} entries)")
            if 'train_losses' in ckpt:
                self.train_losses = ckpt['train_losses']
                self.val_losses  = ckpt['val_losses']
                print(f"  ✓ Loss history restored ({len(self.train_losses)} epochs)")
            if ckpt.get('latent_clamp_range'):
                self.latent_clamp_range = ckpt['latent_clamp_range']
            print(f"  ✓ Resumed RectifiedFlow from epoch {self.start_epoch - 1}")
        except Exception as e:
            print(f"  ✗ Checkpoint load failed: {e}")
            self.start_epoch = 1

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
        """G2: recompute latent_scale_factor = 1/std(z) from data."""
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
        """G3: dynamic latent clamp range from data (scaled-latent space)."""
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

    # ---- Core training step ----
    def _train_step(self, batch, use_reflow_pairs=False):
        """
        One training step. If use_reflow_pairs, generate x1 by running the
        current model from noise instead of using ground truth.

        G1 fix: when generating reflow pairs, the conditioning vector must
        come from the SAME batch that seeded the noise — including the per-frame
        coords. Without this, the reflowed pairs are mis-conditioned and the
        second-stage objective collapses toward unconditional flow.
        """
        gridsat = batch['gridsat'].to(self.device)
        era5_in = batch['era5'].to(self.device)
        timestamps = batch['timestamp']
        cur_coords = batch['current_coords'].to(self.device)   # G1
        target_gs = batch['target'].to(self.device)
        target_era5 = batch['target_era5'].to(self.device)
        target_5ch = torch.cat([target_gs, target_era5], dim=1)
        bs = target_5ch.shape[0]

        with torch.no_grad():
            x1_gt = self.encode_to_latent(target_5ch)

        x0 = torch.randn_like(x1_gt)

        if use_reflow_pairs:
            # Generate x̂1 from x0 using current model (straightened target).
            # G1: condition with the same coords as the loss forward below.
            with torch.no_grad():
                enc_cond_full, ctx = self.unet.condition_encoder(
                    gridsat, era5_in, timestamps, coords=cur_coords)
                enc_cond_ds = self.condition_downsample(enc_cond_full)
                x1 = ODESampler.euler(self.unet, x0, gridsat, era5_in, timestamps,
                                       enc_cond_ds, ctx,
                                       num_steps=self.reflow_sample_steps,
                                       coords=cur_coords)
        else:
            x1 = x1_gt

        t = torch.rand(bs, device=self.device)
        t_spatial = t[:, None, None, None]
        x_t = (1 - t_spatial) * x0 + t_spatial * x1
        v_target = x1 - x0

        enc_cond, ctx = self.unet.condition_encoder(
            gridsat, era5_in, timestamps, coords=cur_coords)
        enc_cond = self.condition_downsample(enc_cond)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            v_pred = self.unet(x_t, gridsat, era5_in, timestamps, t,
                               coords=cur_coords,
                               encoded_condition=enc_cond, context_emb=ctx)
            loss = F.mse_loss(v_pred, v_target)

        return loss

    def train_epoch(self, loader, epoch, use_reflow=False):
        self.unet.train(); self.condition_downsample.train(); self.vae.eval()
        total_loss, n = 0, 0
        label = 'Reflow' if use_reflow else 'RF'
        pbar = tqdm(loader, desc=f'{label} Train Ep {epoch}')

        for bi, batch in enumerate(pbar):
            if bi % self.grad_accum == 0:
                self.optimizer.zero_grad()

            loss = self._train_step(batch, use_reflow_pairs=use_reflow) / self.grad_accum

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
        total, n = 0, 0
        for batch in tqdm(loader, desc='RF Val'):
            gridsat = batch['gridsat'].to(self.device)
            era5_in = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            cur_coords = batch['current_coords'].to(self.device)   # G1
            tgt_gs = batch['target'].to(self.device)
            tgt_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([tgt_gs, tgt_era5], dim=1)
            bs = target_5ch.shape[0]

            x1 = self.encode_to_latent(target_5ch)
            x0 = torch.randn_like(x1)
            t = torch.rand(bs, device=self.device)
            t_sp = t[:, None, None, None]
            x_t = (1 - t_sp) * x0 + t_sp * x1
            v_target = x1 - x0

            enc_cond, ctx = self.unet.condition_encoder(
                gridsat, era5_in, timestamps, coords=cur_coords)
            enc_cond = self.condition_downsample(enc_cond)
            v_pred = self.unet(x_t, gridsat, era5_in, timestamps, t,
                               coords=cur_coords,
                               encoded_condition=enc_cond, context_emb=ctx)
            total += F.mse_loss(v_pred, v_target).item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def sample(self, gridsat, era5, timestamps, coords=None, num_steps=50):
        """G1: coords passed through to condition encoder and ODE.
        G3: clamp integrated latent before decoding to stop ODE drift."""
        self.unet.eval(); self.vae.eval()
        bs = gridsat.shape[0]
        lat_size = self.config['img_size'] // 4
        lat_dim = self.config.get('latent_dim', 4)
        x0 = torch.randn(bs, lat_dim, lat_size, lat_size, device=self.device)

        enc_cond, ctx = self.unet.condition_encoder(
            gridsat, era5, timestamps, coords=coords)
        enc_cond = self.condition_downsample(enc_cond)

        z = ODESampler.euler(self.unet, x0, gridsat, era5, timestamps,
                              enc_cond, ctx, num_steps=num_steps, coords=coords)

        if hasattr(self, 'latent_clamp_range') and self.latent_clamp_range is not None:
            z = torch.clamp(z, self.latent_clamp_range[0], self.latent_clamp_range[1])

        return self.decode_from_latent(z)

    def save_checkpoint(self, epoch, val_loss, phase='initial'):
        """G7: persist metrics_history, train_losses, val_losses for resume."""
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
            'config': self.config,
        }
        torch.save(ckpt, self.checkpoint_dir / f'rf_{phase}_latest.pt')
        if is_best:
            self.ema.apply(self.unet)
            ckpt['unet_state_dict'] = self.unet.state_dict()
            torch.save(ckpt, self.checkpoint_dir / f'rf_{phase}_best.pt')
            self.ema.restore(self.unet)
            print(f"  ✓ New best RF-{phase} (val_loss: {val_loss:.6f})")

    def save_samples(self, loader, epoch, phase='initial'):
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
        rows_per_ch = 3
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

        plt.suptitle(f'RF-{phase} Samples — Epoch {epoch}', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'rf_{phase}_samples_ep{epoch}.png', dpi=150)
        plt.close()

    @torch.no_grad()
    def evaluate_era5(self, loader, num_samples=200):
        """G5 + G8: per-channel MSE/MAE/PSNR/SSIM + overall + steering-flow,
        with EMA weights applied throughout."""
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

            # Steering flow displacement (3h) from ERA5 U/V
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
        """3-panel steering-flow analysis."""
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
        plt.savefig(self.output_dir / 'rf_steering_flow.png', dpi=150)
        plt.close()

    @torch.no_grad()
    def run_sampler_step_sweep(self, loader, steps_list=(1, 4, 16, 50), num_samples=100):
        """G9: sweep ODE step counts at the final epoch. The 1-step row is
        the headline of the 2-RF few-step claim."""
        self.ema.apply(self.unet)
        self.unet.eval(); self.vae.eval()
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
                'sampler': 'euler',
            })
            print(f"  [RF sampler sweep] {n_steps:>3} steps: "
                  f"PSNR={rows[-1]['overall_psnr']:.2f}, "
                  f"SSIM={rows[-1]['overall_ssim']:.4f}, "
                  f"{ms_per_sample:.1f} ms/sample")
        self.ema.restore(self.unet)
        pd.DataFrame(rows).to_csv(self.metrics_dir / 'sampler_steps.csv', index=False)
        return rows

    def train(self, train_loader, val_loader):
        num_epochs = self.config.get('diffusion_epochs', 100)

        # G2: recompute scale factor from data if VAE checkpoint missed it
        if self.latent_scale_factor is None:
            print("  Recomputing latent_scale_factor from validation data...")
            self.compute_latent_scale_factor(val_loader, num_batches=50)

        # G3: dynamic latent clamp range from data
        self.compute_latent_clamp_range(val_loader, num_batches=50)

        print(f"\n{'='*60}\nSTRATEGY 3a: RECTIFIED FLOWS\n{'='*60}")
        print(f"Latent scale factor: {self.latent_scale_factor:.5f}")

        # Phase 1: Initial flow training
        print(f"\n--- Phase 1: Initial Flow Training ({num_epochs} epochs) ---")
        for epoch in range(self.start_epoch, num_epochs + 1):
            # G4: LR warmup (initial-flow phase only)
            if epoch <= self.warmup_epochs:
                warmup_factor = 0.1 + 0.9 * (epoch / self.warmup_epochs)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.base_lr * warmup_factor
                print(f"  Warmup LR: {self.base_lr * warmup_factor:.2e}")

            self.logger.start_epoch()
            train_loss = self.train_epoch(train_loader, epoch, use_reflow=False)
            val_loss = self.validate(val_loader)
            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"Ep {epoch}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.1e}")

            eval_aggregate: dict = {}
            if epoch % self.config.get('eval_every', 5) == 0 or epoch == num_epochs:
                try:
                    ch_m = self.evaluate_era5(val_loader, self.config.get('eval_samples', 200))
                    eval_aggregate.update(ch_m)
                    print(f"  ERA5: " + ", ".join(f"{k}={v:.4f}" for k, v in ch_m.items()))
                except Exception as e:
                    print(f"  Eval failed: {e}")

            is_best = val_loss < self.best_val_loss
            self.logger.end_epoch(
                epoch,
                learning_rate=lr,
                phase='initial',
                is_best=is_best,
                latent_scale_factor=self.latent_scale_factor,
                train_loss=train_loss,
                val_loss=val_loss,
                eval_metrics=eval_aggregate,
            )
            self.metrics_history = list(self.logger.history)
            self.save_checkpoint(epoch, val_loss, phase='initial')
            if epoch % 10 == 0:
                self.ema.apply(self.unet)
                self.save_samples(val_loader, epoch, phase='initial')
                self.ema.restore(self.unet)

        # G9: sampler-step sweep after the initial phase (1-RF baseline numbers)
        try:
            self.run_sampler_step_sweep(
                val_loader, steps_list=self.config.get('sampler_step_sweep', [1, 4, 16, 50]))
        except Exception as e:
            print(f"  Sampler-step sweep (initial) failed: {e}")

        # Phase 2: Reflow iterations
        for reflow_iter in range(1, self.reflow_iterations + 1):
            reflow_epochs = self.config.get('reflow_epochs', num_epochs // 2)
            print(f"\n--- Phase 2: Reflow Iteration {reflow_iter}/{self.reflow_iterations} "
                  f"({reflow_epochs} epochs) ---")

            # Reset optimizer and scheduler for reflow
            self.optimizer = AdamW(
                list(self.unet.parameters()) + list(self.condition_downsample.parameters()),
                lr=self.config['learning_rate'] * 0.5,  # lower LR for refinement
                weight_decay=self.config['weight_decay'],
            )
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=reflow_epochs,
                eta_min=self.config.get('min_lr', 1e-6),
            )

            for epoch in range(1, reflow_epochs + 1):
                self.logger.start_epoch()
                train_loss = self.train_epoch(train_loader, epoch, use_reflow=True)
                val_loss = self.validate(val_loader)
                self.lr_scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Reflow {reflow_iter} Ep {epoch}/{reflow_epochs} | "
                      f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

                is_best = val_loss < self.best_val_loss
                self.logger.end_epoch(
                    epoch,
                    learning_rate=lr,
                    phase=f'reflow_{reflow_iter}',
                    is_best=is_best,
                    latent_scale_factor=self.latent_scale_factor,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                self.metrics_history = list(self.logger.history)
                self.save_checkpoint(epoch, val_loss, phase=f'reflow_{reflow_iter}')

            # G9: sampler-step sweep after this reflow stage
            try:
                self.run_sampler_step_sweep(
                    val_loader,
                    steps_list=self.config.get('sampler_step_sweep', [1, 4, 16, 50]))
            except Exception as e:
                print(f"  Sampler-step sweep (reflow {reflow_iter}) failed: {e}")

            # Quick few-step diagnostic
            print(f"\n  Few-step diagnostic after reflow {reflow_iter}:")
            try:
                self.ema.apply(self.unet)
                batch = next(iter(val_loader))
                gs = batch['gridsat'][:4].to(self.device)
                e5 = batch['era5'][:4].to(self.device)
                ts = batch['timestamp'][:4]
                cc = batch['current_coords'][:4].to(self.device)
                tgt = torch.cat([batch['target'][:4], batch['target_era5'][:4]], dim=1).to(self.device)
                for steps in [1, 2, 5, 10, 50]:
                    gen = self.sample(gs, e5, ts, coords=cc, num_steps=steps)
                    mse = F.mse_loss(gen, tgt).item()
                    print(f"    {steps}-step MSE: {mse:.6f}")
            except Exception as e:
                print(f"    Few-step diagnostic failed: {e}")
            finally:
                self.ema.restore(self.unet)

        print(f"\n{'='*60}\nRECTIFIED FLOWS COMPLETE\n{'='*60}")


# ============================================================================
# SCORE MATCHING (DENOISING SCORE MATCHING) TRAINER
# ============================================================================

class ScoreMatchingTrainer:
    """
    Denoising Score Matching (DSM) with multiple noise levels.
    Learns ∇_x log p_σ(x) at geometrically spaced noise levels.
    Sampling via annealed Langevin dynamics.

    DSM loss:  E[σ² ||s_θ(x + σε, σ) - (-ε/σ)||²]
             = E[σ² ||s_θ(x̃, σ) + (x̃ - x)/σ²||²]
    """

    def __init__(self, config, mc_vae_checkpoint_path, sm_checkpoint_path=None):
        self.config = config
        self.device = torch.device(config['device'])

        self.checkpoint_dir = Path(config['checkpoint_dir']) / 'score_matching'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config['output_dir']) / 'score_matching'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load & freeze MC-VAE ----
        ckpt = torch.load(mc_vae_checkpoint_path, map_location=self.device)
        self.vae = MultiChannelVAE(
            in_channels=5,
            latent_dim=config.get('latent_dim', 4),
            base_channels=config.get('vae_base_channels', 64),
        ).to(self.device)
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        self.latent_scale_factor = ckpt.get('latent_scale_factor', 0.18215)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # ---- Score UNet ----
        latent_size = config['img_size'] // 4
        self.score_net = ScoreUNet(
            in_channels=config.get('latent_dim', 4),
            out_channels=config.get('latent_dim', 4),
            base_channels=config['base_channels'],
            channel_mults=tuple(config['channel_mults']),
            t_emb_dim=config['t_emb_dim'],
            num_heads=config['num_heads'],
            gridsat_channels=1, era5_channels=4,
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

        # Geometric noise schedule:  σ_min to σ_max
        self.num_noise_levels = config.get('num_noise_levels', 10)
        self.sigma_min = config.get('sigma_min', 0.01)
        self.sigma_max = config.get('sigma_max', 1.0)
        self.sigmas = torch.exp(torch.linspace(
            np.log(self.sigma_max), np.log(self.sigma_min),
            self.num_noise_levels, device=self.device))

        # Langevin dynamics parameters
        self.langevin_steps = config.get('langevin_steps', 100)
        self.langevin_eps = config.get('langevin_eps', 2e-5)

        self.optimizer = AdamW(
            list(self.score_net.parameters()) + list(self.condition_downsample.parameters()),
            lr=config['learning_rate'], weight_decay=config['weight_decay'],
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.get('diffusion_epochs', 100),
            eta_min=config.get('min_lr', 1e-6),
        )

        self.ema = EMA(self.score_net, decay=config.get('ema_decay', 0.9999))
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.grad_accum = config.get('gradient_accumulation_steps', 1)

        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.train_losses, self.val_losses = [], []
        self.metrics_history = []

        if sm_checkpoint_path:
            self._load_checkpoint(sm_checkpoint_path)

        print(f"ScoreMatching Trainer ready. σ range: [{self.sigma_min}, {self.sigma_max}], "
              f"levels: {self.num_noise_levels}, Langevin steps: {self.langevin_steps}")

    def _load_checkpoint(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.score_net.load_state_dict(ckpt['score_net_state_dict'])
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
            print(f"  ✓ Resumed ScoreMatching from epoch {self.start_epoch - 1}")
        except Exception as e:
            print(f"  ✗ Checkpoint load failed: {e}")
            self.start_epoch = 1

    @torch.no_grad()
    def encode_to_latent(self, x_5ch):
        mu, logvar = self.vae.encode(x_5ch)
        z = self.vae.reparameterize(mu, logvar)
        return z * self.latent_scale_factor

    @torch.no_grad()
    def decode_from_latent(self, z_scaled):
        return self.vae.decode(z_scaled / self.latent_scale_factor)

    def _dsm_loss(self, score_pred, noise, sigma):
        """
        Denoising score matching loss.
        target score = -noise / sigma   (since x̃ = x + σ*noise)
        loss = σ² * ||score_pred - (-noise/σ)||²
             = σ² * ||score_pred + noise/σ||²
        """
        target_score = -noise / sigma[:, None, None, None]
        diff = score_pred - target_score
        # Weight by σ² for proper scaling (Song & Ermon 2019)
        loss = 0.5 * (sigma[:, None, None, None] ** 2 * diff ** 2).mean()
        return loss

    def train_epoch(self, loader, epoch):
        self.score_net.train(); self.condition_downsample.train(); self.vae.eval()
        total_loss, n = 0, 0
        pbar = tqdm(loader, desc=f'SM Train Ep {epoch}')

        for bi, batch in enumerate(pbar):
            gridsat = batch['gridsat'].to(self.device)
            era5_in = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            tgt_gs = batch['target'].to(self.device)
            tgt_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([tgt_gs, tgt_era5], dim=1)
            bs = target_5ch.shape[0]

            with torch.no_grad():
                x = self.encode_to_latent(target_5ch)

            # Sample random noise level index
            sigma_idx = torch.randint(0, self.num_noise_levels, (bs,), device=self.device)
            sigma = self.sigmas[sigma_idx]

            # Add noise
            noise = torch.randn_like(x)
            x_noisy = x + sigma[:, None, None, None] * noise

            enc_cond, ctx = self.score_net.condition_encoder(gridsat, era5_in, timestamps)
            enc_cond = self.condition_downsample(enc_cond)

            if bi % self.grad_accum == 0:
                self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                score_pred = self.score_net(x_noisy, gridsat, era5_in, timestamps, sigma,
                                            encoded_condition=enc_cond, context_emb=ctx)
                loss = self._dsm_loss(score_pred, noise, sigma) / self.grad_accum

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (bi + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.score_net.parameters()) + list(self.condition_downsample.parameters()), 1.0)
                    self.scaler.step(self.optimizer); self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.score_net.parameters()) + list(self.condition_downsample.parameters()), 1.0)
                    self.optimizer.step()
                self.ema.update(self.score_net)

            total_loss += loss.item() * self.grad_accum; n += 1
            pbar.set_postfix(loss=f"{loss.item()*self.grad_accum:.4f}")
            if bi % 50 == 0:
                torch.cuda.empty_cache()

        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.score_net.eval(); self.vae.eval()
        total, n = 0, 0
        for batch in tqdm(loader, desc='SM Val'):
            gridsat = batch['gridsat'].to(self.device)
            era5_in = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            tgt_gs = batch['target'].to(self.device)
            tgt_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([tgt_gs, tgt_era5], dim=1)
            bs = target_5ch.shape[0]

            x = self.encode_to_latent(target_5ch)
            sigma_idx = torch.randint(0, self.num_noise_levels, (bs,), device=self.device)
            sigma = self.sigmas[sigma_idx]
            noise = torch.randn_like(x)
            x_noisy = x + sigma[:, None, None, None] * noise

            enc_cond, ctx = self.score_net.condition_encoder(gridsat, era5_in, timestamps)
            enc_cond = self.condition_downsample(enc_cond)
            score_pred = self.score_net(x_noisy, gridsat, era5_in, timestamps, sigma,
                                        encoded_condition=enc_cond, context_emb=ctx)
            total += self._dsm_loss(score_pred, noise, sigma).item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def sample(self, gridsat, era5, timestamps):
        """Annealed Langevin dynamics sampling."""
        self.score_net.eval(); self.vae.eval()
        bs = gridsat.shape[0]
        lat_size = self.config['img_size'] // 4
        lat_dim = self.config.get('latent_dim', 4)

        # Start from random noise
        x = torch.randn(bs, lat_dim, lat_size, lat_size, device=self.device) * self.sigma_max

        enc_cond, ctx = self.score_net.condition_encoder(gridsat, era5, timestamps)
        enc_cond = self.condition_downsample(enc_cond)

        # Anneal through noise levels from large to small
        for sigma_val in self.sigmas:
            sigma_batch = torch.full((bs,), sigma_val.item(), device=self.device)
            step_size = self.langevin_eps * (sigma_val / self.sigmas[-1]) ** 2

            for _ in range(self.langevin_steps):
                score = self.score_net(x, gridsat, era5, timestamps, sigma_batch,
                                       encoded_condition=enc_cond, context_emb=ctx)
                noise = torch.randn_like(x)
                x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        return self.decode_from_latent(x)

    def save_checkpoint(self, epoch, val_loss):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        ckpt = {
            'epoch': epoch,
            'score_net_state_dict': self.score_net.state_dict(),
            'condition_downsample_state_dict': self.condition_downsample.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema.state_dict(),
            'best_val_loss': self.best_val_loss,
            'latent_scale_factor': self.latent_scale_factor,
            'config': self.config,
        }
        torch.save(ckpt, self.checkpoint_dir / 'sm_latest.pt')
        if is_best:
            self.ema.apply(self.score_net)
            ckpt['score_net_state_dict'] = self.score_net.state_dict()
            torch.save(ckpt, self.checkpoint_dir / 'sm_best.pt')
            self.ema.restore(self.score_net)
            print(f"  ✓ New best SM (val_loss: {val_loss:.6f})")

    def save_samples(self, loader, epoch):
        batch = next(iter(loader))
        ns = min(6, len(batch['gridsat']))
        gridsat = batch['gridsat'][:ns].to(self.device)
        era5_in = batch['era5'][:ns].to(self.device)
        target_gs = batch['target'][:ns]
        target_era5 = batch['target_era5'][:ns]
        timestamps = batch['timestamp'][:ns]

        gen5 = self.sample(gridsat, era5_in, timestamps)
        gen_gs = gen5[:, 0:1].cpu().numpy()
        gen_era5 = gen5[:, 1:5].cpu().numpy()
        tgt_gs = target_gs.numpy()
        tgt_era5 = target_era5.numpy()

        ch_names = ['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']
        fig, axes = plt.subplots(2*len(ch_names), ns, figsize=(2*ns, 2*2*len(ch_names)))
        if ns == 1:
            axes = axes.reshape(-1, 1)
        for s in range(ns):
            axes[0, s].imshow(tgt_gs[s, 0], cmap='gray'); axes[0, s].axis('off')
            axes[1, s].imshow(gen_gs[s, 0], cmap='gray'); axes[1, s].axis('off')
            if s == 0:
                axes[0, s].set_ylabel('GS Tgt', fontsize=7)
                axes[1, s].set_ylabel('GS Pred', fontsize=7)
            for ch in range(4):
                r_t, r_p = 2+2*ch, 3+2*ch
                axes[r_t, s].imshow(tgt_era5[s, ch], cmap='viridis'); axes[r_t, s].axis('off')
                axes[r_p, s].imshow(gen_era5[s, ch], cmap='viridis'); axes[r_p, s].axis('off')
                if s == 0:
                    axes[r_t, s].set_ylabel(f'{ch_names[ch+1]} Tgt', fontsize=6)
                    axes[r_p, s].set_ylabel(f'{ch_names[ch+1]} Pred', fontsize=6)
        plt.suptitle(f'Score Matching Samples — Epoch {epoch}', fontsize=10)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'sm_samples_ep{epoch}.png', dpi=120)
        plt.close()

    @torch.no_grad()
    def evaluate_era5(self, loader, num_samples=100):
        self.ema.apply(self.score_net)
        self.score_net.eval(); self.vae.eval()
        ch_names = ['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']
        ch_mse = {c: [] for c in ch_names}
        ch_mae = {c: [] for c in ch_names}
        n = 0
        for batch in loader:
            if n >= num_samples:
                break
            gridsat = batch['gridsat'].to(self.device)
            era5_in = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            tgt_gs = batch['target'].to(self.device)
            tgt_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([tgt_gs, tgt_era5], dim=1)
            gen5 = self.sample(gridsat, era5_in, timestamps)
            for ci, cn in enumerate(ch_names):
                ch_mse[cn].append(F.mse_loss(gen5[:, ci:ci+1], target_5ch[:, ci:ci+1]).item())
                ch_mae[cn].append(torch.mean(torch.abs(gen5[:, ci:ci+1] - target_5ch[:, ci:ci+1])).item())
            n += gridsat.shape[0]
        self.ema.restore(self.score_net)
        return {f'{cn}_{m}': float(np.mean(vals)) if vals else 0
                for cn in ch_names for m, vals in [('mse', ch_mse[cn]), ('mae', ch_mae[cn])]}

    def train(self, train_loader, val_loader):
        num_epochs = self.config.get('diffusion_epochs', 100)
        print(f"\n{'='*60}\nSTRATEGY 3b: DENOISING SCORE MATCHING\n{'='*60}")
        print(f"σ schedule: {self.sigma_max:.4f} → {self.sigma_min:.4f} ({self.num_noise_levels} levels)")

        for epoch in range(self.start_epoch, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"Ep {epoch}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.1e}")

            epoch_m = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr}
            if epoch % self.config.get('eval_every', 5) == 0 or epoch == num_epochs:
                try:
                    ch_m = self.evaluate_era5(val_loader, self.config.get('eval_samples', 50))
                    epoch_m.update(ch_m)
                    print(f"  ERA5: " + ", ".join(f"{k}={v:.4f}" for k, v in ch_m.items()))
                except Exception as e:
                    print(f"  Eval failed: {e}")
            self.metrics_history.append(epoch_m)
            pd.DataFrame(self.metrics_history).to_csv(self.metrics_dir / 'sm_metrics.csv', index=False)
            self.save_checkpoint(epoch, val_loss)
            if epoch % 10 == 0:
                self.ema.apply(self.score_net)
                self.save_samples(val_loader, epoch)
                self.ema.restore(self.score_net)

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train'); plt.plot(self.val_losses, label='Val')
        plt.xlabel('Epoch'); plt.ylabel('DSM Loss'); plt.title('Score Matching Training')
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(self.output_dir / 'sm_training_curves.png', dpi=150); plt.close()

        print(f"\n{'='*60}\nSCORE MATCHING COMPLETE — best val: {self.best_val_loss:.6f}\n{'='*60}")


# ============================================================================
# MAIN — runs both 3a (Rectified Flows) and 3b (Score Matching)
# ============================================================================

def main():
    """Thin entry-point. The canonical config lives in scripts/assembly.py.

    Equivalent to: python scripts/assembly.py --stage train_single_frame_rf
    (this script also runs the Score Matching path, which assembly.py skips).
    """
    from assembly import CONFIG, _p1_config
    from multi_channel_diffusion import get_dataloaders

    cfg = _p1_config(CONFIG)

    print("\n=== Strategy 3: Rectified Flows + Score Matching ===\n")
    train_loader, val_loader, _ = get_dataloaders(
        root_dir=cfg['root_dir'],
        train_years=cfg['train_years'],
        val_years=cfg['val_years'],
        test_years=cfg['test_years'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        img_size=cfg['img_size'],
        coord_lookup_path=cfg.get('coord_lookup_path'),
    )

    mc_vae_path = cfg['mc_vae_checkpoint']

    # Strategy 3a: Rectified Flows (the Phase-1 RF reference)
    print("\n" + "=" * 60)
    print("RUNNING STRATEGY 3a: RECTIFIED FLOWS")
    print("=" * 60)
    rf_trainer = RectifiedFlowTrainer(cfg, mc_vae_path)
    rf_trainer.train(train_loader, val_loader)

    # Strategy 3b: Score Matching (low priority — see phase1.md §1.7)
    print("\n" + "=" * 60)
    print("RUNNING STRATEGY 3b: DENOISING SCORE MATCHING")
    print("=" * 60)
    sm_trainer = ScoreMatchingTrainer(cfg, mc_vae_path)
    sm_trainer.train(train_loader, val_loader)

    print("\n=== Strategy 3 Complete ===")


if __name__ == '__main__':
    main()
