"""
Multi-channel VAE — Stage 1 of the latent generative pipeline.

Encodes/decodes 5-channel cyclone tensors (1 GRIDSAT IR + 4 ERA5 fields:
U-wind, V-wind, Temperature, Pressure) to a 4-dim latent space at 1/4
spatial resolution.

This module is the canonical home for the VAE. All downstream models
(multi-channel diffusion, single-frame rectified flow, multi-frame
rectified flow) import the VAE from here. Configs are centralised in
`scripts/assembly.py` — this module's `main()` is a thin standalone
entry point for VAE-only debugging.

Standalone run:
    python scripts/vae.py            # uses assembly.CONFIG defaults

Library use:
    from vae import MultiChannelVAE, MultiChannelVAETrainer
"""

from __future__ import annotations

import json
from pathlib import Path

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

from eval_metrics_extended import MetricsCalculator, FID
from epoch_logger import EpochLogger


# ============================================================================
# CORE VAE  — 5-channel joint GRIDSAT + ERA5
# ============================================================================

class MultiChannelVAE(nn.Module):
    """VAE that encodes/decodes 5-channel tensors (GRIDSAT + ERA5).

    Coordinates handled separately at the conditioning level — *not* an input
    channel here.
    """

    def __init__(self, in_channels: int = 5, latent_dim: int = 4,
                 base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim  = latent_dim

        # Encoder: 256 → 128 → 64
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.GroupNorm(8, base_channels), nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.GroupNorm(8, base_channels * 2), nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.GroupNorm(8, base_channels * 2), nn.SiLU(),
        )
        self.fc_mu     = nn.Conv2d(base_channels * 2, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(base_channels * 2, latent_dim, 1)

        # Decoder: 64 → 128 → 256, Upsample+Conv (no checkerboard from ConvTranspose)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, base_channels * 2, 1),
            nn.GroupNorm(8, base_channels * 2), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.GroupNorm(8, base_channels), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.GroupNorm(8, base_channels), nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1),
            # No Tanh — z-score data is unbounded
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ============================================================================
# TRAINER
# ============================================================================

class MultiChannelVAETrainer:
    """Trains MultiChannelVAE on (target_gs, target_era5) pairs from the
    standard `CycloneDataset` (single-frame) — or any loader that yields
    batches with keys 'target' and 'target_era5'."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['device'])

        self.checkpoint_dir = Path(config['checkpoint_dir']) / 'mc_vae'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config['output_dir']) / 'mc_vae'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.vae = MultiChannelVAE(
            in_channels=5,
            latent_dim=config.get('latent_dim', 4),
            base_channels=config.get('vae_base_channels', 64),
        ).to(self.device)

        self.optimizer = AdamW(
            self.vae.parameters(),
            lr=config.get('vae_lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
        )
        # T_max = 2x so cosine only decays, never cycles back up
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('vae_epochs', 50) * 2,
            eta_min=config.get('min_lr', 1e-6),
        )

        self.kl_weight_target = config.get('kl_weight', 1e-4)
        self.kl_anneal_start  = config.get('kl_anneal_start', 0)
        self.kl_anneal_end    = config.get('kl_anneal_end',  10)

        self.best_val_loss = float('inf')
        self.metrics_history = []

        # Rich per-epoch metric logger (CSV + human-readable .log)
        self.logger = EpochLogger(self.metrics_dir, run_tag='mc_vae')

        n_params = sum(p.numel() for p in self.vae.parameters())
        print(f"MultiChannelVAE initialized: {n_params:,} params, 5-channel I/O")

    # --- helpers ---
    def _get_kl_weight(self, epoch):
        if epoch <= self.kl_anneal_start:
            return 0.0
        if epoch >= self.kl_anneal_end:
            return self.kl_weight_target
        progress = (epoch - self.kl_anneal_start) / max(
            1, self.kl_anneal_end - self.kl_anneal_start)
        return self.kl_weight_target * progress

    @staticmethod
    def _build_target(batch, device):
        gs   = batch['target'].to(device)      # (B, 1, H, W)
        era5 = batch['target_era5'].to(device) # (B, 4, H, W)
        return torch.cat([gs, era5], dim=1)    # (B, 5, H, W)

    # --- training ---
    def train_epoch(self, loader, epoch):
        self.vae.train()
        kl_w = self._get_kl_weight(epoch)
        total_loss, total_recon, total_kl, n = 0, 0, 0, 0

        pbar = tqdm(loader, desc=f'MC-VAE Train Ep {epoch} [β={kl_w:.2e}]')
        for batch in pbar:
            target = self._build_target(batch, self.device)
            self.optimizer.zero_grad()
            recon, mu, logvar = self.vae(target)

            per_ch_mae = torch.abs(recon - target).mean(dim=[0, 2, 3])
            ch_var = target.var(dim=[0, 2, 3]).detach()
            recon_loss = (per_ch_mae / (ch_var + 1e-6)).mean()
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_w * kl_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
            self.optimizer.step()
            self.logger.log_batch(loss=loss.item(),
                                  grad_norm=float(grad_norm.item()))

            total_loss += loss.item(); total_recon += recon_loss.item()
            total_kl   += kl_loss.item(); n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             recon=f"{recon_loss.item():.4f}")

        return {k: v / max(n, 1) for k, v in
                zip(['total', 'recon', 'kl'], [total_loss, total_recon, total_kl])}

    @torch.no_grad()
    def validate(self, loader):
        self.vae.eval()
        total_loss, total_recon, total_kl, n = 0, 0, 0, 0
        mse_gs, mse_era5 = 0, 0
        metric_accum = {'mse': 0, 'psnr': 0, 'ssim': 0, 'sid': 0, 'kl_divergence': 0}

        for batch in tqdm(loader, desc='MC-VAE Val'):
            target = self._build_target(batch, self.device)
            recon, mu, logvar = self.vae(target)

            per_ch_mae = torch.abs(recon - target).mean(dim=[0, 2, 3])
            ch_var = target.var(dim=[0, 2, 3]).detach()
            recon_loss = (per_ch_mae / (ch_var + 1e-6)).mean()
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + self.kl_weight_target * kl_loss

            total_loss += loss.item(); total_recon += recon_loss.item()
            total_kl   += kl_loss.item()
            mse_gs   += F.mse_loss(recon[:, 0:1], target[:, 0:1]).item()
            mse_era5 += F.mse_loss(recon[:, 1:5], target[:, 1:5]).item()

            metric_accum['mse']           += MetricsCalculator.mse(recon, target)
            metric_accum['psnr']          += MetricsCalculator.psnr(recon, target)
            metric_accum['ssim']          += MetricsCalculator.ssim(recon, target)
            metric_accum['sid']           += MetricsCalculator.sid(recon, target)
            metric_accum['kl_divergence'] += kl_loss.item()
            n += 1

        d = max(n, 1)
        for key in metric_accum:
            metric_accum[key] /= d
        result = {'total': total_loss/d, 'recon': total_recon/d, 'kl': total_kl/d,
                  'mse_gridsat': mse_gs/d, 'mse_era5': mse_era5/d}
        result.update(metric_accum)
        return result

    def check_latent_statistics(self, loader, num_batches=50, use_mu_only=True):
        """Estimate 1/std(latent) for downstream flow/diffusion normalisation.

        IMPORTANT correctness choices:
          - Use **mu** (deterministic), not the sampled `z = mu + std*eps`.
            The reparameterisation noise inflates std and biases sf low.
          - Caller is expected to pass the **train** loader so the scale
            matches the data the downstream model will train on.
            (Earlier code used val_loader — a minor data leak.)
        """
        if len(loader.dataset) == 0:
            raise RuntimeError(
                "check_latent_statistics: loader is empty — check your years config.")
        self.vae.eval()
        all_mu = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                target = self._build_target(batch, self.device)
                mu, logvar = self.vae.encode(target)
                if use_mu_only:
                    all_mu.append(mu.cpu())
                else:
                    z = self.vae.reparameterize(mu, logvar)
                    all_mu.append(z.cpu())
        if not all_mu:
            raise RuntimeError(
                "check_latent_statistics: collected zero latent batches.")
        z = torch.cat(all_mu, 0)
        z_std = z.std().item()
        sf = 1.0 / z_std
        self._latent_scale_factor = sf
        print(f"[MC-VAE latent stats] mu mean={z.mean().item():.4f}, "
              f"std={z_std:.4f}, scale_factor={sf:.5f}  "
              f"(source: {'mu' if use_mu_only else 'sampled z'}, "
              f"{len(z)} samples)")
        return sf

    # --- checkpoint ---
    def save_checkpoint(self, epoch, val_loss):
        # Refuse to mark a zero-val-loss epoch as "best" — that means the val
        # loader was empty and the loss is meaningless. Better to keep no best
        # than save a poisoned one.
        if val_loss == 0.0:
            print(f"  ⚠ val_loss == 0.0 — refusing to update best-checkpoint. "
                  f"Empty val loader?")
            torch.save({
                'epoch': epoch,
                'vae_state_dict': self.vae.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'latent_scale_factor': getattr(self, '_latent_scale_factor', None),
                'config': self.config,
            }, self.checkpoint_dir / 'mc_vae_latest.pt')
            return

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        ckpt = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'latent_scale_factor': getattr(self, '_latent_scale_factor', None),
            'config': self.config,
        }
        torch.save(ckpt, self.checkpoint_dir / 'mc_vae_latest.pt')
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'mc_vae_best.pt')
            print(f"  ✓ New best MC-VAE (val_loss: {val_loss:.4f})")

    def load_checkpoint(self):
        p = self.checkpoint_dir / 'mc_vae_latest.pt'
        if not p.exists():
            print("  No MC-VAE checkpoint found — training from scratch.")
            return 0
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        ep = ckpt['epoch']
        print(f"  ✓ Resumed MC-VAE from epoch {ep}")
        return ep

    def save_reconstruction_samples(self, loader, epoch):
        self.vae.eval()
        batch = next(iter(loader))
        bs = min(6, len(batch['target']))
        target = self._build_target(batch, self.device)[:bs]
        with torch.no_grad():
            recon, _, _ = self.vae(target)

        target_np = target.cpu().numpy()
        recon_np  = recon.cpu().numpy()
        ch_names = ['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']

        fig, axes = plt.subplots(2 * len(ch_names), bs,
                                  figsize=(2 * bs, 2 * 2 * len(ch_names)))
        for ch_i, ch_name in enumerate(ch_names):
            for s in range(bs):
                r_orig  = 2 * ch_i
                r_recon = r_orig + 1
                axes[r_orig,  s].imshow(target_np[s, ch_i], cmap='viridis')
                axes[r_orig,  s].axis('off')
                if s == 0:
                    axes[r_orig, s].set_ylabel(f'{ch_name}\nOrig',  fontsize=7)
                axes[r_recon, s].imshow(recon_np[s, ch_i], cmap='viridis')
                axes[r_recon, s].axis('off')
                if s == 0:
                    axes[r_recon, s].set_ylabel(f'{ch_name}\nRecon', fontsize=7)
        plt.suptitle(f'MC-VAE Recon — Epoch {epoch}', fontsize=10)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'mc_vae_recon_ep{epoch}.png', dpi=120)
        plt.close()

    def _compute_fid(self, val_loader, max_batches=20):
        try:
            fid_calc = FID(device=str(self.device))
            real_feats, fake_feats = [], []
            self.vae.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= max_batches:
                        break
                    target = self._build_target(batch, self.device)
                    recon, _, _ = self.vae(target)
                    real_feats.append(fid_calc.get_features(target[:, 0:1]))
                    fake_feats.append(fid_calc.get_features(recon[:, 0:1]))
            real_feats = np.concatenate(real_feats, axis=0)
            fake_feats = np.concatenate(fake_feats, axis=0)
            fid_score = fid_calc.calculate_fid(real_feats, fake_feats)
            print(f"  FID Score (MC-VAE recon vs. real): {fid_score:.4f}")
            return fid_score
        except Exception as e:
            print(f"  Warning: FID computation failed: {e}")
            return float('nan')

    # --- main train loop ---
    def train(self, train_loader, val_loader):
        num_epochs = self.config.get('vae_epochs', 50)
        start_epoch = self.load_checkpoint()
        if start_epoch >= num_epochs:
            print("MC-VAE already trained — skipping.")
            # Compute scale-factor from TRAIN (the distribution the downstream
            # flow/diffusion model will see), using mu (deterministic).
            return self.check_latent_statistics(train_loader, use_mu_only=True)

        print(f"\n{'='*60}\nSTAGE 1: MULTI-CHANNEL VAE TRAINING "
              f"({num_epochs - start_epoch} epochs remaining)\n{'='*60}")

        fid_every = self.config.get('fid_every', 5)

        for epoch in range(start_epoch, num_epochs):
            ep = epoch + 1
            kl_weight = self._get_kl_weight(ep)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.logger.start_epoch()
            train_m = self.train_epoch(train_loader, ep)
            val_m   = self.validate(val_loader)
            self.lr_scheduler.step()

            epoch_fid = float('nan')
            if ep % fid_every == 0 or ep == num_epochs:
                epoch_fid = self._compute_fid(val_loader)

            # Determine is_best BEFORE save_checkpoint (which mutates best_val_loss)
            is_best = val_m['total'] < self.best_val_loss and val_m['total'] > 0.0

            # Rich per-epoch row — schema matches the legacy LD metrics CSV
            self.logger.end_epoch(
                ep,
                learning_rate=current_lr,
                phase='vae',
                is_best=is_best,
                kl_weight=kl_weight,
                latent_scale_factor=getattr(self, '_latent_scale_factor', None),
                train_loss=train_m['total'],
                train_recon=train_m['recon'],
                train_kl=train_m['kl'],
                val_loss=val_m['total'],
                eval_metrics={
                    'mae':         val_m.get('mae', val_m.get('recon')),
                    'mse':         val_m['mse'],
                    'rmse':        float(val_m['mse']) ** 0.5 if val_m['mse'] is not None else float('nan'),
                    'psnr':        val_m['psnr'],
                    'ssim':        val_m['ssim'],
                    'sid':         val_m['sid'],
                    'fid':         epoch_fid,
                    'mse_gridsat': val_m['mse_gridsat'],
                    'mse_era5':    val_m['mse_era5'],
                    'kl':          val_m['kl'],
                    'kl_divergence': val_m['kl_divergence'],
                },
            )
            self.metrics_history = list(self.logger.history)

            self.save_checkpoint(ep, val_m['total'])
            if ep % 10 == 0:
                self.save_reconstruction_samples(val_loader, ep)

        # Calibrate scale factor from TRAIN data (not val) using mu only.
        # This is what downstream flow/diffusion will normalise against.
        scale_factor = self.check_latent_statistics(train_loader, use_mu_only=True)

        if num_epochs % fid_every != 0:
            final_fid = self._compute_fid(val_loader)
            if self.logger.history:
                self.logger.history[-1]['fid'] = final_fid
                # rewrite CSV with the corrected last row
                pd.DataFrame(self.logger.history).to_csv(
                    self.metrics_dir / 'mc_vae_metrics.csv', index=False)
            self.metrics_history = list(self.logger.history)
        else:
            final_fid = (self.logger.history[-1].get('fid', float('nan'))
                         if self.logger.history else float('nan'))

        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'config': self.config,
            'latent_scale_factor': scale_factor,
            'best_val_loss': self.best_val_loss,
            'fid_score': final_fid,
        }, self.checkpoint_dir / 'mc_vae_final.pt')

        pd.DataFrame(self.metrics_history).to_csv(
            self.metrics_dir / 'mc_vae_metrics.csv', index=False)

        config_save = {k: str(v) if isinstance(v, Path) else v
                        for k, v in self.config.items()}
        try:
            with open(self.output_dir / 'mc_vae_config.json', 'w') as f:
                json.dump(config_save, f, indent=4, default=str)
        except Exception as e:
            print(f"  Warning: Could not save config JSON: {e}")

        print("MC-VAE TRAINING COMPLETE")
        return scale_factor


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

def main():
    """Standalone VAE-only training. Pulls config from assembly.CONFIG."""
    from assembly import CONFIG, build_vae_dataloaders

    train_loader, val_loader, _ = build_vae_dataloaders(CONFIG)
    trainer = MultiChannelVAETrainer(CONFIG)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
