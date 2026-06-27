import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from pathlib import Path
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import inception_v3
import numpy as np
import pandas as pd
import random
import json
import math
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    """Maintains an exponential moving average of model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply(self, model):
        """Swap EMA weights into model (save originals for restore)."""
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model):
        """Restore original training weights."""
        if self.backup is not None:
            model.load_state_dict(self.backup)
            self.backup = None

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = copy.deepcopy(state_dict)


# ============================================================================
# DATA LOADER (unchanged from original)
# ============================================================================

class CycloneDataset(Dataset):
    def __init__(self, root_dir, years, gridsat_type='GRIDSAT_data.npy',
                 era5_type='ERA5_data.npy', transform=None, max_samples=None,
                 mode='train', img_size=256):
        self.root_dir = Path(root_dir)
        self.gridsat_type = gridsat_type
        self.era5_type = era5_type
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        self.mode = mode  # 'train' / 'val' / 'test' — controls augmentation only

        # Global normalization stats — hardcoded from 5,000-sample dataset scan
        # GRIDSAT  mean=169.8992  std=122.2612
        # ERA5[0]  mean=  -1.1655  std=  5.3889
        # ERA5[1]  mean=   0.3835  std=  5.0042
        # ERA5[2]  mean= 298.4892  std=  4.2326
        # ERA5[3]  mean=100976.29  std=521.1500
        self.gs_mean   = torch.tensor([169.8992],                              dtype=torch.float32).view(-1, 1, 1)
        self.gs_std    = torch.tensor([122.2612],                              dtype=torch.float32).view(-1, 1, 1)
        self.era5_mean = torch.tensor([-1.1655, 0.3835, 298.4892, 100976.2891], dtype=torch.float32).view(-1, 1, 1)
        self.era5_std  = torch.tensor([5.3889,  5.0042,   4.2326,    521.1500], dtype=torch.float32).view(-1, 1, 1)
        self.use_global_norm = True
        
        self._load_samples(years, max_samples)
        
    def _load_samples(self, years, max_samples):
        for year_folder in years:
            year_path = self.root_dir / year_folder
            if not year_path.exists():
                continue
                
            nested_path = year_path / year_folder
            if nested_path.exists():
                year_path = nested_path
                
            for cyclone_folder in year_path.iterdir():
                if not cyclone_folder.is_dir():
                    continue

                storm_name = cyclone_folder.name
                # No storm-level filtering — the `years` argument is the only splitter.
                timestamps = sorted([t.name for t in cyclone_folder.iterdir() if t.is_dir()])
                
                for i in range(len(timestamps) - 1):
                    current_ts = timestamps[i]
                    next_ts = timestamps[i + 1]
                    
                    current_gridsat = cyclone_folder / current_ts / self.gridsat_type
                    current_era5 = cyclone_folder / current_ts / self.era5_type
                    next_gridsat = cyclone_folder / next_ts / self.gridsat_type
                    next_era5 = cyclone_folder / next_ts / self.era5_type
                    
                    if current_gridsat.exists() and current_era5.exists() and next_gridsat.exists() and next_era5.exists():
                        self.samples.append({
                            'current_gridsat': str(current_gridsat),
                            'current_era5': str(current_era5),
                            'next_gridsat': str(next_gridsat),
                            'next_era5': str(next_era5),
                            'timestamp': current_ts,
                            'storm_name': storm_name,
                            'year': year_folder
                        })
                        
                        if max_samples and len(self.samples) >= max_samples:
                            return
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        current_gridsat = np.load(sample_info['current_gridsat']).astype(np.float32)
        current_gridsat = np.nan_to_num(current_gridsat, nan=0.0)
        
        current_era5 = np.load(sample_info['current_era5']).astype(np.float32)
        current_era5 = np.nan_to_num(current_era5, nan=0.0)
        
        next_gridsat = np.load(sample_info['next_gridsat']).astype(np.float32)
        next_gridsat = np.nan_to_num(next_gridsat, nan=0.0)
        
        next_era5 = np.load(sample_info['next_era5']).astype(np.float32)
        next_era5 = np.nan_to_num(next_era5, nan=0.0)
        
        if current_gridsat.ndim == 3:
            current_gridsat = current_gridsat[0:1]
        elif current_gridsat.ndim == 2:
            current_gridsat = current_gridsat[np.newaxis, ...]
        
        if next_gridsat.ndim == 3:
            next_gridsat = next_gridsat[0:1]
        elif next_gridsat.ndim == 2:
            next_gridsat = next_gridsat[np.newaxis, ...]
        
        if current_era5.ndim == 3:
            current_era5 = current_era5[:4]
        elif current_era5.ndim == 2:
            current_era5 = current_era5[np.newaxis, ...]
        
        if next_era5.ndim == 3:
            next_era5 = next_era5[:4]
        elif next_era5.ndim == 2:
            next_era5 = next_era5[np.newaxis, ...]
        
        current_gridsat = torch.from_numpy(current_gridsat)
        current_era5 = torch.from_numpy(current_era5)
        next_gridsat = torch.from_numpy(next_gridsat)
        next_era5 = torch.from_numpy(next_era5)
        
        target_size = (self.img_size, self.img_size)
        
        current_gridsat = F.interpolate(
            current_gridsat.unsqueeze(0), size=target_size, 
            mode='bilinear', align_corners=False
        ).squeeze(0)
        
        current_era5 = F.interpolate(
            current_era5.unsqueeze(0), size=target_size, 
            mode='bilinear', align_corners=False
        ).squeeze(0)
        
        next_gridsat = F.interpolate(
            next_gridsat.unsqueeze(0), size=target_size, 
            mode='bilinear', align_corners=False
        ).squeeze(0)
        
        next_era5 = F.interpolate(
            next_era5.unsqueeze(0), size=target_size, 
            mode='bilinear', align_corners=False
        ).squeeze(0)
        
        if self.use_global_norm:
            current_gridsat = (current_gridsat - self.gs_mean) / self.gs_std
            next_gridsat    = (next_gridsat - self.gs_mean) / self.gs_std
            current_era5    = (current_era5 - self.era5_mean) / self.era5_std
            next_era5       = (next_era5 - self.era5_mean) / self.era5_std

        if self.mode == 'train' and random.random() > 0.5:
            current_gridsat = torch.flip(current_gridsat, [-1])
            current_era5    = torch.flip(current_era5, [-1])
            next_gridsat    = torch.flip(next_gridsat, [-1])
            next_era5       = torch.flip(next_era5, [-1])

        return {
            'gridsat': current_gridsat,
            'era5': current_era5,
            'target': next_gridsat,
            'target_era5': next_era5,
            'timestamp': sample_info['timestamp'],
            'storm_name': sample_info['storm_name'],
            'year': sample_info['year']
        }


def get_all_storms(root_dir, years):
    root_dir = Path(root_dir)
    all_storms = []
    
    for year_folder in years:
        year_path = root_dir / year_folder
        if not year_path.exists():
            continue
        
        nested_path = year_path / year_folder
        if nested_path.exists():
            year_path = nested_path
        
        for cyclone_folder in year_path.iterdir():
            if cyclone_folder.is_dir():
                all_storms.append(cyclone_folder.name)
    
    return all_storms


def get_dataloaders(root_dir,
                    train_years, val_years, test_years,
                    batch_size=4, num_workers=2,
                    gridsat_type='GRIDSAT_data.npy',
                    era5_type='ERA5_data.npy',
                    max_samples=None,
                    img_size=256):
    """Year-based split. No random storm selection — fully deterministic.
    train_years / val_years / test_years are disjoint lists of year strings
    (e.g. ['2012','2013',...], ['2021'], ['2022'])."""

    train_dataset = CycloneDataset(
        root_dir=root_dir, years=train_years,
        gridsat_type=gridsat_type, era5_type=era5_type,
        max_samples=max_samples, mode='train', img_size=img_size,
    )
    val_dataset = CycloneDataset(
        root_dir=root_dir, years=val_years,
        gridsat_type=gridsat_type, era5_type=era5_type,
        mode='val', img_size=img_size,
    )
    test_dataset = CycloneDataset(
        root_dir=root_dir, years=test_years,
        gridsat_type=gridsat_type, era5_type=era5_type,
        mode='test', img_size=img_size,
    )

    print(f"Train years {train_years}: {len(train_dataset)} (t, t+1) pairs")
    print(f"Val   years {val_years}:  {len(val_dataset)} (t, t+1) pairs")
    print(f"Test  years {test_years}: {len(test_dataset)} (t, t+1) pairs")

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              **loader_kwargs)

    return train_loader, val_loader, test_loader


# ============================================================================
# VAE ARCHITECTURE
# ============================================================================

class VAEEncoder(nn.Module):
    """Encodes images to latent space (mu, logvar)"""
    def __init__(self, in_channels=1, latent_dim=4, base_channels=64):
        super().__init__()
        # 256 -> 128 -> 64 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        
        self.fc_mu = nn.Conv2d(base_channels * 2, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(base_channels * 2, latent_dim, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=4, out_channels=1, base_channels=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, base_channels * 2, 1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            # 64 -> 128 (Upsample + Conv to avoid checkerboard artifacts)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            # 128 -> 256 (Upsample + Conv to avoid checkerboard artifacts)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
        )
        
    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    """Complete VAE for Stage 1 training"""
    def __init__(self, in_channels=1, latent_dim=4, base_channels=64):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, base_channels)
        self.decoder = VAEDecoder(latent_dim, in_channels, base_channels)
        self.latent_dim = latent_dim
        
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        return self.encoder.reparameterize(mu, logvar)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ============================================================================
# CONDITION ENCODING (for UNet conditioning)
# ============================================================================

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        pos_encoding = self._get_2d_positional_encoding(height, width, channels)
        self.register_buffer('pos_encoding', pos_encoding)
    
    def _get_2d_positional_encoding(self, h, w, d_model):
        pe = torch.zeros(d_model, h, w, dtype=torch.float32)
        y_pos = torch.arange(0, h, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(0, w, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(np.log(10000.0) / d_model))
        
        for i in range(0, d_model, 4):
            idx = i // 2
            if idx < len(div_term):
                pe[i] = torch.sin(y_pos * div_term[idx])
                if i + 1 < d_model:
                    pe[i + 1] = torch.cos(y_pos * div_term[idx])
                if i + 2 < d_model and idx < len(div_term):
                    pe[i + 2] = torch.sin(x_pos * div_term[idx])
                if i + 3 < d_model:
                    pe[i + 3] = torch.cos(x_pos * div_term[idx])
        return pe
    
    def forward(self, x):
        _, _, h, w = x.shape
        if h != self.height or w != self.width:
            pos = F.interpolate(self.pos_encoding.unsqueeze(0), size=(h, w), 
                               mode='bilinear', align_corners=False).squeeze(0)
            return x + pos.unsqueeze(0)
        return x + self.pos_encoding.unsqueeze(0)


class TemporalEncoding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.year_embed = nn.Embedding(50, embed_dim // 4)
        self.month_embed = nn.Embedding(12, embed_dim // 4)
        self.day_embed = nn.Embedding(31, embed_dim // 4)
        self.hour_embed = nn.Embedding(24, embed_dim // 4)
        
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def parse_timestamp(self, timestamp_str):
        timestamp_str = timestamp_str.split('.')[0]
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H_%M_%S")
        return (dt.year - 2000, dt.month - 1, dt.day - 1, dt.hour)
    
    def forward(self, timestamps):
        if isinstance(timestamps, (list, tuple)) and isinstance(timestamps[0], str):
            parsed = [self.parse_timestamp(ts) for ts in timestamps]
            years = torch.tensor([p[0] for p in parsed], dtype=torch.long)
            months = torch.tensor([p[1] for p in parsed], dtype=torch.long)
            days = torch.tensor([p[2] for p in parsed], dtype=torch.long)
            hours = torch.tensor([p[3] for p in parsed], dtype=torch.long)
        else:
            years, months, days, hours = timestamps

        device = self.year_embed.weight.device
        years = years.to(device)
        months = months.to(device)
        days = days.to(device)
        hours = hours.to(device)
        
        temporal_emb = torch.cat([
            self.year_embed(years),
            self.month_embed(months),
            self.day_embed(days),
            self.hour_embed(hours)
        ], dim=-1)
        
        return self.projection(temporal_emb)

class ConditionEncoder(nn.Module):
    """Encodes GRIDSAT + ERA5 + temporal info + (optional) coordinates for UNet
    conditioning. Mirrors the multi_channel_diffusion.ConditionEncoder so that
    flow scripts importing from here get the same API (G1: coord conditioning)."""
    def __init__(self, gridsat_channels=1, era5_channels=4, img_size=256,
                 embed_dim=128, output_channels=64):
        super().__init__()
        self.output_channels = output_channels
        emb_dim = embed_dim * 2
        self.spatial_pos_gridsat = SpatialPositionalEncoding(gridsat_channels, img_size, img_size)
        self.spatial_pos_era5 = SpatialPositionalEncoding(era5_channels, img_size, img_size)
        self.temporal_encoder = TemporalEncoding(embed_dim)
        # Coordinate projection: (lat, lon) → embed_dim — additive into temporal_emb
        self.coord_proj = nn.Sequential(
            nn.Linear(2, embed_dim // 2), nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(embed_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim))
        total_input_channels = gridsat_channels + era5_channels
        self.image_encoder = nn.Sequential(
            nn.Conv2d(total_input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1))
        self.context_to_spatial = nn.Sequential(
            nn.Linear(emb_dim, output_channels), nn.SiLU())

    def forward(self, gridsat, era5, timestamps, coords=None):
        gridsat_pos = self.spatial_pos_gridsat(gridsat)
        era5_pos = self.spatial_pos_era5(era5)
        combined = torch.cat([gridsat_pos, era5_pos], dim=1)
        image_features = self.image_encoder(combined)
        temporal_emb = self.temporal_encoder(timestamps)
        if coords is not None:
            coord_emb = self.coord_proj(coords)         # (B, embed_dim)
            temporal_emb = temporal_emb + coord_emb
        context_emb = self.context_projection(temporal_emb)
        context_spatial = self.context_to_spatial(context_emb).unsqueeze(-1).unsqueeze(-1)
        return image_features + context_spatial, context_emb


# ============================================================================
# NOISE SCHEDULER
# ============================================================================

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end, device='cuda'):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), 
                                              self.alpha_cumprod[:-1]])
    
    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self
        
    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]
        device = original.device
        
        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)[t].reshape(batch_size)
        
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
            
        return sqrt_alpha_cumprod * original + sqrt_one_minus_alpha_cumprod * noise
    
    def sample_prev_timestep_ddim(self, xt, noise_pred, t, t_prev=None, eta=0.0, clamp_range=None):
        """DDIM sampling step.
        
        Args:
            clamp_range: (min, max) tuple for x0_pred clamping. If None, uses [-3, 3].
        """
        device = xt.device
        
        if t_prev is None:
            t_prev = t - 1
        
        alpha_cumprod_t = self.alpha_cumprod.to(device)[t]
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod.to(device)[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod.to(device)[t]
        
        if t_prev >= 0:
            alpha_cumprod_prev = self.alpha_cumprod.to(device)[t_prev]
        else:
            alpha_cumprod_prev = torch.tensor(1.0, device=device)
        
        x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t
        if clamp_range is not None:
            x0_pred = torch.clamp(x0_pred, clamp_range[0], clamp_range[1])
        else:
            x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
        
        sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)
        
        if eta > 0 and (1 - alpha_cumprod_t) > 1e-8:
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)) * \
                      torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_prev)
        else:
            sigma_t = torch.tensor(0.0, device=device)
        
        pred_dir = torch.sqrt(torch.clamp(1 - alpha_cumprod_prev - sigma_t**2, min=0.0)) * noise_pred
        x_prev = sqrt_alpha_cumprod_prev * x0_pred + pred_dir
        
        if eta > 0 and t > 0:
            noise = torch.randn_like(xt)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev


class CosineNoiseScheduler(LinearNoiseScheduler):
    """Cosine noise schedule (Nichol & Dhariwal 2021) — better for latent diffusion.
    
    Inherits add_noise, sample_prev_timestep_ddim, and to() from LinearNoiseScheduler.
    Only overrides __init__ to compute cosine-based betas instead of linear.
    """
    def __init__(self, num_timesteps, s=0.008, device='cuda'):
        # Skip parent __init__ — we compute betas via cosine schedule instead
        self.num_timesteps = num_timesteps
        self.device = device
        
        steps = torch.linspace(0, num_timesteps, num_timesteps + 1,
                               dtype=torch.float64, device=device)
        f = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_cumprod_full = f / f[0]
        
        betas = 1.0 - (alpha_cumprod_full[1:] / alpha_cumprod_full[:-1])
        self.betas = torch.clamp(betas, min=1e-5, max=0.999).float().to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device),
                                              self.alpha_cumprod[:-1]])


# ============================================================================
# UNET FOR DIFFUSION
# ============================================================================

def get_time_embeddings(time_steps, t_emb_dim):
    factor = 10000 ** (torch.arange(0, t_emb_dim // 2, dtype=torch.float32, device=time_steps.device) / (t_emb_dim // 2))
    t_emb = time_steps[:, None].float() / factor[None, :]
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class SelfAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x).view(b, c, h * w).permute(0, 2, 1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + attn_out.permute(0, 2, 1).view(b, c, h, w)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample=True,
                 use_attention=False, num_heads=4):
        super().__init__()
        self.res1 = ResnetBlock(in_channels, out_channels, t_emb_dim)
        self.res2 = ResnetBlock(out_channels, out_channels, t_emb_dim)
        self.attn = SelfAttention2D(out_channels, num_heads) if use_attention else nn.Identity()
        self.down = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if down_sample else nn.Identity()
        self.use_checkpoint = True

    def _forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip

    def forward(self, x, t_emb):
        if self.use_checkpoint and self.training:
            def run(x, t_emb):
                return self._forward(x, t_emb)
            x, skip = torch.utils.checkpoint.checkpoint(run, x, t_emb, use_reentrant=False)
            return x, skip
        return self._forward(x, t_emb)


class MidBlock(nn.Module):
    def __init__(self, channels, t_emb_dim, num_heads=4):
        super().__init__()
        self.res1 = ResnetBlock(channels, channels, t_emb_dim)
        self.attn = SelfAttention2D(channels, num_heads)
        self.res2 = ResnetBlock(channels, channels, t_emb_dim)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, t_emb_dim,
                 up_sample=True, use_attention=False, num_heads=4):
        super().__init__()
        self.res1 = ResnetBlock(in_channels + skip_channels, out_channels, t_emb_dim)
        self.res2 = ResnetBlock(out_channels, out_channels, t_emb_dim)
        self.attn = SelfAttention2D(out_channels, num_heads) if use_attention else nn.Identity()
        # Use Upsample + Conv instead of ConvTranspose2d to avoid checkerboard artifacts
        if up_sample:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1)
            )
        else:
            self.up = nn.Identity()

    def forward(self, x, t_emb, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        return x


class DiffusionUNet(nn.Module):

    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 channel_mults=(1, 2, 4), t_emb_dim=128, num_heads=4,
                 gridsat_channels=1, era5_channels=4, img_size=64,
                 condition_embed_dim=128, condition_img_size=256):
        super().__init__()
        n_levels = len(channel_mults)
        ch_list = [base_channels * m for m in channel_mults]
        emb_dim = t_emb_dim * 4
        cond_emb_dim = condition_embed_dim * 2

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

        # Encoder 
        self.encoders = nn.ModuleList()
        for i in range(n_levels):
            in_ch = ch_list[i - 1] if i > 0 else ch_list[0]
            out_ch = ch_list[i]
            is_last = (i == n_levels - 1)
            self.encoders.append(DownBlock(
                in_ch, out_ch, emb_dim, down_sample=not is_last,
                use_attention=(i == n_levels - 2), num_heads=num_heads))

        # Bottleneck
        self.mid_block = MidBlock(ch_list[-1], emb_dim, num_heads=num_heads)

        # Dynamic decoder configs — supports arbitrary number of levels
        decoder_cfgs = []
        for i in range(n_levels):
            if i == 0:
                # First decoder level matches last encoder (no upsampling)
                decoder_cfgs.append((ch_list[-1], ch_list[-1], ch_list[-1], False, False))
            else:
                in_ch_d = ch_list[n_levels - i]
                skip_ch_d = ch_list[n_levels - i - 1]
                out_ch_d = ch_list[n_levels - i - 1]
                # Attention at the first upsampling level (deepest)
                use_attn = (i == 1)
                decoder_cfgs.append((in_ch_d, skip_ch_d, out_ch_d, True, use_attn))
        self.decoders = nn.ModuleList([
            UpBlock(in_ch, skip_ch, out_ch, emb_dim, up_sample=up, use_attention=attn, num_heads=num_heads)
            for in_ch, skip_ch, out_ch, up, attn in decoder_cfgs])

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch_list[0]), nn.SiLU(),
            nn.Conv2d(ch_list[0], out_channels, 3, padding=1))

    def forward(self, noisy_latent, gridsat, era5, timestamps, t,
                encoded_condition=None, context_emb=None):
        # Time embedding
        t_emb = get_time_embeddings(t, self.t_proj[0].in_features)
        t_emb = self.t_proj(t_emb)

        if encoded_condition is None or context_emb is None:
            encoded_condition, context_emb = self.condition_encoder(gridsat, era5, timestamps)
        ctx_emb = self.context_proj(context_emb)
        comb = t_emb + ctx_emb

        x = torch.cat([noisy_latent, encoded_condition], dim=1)
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
# STAGE 1: VAE TRAINER
# ============================================================================

class VAETrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Directories
        self.checkpoint_dir = Path(config['checkpoint_dir']) / 'vae'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(config['output_dir']) / 'vae'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_dir = Path(config['output_dir']) / 'vae' / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        
        # VAE
        self.vae = VAE(
            in_channels=1,
            latent_dim=config.get('latent_dim', 4),
            base_channels=config.get('vae_base_channels', 64)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.vae.parameters(),
            lr=config.get('vae_lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # LR scheduler
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('vae_epochs', 50),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Loss weights
        self.kl_weight_target = config.get('kl_weight', 1e-4)
        self.kl_anneal_start  = config.get('kl_anneal_start', 0)    # epoch to start KL ramp
        self.kl_anneal_end    = config.get('kl_anneal_end', 10)     # epoch where KL reaches full weight
        self.recon_loss_type  = config.get('recon_loss_type', 'l1')
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        print(f"VAE Trainer initialized:")
        print(f"  Latent dim: {config.get('latent_dim', 4)}")
        print(f"  Reconstruction loss: {self.recon_loss_type}")
    
    def _get_kl_weight(self, epoch):
        if epoch <= self.kl_anneal_start:
            return 0.0
        if epoch >= self.kl_anneal_end:
            return self.kl_weight_target
        progress = (epoch - self.kl_anneal_start) / max(1, self.kl_anneal_end - self.kl_anneal_start)
        return self.kl_weight_target * progress
    
    def compute_loss(self, recon, target, mu, logvar, kl_weight=None):
        # Reconstruction loss
        if self.recon_loss_type == 'l1':
            recon_loss = F.l1_loss(recon, target, reduction='mean')
        elif self.recon_loss_type == 'l2':
            recon_loss = F.mse_loss(recon, target, reduction='mean')
        else:
            recon_loss = F.l1_loss(recon, target, reduction='mean')
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        w = kl_weight if kl_weight is not None else self.kl_weight_target
        total_loss = recon_loss + w * kl_loss
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss,
            'kl_scaled': w * kl_loss
        }
    
    def train_epoch(self, dataloader, epoch):
        self.vae.train()
        kl_weight = self._get_kl_weight(epoch)
        
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'VAE Training - Epoch {epoch} [β={kl_weight:.2e}]')
        
        for batch in pbar:
            target = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            recon, mu, logvar = self.vae(target)
            losses = self.compute_loss(recon, target, mu, logvar, kl_weight=kl_weight)
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['recon'].item():.4f}",
                'kl': f"{losses['kl'].item():.4f}",
                'β': f"{kl_weight:.2e}"
            })
        
        # Average losses
        for key in epoch_losses:
            if num_batches > 0:
                epoch_losses[key] /= num_batches
        
        epoch_losses['kl_weight'] = kl_weight  # log the actual weight used
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.vae.eval()
        
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
        metric_accum = {'mse': 0, 'psnr': 0, 'ssim': 0, 'kl_divergence': 0}
        num_batches = 0
        
        for batch in tqdm(dataloader, desc='VAE Validation'):
            target = batch['target'].to(self.device)
            
            recon, mu, logvar = self.vae(target)
            losses = self.compute_loss(recon, target, mu, logvar)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Image quality metrics (MSE, PSNR, SSIM)
            metric_accum['mse'] += MetricsCalculator.mse(recon, target)
            metric_accum['psnr'] += MetricsCalculator.psnr(recon, target)
            metric_accum['ssim'] += MetricsCalculator.ssim(recon, target)
            
            # KL divergence 
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            metric_accum['kl_divergence'] += kl_div.item()
            
            num_batches += 1
        
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
            for key in metric_accum:
                metric_accum[key] /= num_batches
        
        # Merge into one result dict
        result = dict(epoch_losses)
        result.update(metric_accum)
        return result
    
    def check_latent_statistics(self, dataloader, num_batches=50):
        self.vae.eval()
        
        all_mu = []
        all_logvar = []
        all_z = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            target = batch['target'].to(self.device)
            mu, logvar = self.vae.encode(target)
            z = self.vae.reparameterize(mu, logvar)
            
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            all_z.append(z.cpu())
        
        mu = torch.cat(all_mu, dim=0)
        logvar = torch.cat(all_logvar, dim=0)
        z = torch.cat(all_z, dim=0)
        
        # Scale factor to normalize latents
        z_std = z.std().item()
        scale_factor = 1.0 / z_std  # This should be ~0.18215 for good VAEs

        return scale_factor
    
    def save_checkpoint(self, epoch, val_loss):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'vae_latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'vae_best.pt')
            print(f"  ✓ New best VAE model saved (val_loss: {val_loss:.4f})")
    
    def load_checkpoint(self):
        """Load checkpoint only if explicitly specified in config."""
        # Check for explicit checkpoint path in config
        ckpt_path = self.config.get('vae_checkpoint_path', None)
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
        else:
            # No checkpoint specified — train from scratch
            print("  No VAE checkpoint specified in config — training from scratch.")
            return 0
        
        if not ckpt_path.exists():
            print(f"  VAE checkpoint not found at: {ckpt_path} — training from scratch.")
            return 0
        
        print(f"  Loading VAE checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in ckpt:
            self.lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        
        resumed_epoch = ckpt['epoch']
        return resumed_epoch
    
    def save_reconstruction_samples(self, dataloader, epoch):
        self.vae.eval()
        
        with torch.no_grad():
            batch = next(iter(dataloader))
            
            batch_size = len(batch['target'])
            num_samples = min(8, batch_size)
            
            target = batch['target'][:num_samples].to(self.device)
            
            recon, mu, logvar = self.vae(target)
            
            target_np = target.cpu().numpy()
            recon_np = recon.cpu().numpy()
            
            fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
           
            if num_samples == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(num_samples):
                axes[0, i].imshow(target_np[i, 0], cmap='viridis')
                axes[0, i].axis('off')
                axes[0, i].set_title('Original')
                
                axes[1, i].imshow(recon_np[i, 0], cmap='viridis')
                axes[1, i].axis('off')
                axes[1, i].set_title('Recon')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'vae_recon_epoch_{epoch}.png', dpi=150)
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
                    target = batch['target'].to(self.device)
                    recon, _, _ = self.vae(target)
                    real_feats.append(fid_calc.get_features(target))
                    fake_feats.append(fid_calc.get_features(recon))
            real_feats = np.concatenate(real_feats, axis=0)
            fake_feats = np.concatenate(fake_feats, axis=0)
            fid_score = fid_calc.calculate_fid(real_feats, fake_feats)

            return fid_score
        except Exception as e:
            print(f"  Warning: FID computation failed: {e}")
            return float('nan')

    def train(self, train_loader, val_loader):
        num_epochs = self.config.get('vae_epochs', 50)
        
        print("STAGE 1: VAE TRAINING")
        start_epoch = self.load_checkpoint()
        
        if start_epoch >= num_epochs:
            scale_factor = self.check_latent_statistics(val_loader)
            return scale_factor
        
        remaining = num_epochs - start_epoch
        print(f"Training VAE for {remaining} remaining epochs ({start_epoch}/{num_epochs} done)")
    
        fid_every = self.config.get('fid_every', 5)
        
        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch + 1
            kl_weight = self._get_kl_weight(current_epoch)
            
            # Train
            train_losses = self.train_epoch(train_loader, current_epoch)
            self.train_losses.append(train_losses['total'])
            
            print(f"  Train - Loss: {train_losses['total']:.4f} | "
                  f"Recon: {train_losses['recon']:.4f} | KL: {train_losses['kl']:.4f}")
            
            # Validate 
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['total'])
            
            # LR step
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_scheduler.step()
            
            # Periodic FID every fid_every epochs
            epoch_fid = float('nan')
            if current_epoch % fid_every == 0 or current_epoch == num_epochs:
                epoch_fid = self._compute_fid(val_loader)
            
            # Collect per-epoch metrics into metrics_history
            self.metrics_history.append({
                'epoch': current_epoch,
                'train_loss': train_losses['total'],
                'train_recon_loss': train_losses['recon'],
                'train_kl_loss': train_losses['kl'],
                'kl_weight': kl_weight,
                'val_loss': val_metrics['total'],
                'val_recon_loss': val_metrics['recon'],
                'val_kl_loss': val_metrics['kl'],
                'val_mse': val_metrics['mse'],
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim'],
                'val_kl_divergence': val_metrics['kl_divergence'],
                'learning_rate': current_lr,
                'fid': epoch_fid,
            })
            
            # Save checkpoint
            self.save_checkpoint(current_epoch, val_metrics['total'])
            
            # Save samples every 10 epochs
            if current_epoch % 10 == 0:
                self.save_reconstruction_samples(val_loader, current_epoch)
        
        scale_factor = self.check_latent_statistics(val_loader)

        if num_epochs % fid_every != 0:
            final_fid = self._compute_fid(val_loader)
            if self.metrics_history:
                self.metrics_history[-1]['fid'] = final_fid
        else:
            final_fid = self.metrics_history[-1]['fid'] if self.metrics_history else float('nan')
        
        # Save final model with scale factor
        final_checkpoint = {
            'vae_state_dict': self.vae.state_dict(),
            'config': self.config,
            'latent_scale_factor': scale_factor,
            'best_val_loss': self.best_val_loss,
            'fid_score': final_fid,
        }
        torch.save(final_checkpoint, self.checkpoint_dir / 'vae_final.pt')

        # Save final metrics CSV
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(self.metrics_dir / 'vae_final_metrics.csv', index=False)
        
        # Save config as JSON
        config_save = {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()}
        try:
            with open(self.output_dir / 'vae_config.json', 'w') as f:
                json.dump(config_save, f, indent=4, default=str)
        except Exception as e:
            print(f"  Warning: Could not save config JSON: {e}")
        
        # Plot training curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs_x = [m['epoch'] for m in self.metrics_history]
        
        # Loss curves
        axes[0, 0].plot(epochs_x, [m['train_loss'] for m in self.metrics_history], label='Train Loss')
        axes[0, 0].plot(epochs_x, [m['val_loss'] for m in self.metrics_history], label='Val Loss')
        axes[0, 0].set_title('Total Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
        
        # Recon + KL loss
        axes[0, 1].plot(epochs_x, [m['train_recon_loss'] for m in self.metrics_history], label='Train Recon')
        axes[0, 1].plot(epochs_x, [m['val_recon_loss'] for m in self.metrics_history], label='Val Recon')
        axes[0, 1].set_title('Reconstruction Loss'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel('Epoch')
        
        axes[0, 2].plot(epochs_x, [m['val_kl_divergence'] for m in self.metrics_history], label='Val KL-div', color='orange')
        axes[0, 2].set_title('KL Divergence (Val)'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlabel('Epoch')
        
        # Image quality metrics
        axes[1, 0].plot(epochs_x, [m['val_mse'] for m in self.metrics_history], color='red')
        axes[1, 0].set_title('Val MSE'); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_xlabel('Epoch')
        
        axes[1, 1].plot(epochs_x, [m['val_psnr'] for m in self.metrics_history], color='green')
        axes[1, 1].set_title('Val PSNR (dB)'); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].set_xlabel('Epoch')
        
        axes[1, 2].plot(epochs_x, [m['val_ssim'] for m in self.metrics_history], color='purple')
        axes[1, 2].set_title('Val SSIM'); axes[1, 2].grid(True, alpha=0.3); axes[1, 2].set_xlabel('Epoch')
        
        plt.suptitle('VAE Training Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'vae_training_curves.png', dpi=150)
        plt.close()
        
        return scale_factor


# ============================================================================
# STAGE 2: DIFFUSION TRAINER
# ============================================================================

class DiffusionTrainer:

    def __init__(self, config, vae_checkpoint_path, diffusion_checkpoint_path=None):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Directories
        self.checkpoint_dir = Path(config['checkpoint_dir']) / 'diffusion'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(config['output_dir']) / 'diffusion'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LOAD FREEZE VAE
        # Load VAE: use explicit checkpoint path or default to vae_best.pt
        if vae_checkpoint_path is None:
            vae_checkpoint_path = str(Path(config['checkpoint_dir']) / 'vae' / 'vae_best.pt')
        print(f"\nLoading pretrained VAE from: {vae_checkpoint_path}")
        vae_checkpoint = torch.load(vae_checkpoint_path, map_location=self.device)
        
        self.vae = VAE(
            in_channels=1,
            latent_dim=config.get('latent_dim', 4),
            base_channels=config.get('vae_base_channels', 64)
        ).to(self.device)
        
        self.vae.load_state_dict(vae_checkpoint['vae_state_dict'])
        
        self.latent_scale_factor = vae_checkpoint.get('latent_scale_factor', 0.18215)
        
        # Dynamic clamp range — will be computed from actual latent statistics
        # before training starts. Default to [-5, 5] as fallback.
        self.latent_clamp_range = (-5.0, 5.0)
        
        # FREEZE VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Verify VAE is frozen
        vae_grad_params = sum(p.requires_grad for p in self.vae.parameters())
        assert vae_grad_params == 0, "VAE should have no trainable parameters!"
       
        # =====================================================================
        # INITIALIZE UNET
        # =====================================================================
        latent_size = config['img_size'] // 4  
        
        self.unet = DiffusionUNet(
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
            condition_img_size=config['img_size']
        ).to(self.device)
        
        # Condition downsampler (full res -> latent res)
        self.condition_downsample = nn.Sequential(
            nn.Conv2d(config['base_channels'], config['base_channels'], 3, 2, 1),
            nn.GroupNorm(8, config['base_channels']),
            nn.SiLU(),
            nn.Conv2d(config['base_channels'], config['base_channels'], 3, 2, 1),
            nn.GroupNorm(8, config['base_channels']),
            nn.SiLU(),
        ).to(self.device)
        
        # Noise scheduler (cosine schedule is better for latent diffusion)
        noise_schedule = config.get('noise_schedule', 'cosine')
        if noise_schedule == 'cosine':
            self.scheduler = CosineNoiseScheduler(
                num_timesteps=config['num_timesteps'],
                device=self.device
            )
            print(f"  Using cosine noise schedule")
        else:
            self.scheduler = LinearNoiseScheduler(
                num_timesteps=config['num_timesteps'],
                beta_start=config.get('beta_start', 1e-4),
                beta_end=config.get('beta_end', 0.02),
                device=self.device
            )
            print(f"  Using linear noise schedule")
        
        # OPTIMIZER
        self.optimizer = AdamW(
            list(self.unet.parameters()) + list(self.condition_downsample.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('diffusion_epochs', 200),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 1
        
        self.metrics_dir = Path(config['output_dir']) / 'diffusion' / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        # MEMORY OPTIMIZATIONS
        self.use_amp = config.get('use_amp', True)
        # Conservative init_scale (1024 vs default 65536) prevents float16 overflow
        # with the larger 128-channel, 4-level UNet
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1024, growth_interval=2000) if self.use_amp else None
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # LR warmup: ramp from base_lr/10 to base_lr over warmup_epochs
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.base_lr = config['learning_rate']
        
        # EMA for stable evaluation
        self.ema = EMA(self.unet, decay=config.get('ema_decay', 0.9999))
        

        # Only load diffusion checkpoint if explicitly provided
        if diffusion_checkpoint_path is not None and Path(diffusion_checkpoint_path).exists():
            self._load_diffusion_checkpoint(diffusion_checkpoint_path)
        elif diffusion_checkpoint_path is not None:
            print(f"  ⚠ Diffusion checkpoint not found at: {diffusion_checkpoint_path} — starting from scratch")
        else:
            print(f"  No diffusion checkpoint specified — starting from scratch")
        
        print(f"\nDiffusion Trainer initialized:")
        print(f"  UNet params: {sum(p.numel() for p in self.unet.parameters()):,}")
        print(f"  Mixed Precision (AMP): {self.use_amp}")
        print(f"  EMA decay: {config.get('ema_decay', 0.9999)}")
        print(f"  Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"  Resuming from epoch: {self.start_epoch}")
    
    def _load_diffusion_checkpoint(self, checkpoint_path):
        print(f"\nLoading diffusion checkpoint from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            
            self.unet.load_state_dict(ckpt['unet_state_dict'])
            self.condition_downsample.load_state_dict(ckpt['condition_downsample_state_dict'])
            
            # Restore optimizer state
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
            # Restore AMP scaler if it was saved
            if self.scaler is not None and 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            
            # Restore training metadata
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            resumed_epoch = ckpt.get('epoch', 0)
            self.start_epoch = resumed_epoch + 1
            
            # Restore LR scheduler state
            if 'lr_scheduler_state_dict' in ckpt:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            else: 
                for _ in range(resumed_epoch):
                    self.lr_scheduler.step()
                print(f"  ⚠ LR scheduler fast-forwarded {resumed_epoch} steps (no saved state)")
            if 'ema_state_dict' in ckpt:
                self.ema.load_state_dict(ckpt['ema_state_dict'])
            
            # Restore metrics history for continuous training curves
            if 'train_losses' in ckpt:
                self.train_losses = ckpt['train_losses']
                self.val_losses = ckpt['val_losses']
                print(f"  ✓ Training history restored ({len(self.train_losses)} epochs)")
            if 'metrics_history' in ckpt:
                self.metrics_history = ckpt['metrics_history']
                print(f"  ✓ Metrics history restored ({len(self.metrics_history)} entries)")

        except Exception as e:
            print(f"  ✗ Failed to load diffusion checkpoint: {e}")
            print(f"  → Starting from scratch")
            self.start_epoch = 1
    
    @torch.no_grad()
    def encode_to_latent(self, images):
        mu, logvar = self.vae.encode(images)
        z = self.vae.reparameterize(mu, logvar)
        z_scaled = z * self.latent_scale_factor
        return z_scaled
    
    @torch.no_grad()
    def decode_from_latent(self, z_scaled):
        z = z_scaled / self.latent_scale_factor
        return self.vae.decode(z)
    
    @torch.no_grad()
    def compute_latent_clamp_range(self, dataloader, num_batches=50):
        """Compute the actual scaled-latent range from data for dynamic DDIM clamping."""
        self.vae.eval()
        all_mins, all_maxs = [], []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            target = batch['target'].to(self.device)
            z_scaled = self.encode_to_latent(target)
            all_mins.append(z_scaled.min().item())
            all_maxs.append(z_scaled.max().item())
        
        z_min = min(all_mins)
        z_max = max(all_maxs)
        # Add 20% margin beyond observed range
        margin = (z_max - z_min) * 0.1
        clamp_min = z_min - margin
        clamp_max = z_max + margin
        
        self.latent_clamp_range = (clamp_min, clamp_max)
        print(f"  ✓ Latent clamp range computed: [{clamp_min:.2f}, {clamp_max:.2f}]")
        print(f"    (raw latent range: [{z_min:.3f}, {z_max:.3f}])")
        return self.latent_clamp_range
    
    def train_epoch(self, dataloader, epoch):
        self.unet.train()
        self.condition_downsample.train()
        self.vae.eval()
        
        epoch_loss = 0
        num_batches = 0
        batch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Diffusion Training - Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            target = batch['target'].to(self.device)
            timestamps = batch['timestamp']
            batch_size = target.shape[0]
            
            with torch.no_grad():
                z_scaled = self.encode_to_latent(target)
            
            # Verify VAE has no gradients
            if num_batches == 0 and epoch == 1:
                for name, param in self.vae.named_parameters():
                    assert not param.requires_grad, f"VAE param {name} should not require grad!"
                    assert param.grad is None, f"VAE param {name} should have no gradient!"
            
            # ENCODE CONDITIONS
            encoded_cond, context_emb = self.unet.condition_encoder(
                gridsat, era5, timestamps
            )
            encoded_cond = self.condition_downsample(encoded_cond)
            
            t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)
            noise = torch.randn_like(z_scaled)
            noisy_latent = self.scheduler.add_noise(z_scaled, noise, t)
            
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                noise_pred = self.unet(
                    noisy_latent, gridsat, era5, timestamps, t,
                    encoded_condition=encoded_cond, context_emb=context_emb
                )

                # LOSS: MSE(predicted_noise, true_noise)
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / self.gradient_accumulation_steps
            
            # ===== NaN PROTECTION =====
            if not torch.isfinite(loss):
                print(f"  ⚠ NaN/Inf loss at batch {batch_idx}, skipping")
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.update()
                continue
            
            # Backward with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Verify VAE still has no gradients after backward
            if num_batches == 0:
                for name, param in self.vae.named_parameters():
                    assert param.grad is None, f"VAE param {name} received gradient!"
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.unet.parameters()) + list(self.condition_downsample.parameters()),
                        max_norm=1.0
                    )
                    # Skip step if gradients are NaN after unscaling
                    if not torch.isfinite(grad_norm):
                        print(f"  ⚠ NaN gradients at batch {batch_idx}, skipping step")
                        self.optimizer.zero_grad()
                        self.scaler.update()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.ema.update(self.unet)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.unet.parameters()) + list(self.condition_downsample.parameters()),
                        max_norm=1.0
                    )
                    self.optimizer.step()
                    self.ema.update(self.unet)
            
            # Track (scale back for logging)
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            batch_losses.append(loss.item() * self.gradient_accumulation_steps)
            num_batches += 1
            
            # Clear cache periodically to prevent fragmentation
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            pbar.set_postfix({'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}"})
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.unet.eval()
        self.vae.eval()
        
        epoch_loss = 0
        num_batches = 0
        
        # Fixed seed for reproducible validation (random timesteps match training distribution)
        rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        for batch in tqdm(dataloader, desc='Validation'):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            target = batch['target'].to(self.device)
            timestamps = batch['timestamp']
            batch_size = target.shape[0]
            
            # Encode to latent
            z_scaled = self.encode_to_latent(target)
            
            # Encode conditions
            encoded_cond, context_emb = self.unet.condition_encoder(
                gridsat, era5, timestamps
            )
            encoded_cond = self.condition_downsample(encoded_cond)
            
            # Random timesteps matching training distribution (unbiased validation)
            noise = torch.randn_like(z_scaled)
            t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)
            noisy_latent = self.scheduler.add_noise(z_scaled, noise, t)
            
            noise_pred = self.unet(
                noisy_latent, gridsat, era5, timestamps, t,
                encoded_condition=encoded_cond, context_emb=context_emb
            )
            
            epoch_loss += F.mse_loss(noise_pred, noise).item()
            num_batches += 1
    
        # Restore RNG state so training is unaffected
        torch.random.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        
        avg_val_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"  Validation: avg_loss={avg_val_loss:.6f}, num_batches={num_batches}")
        return avg_val_loss
    
    @torch.no_grad()
    def sample(self, gridsat, era5, timestamps, num_steps=None):
        """Generate samples using DDIM sampling"""
        self.unet.eval()
        self.vae.eval()
        
        batch_size = gridsat.shape[0]
        latent_size = self.config['img_size'] // 4
        latent_dim = self.config.get('latent_dim', 4)
        
        if num_steps is None:
            num_steps = self.scheduler.num_timesteps
        
        # Start from random noise
        z = torch.randn(batch_size, latent_dim, latent_size, latent_size, device=self.device)
        
        # Encode conditions once
        encoded_cond, context_emb = self.unet.condition_encoder(
            gridsat, era5, timestamps
        )
        encoded_cond = self.condition_downsample(encoded_cond)
        
        # DDIM sampling with proper skip steps
        step_size = max(1, self.scheduler.num_timesteps // num_steps)
        timesteps = list(range(0, self.scheduler.num_timesteps, step_size))[::-1]
        
        for i, t_idx in enumerate(timesteps):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            
            noise_pred = self.unet(
                z, gridsat, era5, timestamps, t,
                encoded_condition=encoded_cond, context_emb=context_emb
            )
            
            # Pass the ACTUAL previous timestep in the schedule
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            z = self.scheduler.sample_prev_timestep_ddim(
                z, noise_pred, t_idx, t_prev=t_prev, eta=0.0,
                clamp_range=self.latent_clamp_range
            )
        
        # Decode from latent
        images = self.decode_from_latent(z)
        
        return images
    
    def save_checkpoint(self, epoch, val_loss):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'condition_downsample_state_dict': self.condition_downsample.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'ema_state_dict': self.ema.state_dict(),
            'best_val_loss': self.best_val_loss,
            'latent_scale_factor': self.latent_scale_factor,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'diffusion_latest.pt')
        
        # Save best with EMA weights
        if is_best:
            self.ema.apply(self.unet)
            ema_ckpt = dict(checkpoint)
            ema_ckpt['unet_state_dict'] = self.unet.state_dict()
            torch.save(ema_ckpt, self.checkpoint_dir / 'diffusion_best.pt')
            self.ema.restore(self.unet)
            print(f"  ✓ New best diffusion model saved (val_loss: {val_loss:.4f})")
        
        # Save periodic
        if epoch % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'diffusion_epoch_{epoch}.pt')
    
    def save_samples(self, dataloader, epoch):
        """Save generated samples"""
        self.unet.eval()
        self.vae.eval()
        
        batch = next(iter(dataloader))
        
        # Determine actual number of samples (min of 8 or batch size)
        batch_size = len(batch['gridsat'])
        num_samples = min(8, batch_size)
        
        gridsat = batch['gridsat'][:num_samples].to(self.device)
        era5 = batch['era5'][:num_samples].to(self.device)
        target = batch['target'][:num_samples].to(self.device)
        timestamps = batch['timestamp'][:num_samples]
        
        # Generate samples
        with torch.no_grad():
            generated = self.sample(gridsat, era5, timestamps, num_steps=50)
         
        # Convert to numpy
        gridsat_np = gridsat.cpu().numpy()
        target_np = target.cpu().numpy()
        generated_np = generated.cpu().numpy()
        
        # Plot
        fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))
        
        # Handle case where num_samples == 1 (axes won't be 2D)
        if num_samples == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(num_samples):
            axes[0, i].imshow(gridsat_np[i, 0], cmap='viridis')
            axes[0, i].axis('off')
            axes[0, i].set_title('Input')
            
            axes[1, i].imshow(target_np[i, 0], cmap='viridis')
            axes[1, i].axis('off')
            axes[1, i].set_title('Target')
            
            axes[2, i].imshow(generated_np[i, 0], cmap='viridis')
            axes[2, i].axis('off')
            axes[2, i].set_title('Generated')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'samples_epoch_{epoch}.png', dpi=150)
        plt.close()
    
    def train(self, train_loader, val_loader):
        num_epochs = self.config.get('diffusion_epochs', 100)
        eval_every = 1  # Forced to 1 to evaluate every epoch as per user request
        eval_samples = self.config.get('eval_samples', 50)
        
        print("STAGE 2: DIFFUSION TRAINING")
        print(f"Training UNet for {num_epochs} epochs (starting from epoch {self.start_epoch})")
        print(f"Evaluation every {eval_every} epochs with {eval_samples} samples")
        print(f"Latent scale factor: {self.latent_scale_factor}")
        
        # Compute dynamic clamp range from actual latent statistics
        print("\n  Computing latent clamp range from validation data...")
        self.compute_latent_clamp_range(val_loader, num_batches=50)
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # LR warmup: ramp from base_lr/10 to base_lr over warmup_epochs
            if epoch <= self.warmup_epochs:
                warmup_factor = 0.1 + 0.9 * (epoch / self.warmup_epochs)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.base_lr * warmup_factor
                print(f"  Warmup LR: {self.base_lr * warmup_factor:.2e}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            print(f"  Train Loss: {train_loss:.6f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            print(f"  Val Loss:   {val_loss:.6f}")
            
            # LR step
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  LR: {current_lr:.1e}")
            
            is_best = val_loss < self.best_val_loss
            
            # ===== EVALUATION METRICS =====
            eval_metrics = None
            if epoch % eval_every == 0 or epoch == num_epochs:
                print(f"\n  Running evaluation ({eval_samples} samples)...")
                try:
                    self.ema.apply(self.unet)
                    eval_metrics = evaluate_model(
                        self.unet,
                        val_loader,
                        self.scheduler,
                        self.device,
                        num_samples=eval_samples,
                        vae_decoder=self.vae.decoder,
                        condition_downsample=self.condition_downsample,
                        compute_advanced_metrics=True,
                        latent_scale_factor=self.latent_scale_factor,
                        clamp_range=self.latent_clamp_range
                    )
                    self.ema.restore(self.unet)
                    if eval_metrics:
                        print(f"  Eval Metrics:")
                        print(f"    MAE:  {eval_metrics.get('mae', 0):.4f}")
                        print(f"    MSE:  {eval_metrics.get('mse', 0):.4f}")
                        print(f"    RMSE: {eval_metrics.get('rmse', 0):.4f}")
                        print(f"    PSNR: {eval_metrics.get('psnr', 0):.2f} dB")
                        print(f"    SSIM: {eval_metrics.get('ssim', 0):.4f}")
                        
                        if 'fid' in eval_metrics:
                            print(f"    FID:   {eval_metrics.get('fid', 0):.4f}")
                        if 'is_mean' in eval_metrics:
                            print(f"    IS:    {eval_metrics.get('is_mean', 0):.4f} ± {eval_metrics.get('is_std', 0):.4f}")
                except Exception as e:
                    print(f"  Warning: Evaluation failed: {e}")
                    eval_metrics = None
            
            # ===== CSV METRICS LOGGING =====
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'is_best': is_best,
                'latent_scale_factor': self.latent_scale_factor,
            }
            if eval_metrics:
                for k, v in eval_metrics.items():
                    epoch_metrics[f'eval_{k}'] = v
            
            self.metrics_history.append(epoch_metrics)
            
            # Save CSV after every epoch
            metrics_df = pd.DataFrame(self.metrics_history)
            metrics_df.to_csv(self.metrics_dir / 'diffusion_training_metrics.csv', index=False)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Save samples every 10 epochs (or every 5 in later training)
            if epoch % 10 == 0 or (epoch > num_epochs // 2 and epoch % 5 == 0):
                print(f"\n  Saving sample images...")
                self.ema.apply(self.unet)
                self.save_samples(val_loader, epoch)
                self.ema.restore(self.unet)
        
        # ===== FINAL EVALUATION =====
        print(f"\n{'='*60}")
        print("Running final comprehensive evaluation...")
        try:
            self.ema.apply(self.unet)
            final_metrics = evaluate_model(
                self.unet,
                val_loader,
                self.scheduler,
                self.device,
                num_samples=min(eval_samples * 2, 200),
                vae_decoder=self.vae.decoder,
                condition_downsample=self.condition_downsample,
                compute_advanced_metrics=True,
                latent_scale_factor=self.latent_scale_factor,
                clamp_range=self.latent_clamp_range
            )
            self.ema.restore(self.unet)
            print(f"\nFinal Evaluation Metrics:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Save final metrics as JSON
            with open(self.output_dir / 'final_eval_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=4)
        except Exception as e:
            print(f"Warning: Final evaluation failed: {e}")
            final_metrics = {}
        
        print(f"\n{'='*60}")
        print("DIFFUSION TRAINING COMPLETE")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.checkpoint_dir}")
        print(f"Metrics CSV: {self.metrics_dir / 'diffusion_training_metrics.csv'}")
        print(f"{'='*60}\n")
        
        # Save final metrics CSV
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(self.metrics_dir / 'diffusion_final_metrics.csv', index=False)
        
        # Save config as JSON
        config_save = {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()}
        try:
            with open(self.output_dir / 'diffusion_config.json', 'w') as f:
                json.dump(config_save, f, indent=4, default=str)
        except Exception as e:
            print(f"Warning: Could not save config JSON: {e}")
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        epochs_x = [m['epoch'] for m in self.metrics_history] if self.metrics_history else list(range(1, len(self.train_losses) + 1))
        plt.plot(epochs_x[:len(self.train_losses)], self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(epochs_x[:len(self.val_losses)], self.val_losses, label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Diffusion Training (Noise Prediction)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot evaluation metrics if available
        eval_epochs = [m['epoch'] for m in self.metrics_history if 'eval_psnr' in m]
        eval_psnrs = [m['eval_psnr'] for m in self.metrics_history if 'eval_psnr' in m]
        eval_ssims = [m['eval_ssim'] for m in self.metrics_history if 'eval_ssim' in m]
        
        if eval_epochs:
            plt.subplot(1, 2, 2)
            ax1 = plt.gca()
            ax1.plot(eval_epochs, eval_psnrs, 'b-o', label='PSNR (dB)', markersize=4)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('PSNR (dB)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax1.twinx()
            ax2.plot(eval_epochs, eval_ssims, 'r-s', label='SSIM', markersize=4)
            ax2.set_ylabel('SSIM', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            plt.title('Evaluation Metrics')
            ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'diffusion_training_curves.png', dpi=150)
        plt.close()


# GAN Refinement has been moved to model/refinement_gan.py


# ============================================================================
# EVALUATION AND METRICS
# ============================================================================

from eval_metrics_extended import MetricsCalculator, FID, InceptionScore  # consolidated



def evaluate_model(model, dataloader, scheduler, device, 
                   num_samples=100, vae_decoder=None, condition_downsample=None,
                   compute_advanced_metrics=True, latent_scale_factor=0.18215,
                   clamp_range=None):
    # Save RNG state to prevent eval from corrupting training randomness
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    model.eval()
    if vae_decoder is not None:
        vae_decoder.eval()
    
    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'psnr': [],
        'ssim': [],
        'fid': [],
        'is_mean': [],
        'is_std': []
    }
    
    calc = MetricsCalculator()
    
    # Initialize advanced metrics independently so one failure doesn't block others
    fid_metric = None
    is_metric = None
    all_real_features = []
    all_fake_features = []
    
    if compute_advanced_metrics:
        # --- FID ---
        try:
            fid_metric = FID(device=device)
            print(f"  [Advanced Metrics] FID initialized successfully")
        except Exception as e:
            print(f"  [Advanced Metrics] WARNING: FID initialization failed: {e}")
        # --- Inception Score ---
        try:
            is_metric = InceptionScore(device=device)
            print(f"  [Advanced Metrics] InceptionScore initialized successfully")
        except Exception as e:
            print(f"  [Advanced Metrics] WARNING: InceptionScore initialization failed: {e}")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break
            
            gridsat = batch['gridsat'].to(device)
            era5 = batch['era5'].to(device)
            target = batch['target'].to(device)
            timestamps = batch['timestamp']
            
            with torch.no_grad():
                encoded_cond, context_emb = model.condition_encoder(
                    gridsat, era5, timestamps
                )
                if condition_downsample is not None:
                    encoded_cond = condition_downsample(encoded_cond)
            
            batch_size = target.shape[0]
        
            if vae_decoder is not None:
                latent_spatial_size = target.shape[-1] // 4  
                latent_channels = model.init_conv.in_channels - model.condition_encoder.output_channels
                
                generated = torch.randn(
                    batch_size, latent_channels, 
                    latent_spatial_size, latent_spatial_size, 
                    device=device
                )
            else:
                generated = torch.randn_like(target).to(device)
            
            # DDIM sampling with proper skip steps (50 steps instead of 1000)
            num_sampling_steps = 50
            step_size = max(1, scheduler.num_timesteps // num_sampling_steps)
            sampling_timesteps = list(range(0, scheduler.num_timesteps, step_size))[::-1]
            
            for step_i, t in enumerate(sampling_timesteps):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                noise_pred = model(generated, gridsat, era5, timestamps, t_batch,
                                   encoded_condition=encoded_cond, context_emb=context_emb)
                t_prev = sampling_timesteps[step_i + 1] if step_i + 1 < len(sampling_timesteps) else -1
                generated = scheduler.sample_prev_timestep_ddim(
                    generated, noise_pred, t, t_prev=t_prev, eta=0.0,
                    clamp_range=clamp_range
                )
                
            # Decode latent back to pixel space for metrics
            if vae_decoder is not None:
                generated_unscaled = generated / latent_scale_factor
                generated = vae_decoder(generated_unscaled)
            
            # Calculate basic metrics
            metrics['mae'].append(calc.mae(generated, target))
            metrics['mse'].append(calc.mse(generated, target))
            metrics['rmse'].append(calc.rmse(generated, target))
            metrics['psnr'].append(calc.psnr(generated, target))
            metrics['ssim'].append(calc.ssim(generated, target))
            
            # ============================================================
            # ADVANCED METRICS — each computed independently
            # ============================================================
            if compute_advanced_metrics:
                # --- FID features (per-batch, aggregated later) ---
                if fid_metric is not None:
                    try:
                        real_features = fid_metric.get_features(target)
                        fake_features = fid_metric.get_features(generated)
                        all_real_features.append(real_features)
                        all_fake_features.append(fake_features)
                    except Exception as e:
                        if i == 0:
                            print(f"  [Advanced Metrics] FID feature extraction batch {i} failed: {e}")
                
                # --- Inception Score (per-batch) ---
                if is_metric is not None:
                    try:
                        actual_splits = max(1, min(10, batch_size // 2))  # Ensure at least 1 split
                        is_mean, is_std = is_metric.calculate_inception_score(generated, splits=actual_splits)
                        metrics['is_mean'].append(is_mean)
                        metrics['is_std'].append(is_std)
                    except Exception as e:
                        if i == 0:
                            print(f"  [Advanced Metrics] IS batch {i} failed: {e}")
    
    # Calculate averages
    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    
    # Calculate overall FID score (needs all features aggregated)
    if fid_metric is not None and all_real_features and all_fake_features:
        try:
            all_real_feats = np.concatenate(all_real_features, axis=0)
            all_fake_feats = np.concatenate(all_fake_features, axis=0)
            fid_score = fid_metric.calculate_fid(all_real_feats, all_fake_feats)
            avg_metrics['fid'] = float(fid_score)
            print(f"  [Advanced Metrics] FID computed successfully: {fid_score:.4f}")
        except Exception as e:
            print(f"  [Advanced Metrics] ERROR calculating final FID: {e}")
    
    # Log summary
    print(f"  [Eval Summary] MAE={avg_metrics['mae']:.4f}, MSE={avg_metrics['mse']:.4f}, "
          f"PSNR={avg_metrics['psnr']:.2f}, SSIM={avg_metrics['ssim']:.4f}, "
          f"FID={avg_metrics['fid']:.4f},"
          f"IS={avg_metrics['is_mean']:.4f}")
    
    # Restore RNG state
    torch.random.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    
    return avg_metrics


def save_comparison_images(model, dataloader, scheduler, 
                          device, save_dir, num_samples=8, vae_decoder=None, 
                          condition_downsample=None, latent_scale_factor=0.18215,
                          clamp_range=None):
    model.eval()
    if vae_decoder is not None:
        vae_decoder.eval()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        
        batch_size = len(batch['gridsat'])
        actual_samples = min(num_samples, batch_size)
        
        if actual_samples < num_samples:
            print(f"Warning: Requested {num_samples} samples but batch only contains {batch_size}. Using {actual_samples} samples.")
        
        gridsat = batch['gridsat'][:actual_samples].to(device)
        era5 = batch['era5'][:actual_samples].to(device)
        target = batch['target'][:actual_samples].to(device)
        timestamps = batch['timestamp'][:actual_samples]
        
        with torch.no_grad():
            encoded_cond, context_emb = model.condition_encoder(
                gridsat, era5, timestamps
            )
            if condition_downsample is not None:
                encoded_cond = condition_downsample(encoded_cond)
        
        actual_batch_size = target.shape[0]
        
        if vae_decoder is not None:
            latent_spatial_size = target.shape[-1] // 4  
            latent_channels = model.init_conv.in_channels - model.condition_encoder.output_channels
            
            generated = torch.randn(
                actual_batch_size, latent_channels,
                latent_spatial_size, latent_spatial_size,
                device=device
            )
        else:
            generated = torch.randn_like(target).to(device)
        
        # DDIM sampling with proper skip steps (50 steps instead of 1000)
        num_sampling_steps = 50
        step_size = max(1, scheduler.num_timesteps // num_sampling_steps)
        sampling_timesteps = list(range(0, scheduler.num_timesteps, step_size))[::-1]
        
        print(f"\n===== DDIM Sampling ({len(sampling_timesteps)} steps) =====")
        for step_i, t in enumerate(sampling_timesteps):
            t_batch = torch.full((actual_batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(generated, gridsat, era5, timestamps, t_batch,
                               encoded_condition=encoded_cond, context_emb=context_emb)
            t_prev = sampling_timesteps[step_i + 1] if step_i + 1 < len(sampling_timesteps) else -1
            generated = scheduler.sample_prev_timestep_ddim(
                generated, noise_pred, t, t_prev=t_prev, eta=0.0,
                clamp_range=clamp_range
            )
            
            if step_i == 0 or step_i == len(sampling_timesteps)//4 or step_i == len(sampling_timesteps)//2 or step_i == len(sampling_timesteps)-1:
                print(f"  Step {step_i}/{len(sampling_timesteps)}, t={t}->t_prev={t_prev}: range=[{generated.min().item():.3f}, {generated.max().item():.3f}]")
        
        if vae_decoder is not None:
            print(f"Before VAE decode: generated shape={generated.shape}, range=[{generated.min().item():.3f}, {generated.max().item():.3f}]")
            generated_unscaled = generated / latent_scale_factor
            print(f"After unscaling: range=[{generated_unscaled.min().item():.3f}, {generated_unscaled.max().item():.3f}]")
            generated = vae_decoder(generated_unscaled)
            print(f"After VAE decode: generated shape={generated.shape}, range=[{generated.min().item():.3f}, {generated.max().item():.3f}]")
        
        # Convert to numpy
        gridsat_np = gridsat.cpu().numpy()
        target_np = target.cpu().numpy()
        generated_np = generated.cpu().numpy()

        fig, axes = plt.subplots(actual_samples, 3, figsize=(12, 4 * actual_samples))
        
        if actual_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(actual_samples):
            axes[i, 0].imshow(gridsat_np[i, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 0].set_title(f'Input t\n{timestamps[i]}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target_np[i, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 1].set_title(f'Ground Truth t+1')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(generated_np[i, 0], cmap='gray', vmin=-1, vmax=1)
            axes[i, 2].set_title(f'Predicted t+1')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison images saved to {save_dir / 'comparison.png'}")


def plot_training_curves(train_losses, val_losses, save_path):

    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_path}")


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def train_vae(config):
    """Stage 1: Train VAE"""
    train_loader, val_loader, _ = get_dataloaders(
        root_dir=config['root_dir'],
        train_years=config['train_years'],
        val_years=config['val_years'],
        test_years=config['test_years'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
    )
    
    trainer = VAETrainer(config)
    scale_factor = trainer.train(train_loader, val_loader)
    
    return scale_factor


def train_diffusion(config, vae_checkpoint_path):
    """Stage 2: Train Diffusion with frozen VAE"""
    train_loader, val_loader, _ = get_dataloaders(
        root_dir=config['root_dir'],
        train_years=config['train_years'],
        val_years=config['val_years'],
        test_years=config['test_years'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
    )
    
    trainer = DiffusionTrainer(config, vae_checkpoint_path)
    trainer.train(train_loader, val_loader)


def main():
    config = {
        # Data
        'root_dir': r'E:\thesis_2007039\cyclone dataset',
        'train_years': ['2012','2013','2014','2015','2016','2017','2018','2019','2020'],
        'val_years':   ['2021'],
        'test_years':  ['2022'],
        'batch_size': 4,
        'num_workers': 4,
        'img_size': 256,
        
        # VAE (Stage 1)
        'latent_dim': 4,
        'vae_base_channels': 64,
        'vae_epochs': 50,
        'vae_lr': 1e-4,
        'kl_weight': 1e-4,       
        'kl_anneal_start': 0,    
        'kl_anneal_end': 20,     
        'recon_loss_type': 'l1',
        'fid_every': 5,       
        
        # Diffusion (Stage 2)
        'base_channels': 128,            # was 64 — increased for more capacity
        'channel_mults': (1, 2, 3, 4),   # was (1, 2, 4) — 4 levels for deeper UNet
        'num_heads': 8,                  # was 4
        'embed_dim': 128,
        't_emb_dim': 128,
        'num_timesteps': 1000,
        'noise_schedule': 'cosine',      # cosine schedule for latent diffusion
        'beta_start': 1e-4,              # only used if noise_schedule='linear'
        'beta_end': 0.02,                # only used if noise_schedule='linear'
        'diffusion_epochs': 200,         # was 100
        'learning_rate': 1e-4,           # conservative for larger model + AMP
        'min_lr': 1e-6,                  # was 5e-5 — wider LR range
        'weight_decay': 1e-5,
        'dropout': 0.1,
        'ema_decay': 0.9999,
        
        # Memory optimizations
        'use_amp': True,                 # was False — enables mixed precision
        'gradient_accumulation_steps': 4,
        
        
        # Evaluation
        'eval_every': 5,       
        'eval_samples': 200,    


        # Output
        'checkpoint_dir': './checkpoints',
        'output_dir': './outputs',
        
        # Checkpoint paths (set to None to train from scratch)
        'vae_checkpoint_path': r'E:\thesis_2007039\checkpoints\vae\vae_latest.pt',           # e.g. './checkpoints/vae/vae_latest.pt'
        'diffusion_checkpoint_path': None,     # e.g. './checkpoints/diffusion/diffusion_latest.pt'
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=config['root_dir'],
        train_years=config['train_years'],
        val_years=config['val_years'],
        test_years=config['test_years'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
    )
    
    # STAGE 1: Train VAE
    print("=" * 60)
    print("STAGE 1: TRAINING VAE")
    print("=" * 60)
    
    vae_trainer = VAETrainer(config)
    scale_factor = vae_trainer.train(train_loader, val_loader)
    
    # STAGE 2: Train Diffusion (VAE frozen)
    print("STAGE 2: TRAINING DIFFUSION (VAE FROZEN)")
    
    # Read checkpoint paths from config
    vae_ckpt = config.get('vae_checkpoint_path', None)
    diff_ckpt = config.get('diffusion_checkpoint_path', None)

    diffusion_trainer = DiffusionTrainer(
        config,
        vae_ckpt,
        diffusion_checkpoint_path=diff_ckpt
    )
    diffusion_trainer.train(train_loader, val_loader)
    
if __name__ == '__main__':
    main()
