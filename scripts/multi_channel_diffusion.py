"""
Strategy 1: Multi-Channel Diffusion
====================================
Extends the latent diffusion pipeline to jointly generate both
GRIDSAT (1 ch) and ERA5 (4 ch) as a single 5-channel output.

Key idea:
 - Train a 5-channel VAE  (in=5, out=5, latent=4)
 - Diffusion UNet operates in latent space, conditioned on coords
 - Trajectory accuracy is evaluated via steering flow displacement
   extracted from generated ERA5 U/V wind channels (no explicit head)
 - At inference, decode to 5 channels -> split into GRIDSAT + ERA5

Coordinates are loaded from coordinate_lookup.json (built from IBTrACS)
and used as conditioning signals, NOT encoded through the VAE.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from torchvision.models import inception_v3
import numpy as np
import pandas as pd
import random
import math
from scipy import linalg
import json
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import re


# ============================================================================
# COORDINATE PARSING (IBTrACS SID format)
# ============================================================================

def parse_storm_coordinates(storm_name):
    """Parse IBTrACS SID → (signed_lat, lon_360).
    Format: YYYYDDDHLLLOOO  (e.g. 2000160N21267)
    Returns: (lat_degrees_signed, lon_degrees_0_360)
    """
    try:
        ns_idx = next(i for i, c in enumerate(storm_name) if c in ('N', 'S'))
        hemisphere = storm_name[ns_idx]
        lat = int(storm_name[ns_idx+1:ns_idx+3])
        lon = int(storm_name[ns_idx+3:])
        if hemisphere == 'S':
            lat = -lat
        return float(lat), float(lon)
    except (StopIteration, ValueError):
        return 0.0, 180.0  # fallback: equator, date line


# ---- shared components from existing latent diffusion ----
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

class CycloneDataset(Dataset):
    def __init__(self, root_dir, years, gridsat_type='GRIDSAT_data.npy',
                 era5_type='ERA5_data.npy', transform=None, max_samples=None,
                 mode='train', img_size=256, coord_lookup_path=None):
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

        # Load per-timestep coordinate lookup from IBTrACS
        self.coord_lookup = {}
        self.coord_mean = torch.tensor([0.0, 180.0])   # defaults
        self.coord_std  = torch.tensor([15.0, 100.0])  # defaults
        if coord_lookup_path and Path(coord_lookup_path).exists():
            with open(coord_lookup_path, 'r') as f:
                data = json.load(f)
            self.coord_lookup = data.get('lookup', {})
            stats = data.get('coord_stats', {})
            self.coord_mean = torch.tensor([stats.get('lat_mean', 0.0),
                                            stats.get('lon_mean', 180.0)])
            self.coord_std  = torch.tensor([stats.get('lat_std', 15.0),
                                            stats.get('lon_std', 100.0)])
            print(f"  Loaded {len(self.coord_lookup):,} coordinate entries "  
                  f"(mean=[{self.coord_mean[0]:.1f}, {self.coord_mean[1]:.1f}], "
                  f"std=[{self.coord_std[0]:.1f}, {self.coord_std[1]:.1f}])")
        else:
            print(f"  WARNING: No coordinate lookup — using genesis coords from storm name")
        
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
                            'next_timestamp': next_ts,
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
        
        # Ensure correct dimensions
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
            # Global z-score normalization (preserves absolute intensity)
            current_gridsat = (current_gridsat - self.gs_mean) / self.gs_std
            next_gridsat    = (next_gridsat - self.gs_mean) / self.gs_std
            current_era5    = (current_era5 - self.era5_mean) / self.era5_std
            next_era5       = (next_era5 - self.era5_mean) / self.era5_std


        if self.mode == 'train' and random.random() > 0.5:
            current_gridsat = torch.flip(current_gridsat, [-1])
            current_era5    = torch.flip(current_era5, [-1])
            next_gridsat    = torch.flip(next_gridsat, [-1])
            next_era5       = torch.flip(next_era5, [-1])

        # Look up per-timestep coordinates from IBTrACS lookup
        storm = sample_info['storm_name']
        cur_key = f"{storm}/{sample_info['timestamp']}"
        nxt_ts = sample_info.get('next_timestamp', sample_info['timestamp'])
        nxt_key = f"{storm}/{nxt_ts}"

        if cur_key in self.coord_lookup:
            cur_lat = self.coord_lookup[cur_key]['lat']
            cur_lon = self.coord_lookup[cur_key]['lon']
        else:
            cur_lat, cur_lon = parse_storm_coordinates(storm)  # fallback to genesis

        if nxt_key in self.coord_lookup:
            nxt_lat = self.coord_lookup[nxt_key]['lat']
            nxt_lon = self.coord_lookup[nxt_key]['lon']
        else:
            nxt_lat, nxt_lon = cur_lat, cur_lon  # fallback: same as current

        # Normalize coordinates to zero-mean, unit-variance scalars
        cur_coords = torch.tensor([
            (cur_lat - self.coord_mean[0].item()) / self.coord_std[0].item(),
            (cur_lon - self.coord_mean[1].item()) / self.coord_std[1].item(),
        ], dtype=torch.float32)
        tgt_coords = torch.tensor([
            (nxt_lat - self.coord_mean[0].item()) / self.coord_std[0].item(),
            (nxt_lon - self.coord_mean[1].item()) / self.coord_std[1].item(),
        ], dtype=torch.float32)

        return {
            'gridsat': current_gridsat,
            'era5': current_era5,
            'target': next_gridsat,
            'target_era5': next_era5,
            'current_coords': cur_coords,   # (2,) normalized scalars
            'target_coords': tgt_coords,    # (2,) normalized scalars for supervision
            'raw_lat': cur_lat,
            'raw_lon': cur_lon,
            'raw_target_lat': nxt_lat,
            'raw_target_lon': nxt_lon,
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
                    img_size=256,
                    coord_lookup_path=None):
    """Year-based split. No random storm selection — fully deterministic.
    train_years / val_years / test_years are disjoint lists of year strings
    (e.g. ['2012','2013',...], ['2021'], ['2022'])."""

    train_dataset = CycloneDataset(
        root_dir=root_dir, years=train_years,
        gridsat_type=gridsat_type, era5_type=era5_type,
        max_samples=max_samples, mode='train', img_size=img_size,
        coord_lookup_path=coord_lookup_path,
    )
    val_dataset = CycloneDataset(
        root_dir=root_dir, years=val_years,
        gridsat_type=gridsat_type, era5_type=era5_type,
        mode='val', img_size=img_size,
        coord_lookup_path=coord_lookup_path,
    )
    test_dataset = CycloneDataset(
        root_dir=root_dir, years=test_years,
        gridsat_type=gridsat_type, era5_type=era5_type,
        mode='test', img_size=img_size,
        coord_lookup_path=coord_lookup_path,
    )

    print(f"Train years {train_years}: {len(train_dataset)} (t, t+1) pairs")
    print(f"Val   years {val_years}:  {len(val_dataset)} (t, t+1) pairs")
    print(f"Test  years {test_years}: {len(test_dataset)} (t, t+1) pairs")

    # persistent_workers=True keeps workers alive across epochs (saves ~30s/epoch
    # respawn cost). prefetch_factor=4 has each worker queue 4 batches ahead, so
    # the GPU doesn't wait when one worker stalls on disk I/O.
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
        """DDIM sampling - deterministic when eta=0
        
        Args:
            xt: Current noisy sample
            noise_pred: Predicted noise from UNet
            t: Current timestep (int)
            t_prev: Previous timestep in the DDIM schedule (int).
                    If None, defaults to t-1 (single-step).
                    For skip-step DDIM (e.g. 50 steps over 1000), pass the actual
                    next timestep in the schedule to denoise correctly.
            eta: Stochasticity (0=deterministic DDIM, 1=DDPM)
            clamp_range: (min, max) tuple for x0_pred clamping. If None, uses [-5, 5].
        """
        device = xt.device
        
        if t_prev is None:
            t_prev = t - 1
        
        alpha_cumprod_t = self.alpha_cumprod.to(device)[t]
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod.to(device)[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod.to(device)[t]
        
        # Use the ACTUAL previous timestep's alpha_cumprod for correct skip-step denoising
        if t_prev >= 0:
            alpha_cumprod_prev = self.alpha_cumprod.to(device)[t_prev]
        else:
            # t_prev = -1 means we're at the final step, alpha_cumprod_prev = 1.0 (fully denoised)
            alpha_cumprod_prev = torch.tensor(1.0, device=device)
        
        # Predict x0 and clamp to valid latent range
        x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t
        if clamp_range is not None:
            x0_pred = torch.clamp(x0_pred, clamp_range[0], clamp_range[1])
        else:
            x0_pred = torch.clamp(x0_pred, -5.0, 5.0)
        
        sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)
        
        # Compute sigma for stochastic sampling
        # Guard against division by zero when alpha_cumprod_t ≈ alpha_cumprod_prev
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
    """Encodes GRIDSAT + ERA5 + temporal info + coordinates for UNet conditioning.
    Matches proven architecture from diffusion_unet_train.py.
    """
    def __init__(self, gridsat_channels=1, era5_channels=4, img_size=256,
                 embed_dim=128, output_channels=64):
        super().__init__()
        self.output_channels = output_channels
        emb_dim = embed_dim * 2
        self.spatial_pos_gridsat = SpatialPositionalEncoding(gridsat_channels, img_size, img_size)
        self.spatial_pos_era5 = SpatialPositionalEncoding(era5_channels, img_size, img_size)
        self.temporal_encoder = TemporalEncoding(embed_dim)
        # Coordinate projection: (lat, lon) → embed_dim
        self.coord_proj = nn.Sequential(
            nn.Linear(2, embed_dim // 2), nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim)
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
        # Add coordinate embedding if provided
        if coords is not None:
            coord_emb = self.coord_proj(coords)  # (B, embed_dim)
            temporal_emb = temporal_emb + coord_emb
        context_emb = self.context_projection(temporal_emb)
        context_spatial = self.context_to_spatial(context_emb).unsqueeze(-1).unsqueeze(-1)
        return image_features + context_spatial, context_emb

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
    """UNet for noise prediction in latent space.
    Proven architecture from diffusion_unet_train.py, adapted for latent-space inputs.
    Attention at middle encoder/decoder level + MidBlock only.
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

        # Encoder: attention only at middle level (i == n_levels-2)
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

from eval_metrics_extended import MetricsCalculator, FID, InceptionScore  # consolidated
from epoch_logger import EpochLogger


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
                if i == 0:
                    print(f"\n===== DEBUG: Evaluation Sampling (batch 0) =====")
                    print(f"  Initial noise: shape={generated.shape}, range=[{generated.min().item():.4f}, {generated.max().item():.4f}]")
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
                generated = scheduler.sample_prev_timestep_ddim(generated, noise_pred, t, t_prev=t_prev, eta=0.0, clamp_range=clamp_range)
                
                if i == 0 and (t == sampling_timesteps[0] or t == sampling_timesteps[len(sampling_timesteps)//4] or t == sampling_timesteps[len(sampling_timesteps)//2] or t == sampling_timesteps[-1]):
                    print(f"  t={t} (t_prev={t_prev}): generated range=[{generated.min().item():.4f}, {generated.max().item():.4f}]")
            
            if vae_decoder is not None:
                generated_unscaled = generated / latent_scale_factor
                generated = vae_decoder(generated_unscaled)
                if i == 0:
                    print(f"  After VAE decode: range=[{generated.min().item():.4f}, {generated.max().item():.4f}]")
                    print(f"  Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
            
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
          f"FID={avg_metrics['fid']:.4f}, IS={avg_metrics['is_mean']:.4f}")
    
    # Restore RNG state
    torch.random.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    
    return avg_metrics

# VAE classes moved to vae.py — re-exported below for backwards compatibility.
# (Avoid circular import: vae.MultiChannelVAETrainer needs MetricsCalculator/FID
# defined below, so we import here at module-bottom, see end of file.)
import random




# ============================================================================
# MultiChannelVAE + MultiChannelVAETrainer live in scripts/vae.py.
# Re-exported at module bottom for backwards compatibility.
# ============================================================================

from vae import MultiChannelVAE, MultiChannelVAETrainer  # re-export



# ============================================================================
# MULTI-CHANNEL DIFFUSION TRAINER  (Stage 2)
# ============================================================================

class MultiChannelDiffusionTrainer:
    """
    Train diffusion UNet in latent space of the 5-channel VAE.
    Architecture is identical to the original DiffusionTrainer but targets
    the joint GRIDSAT+ERA5 latent space.
    """

    def __init__(self, config, mc_vae_checkpoint_path, diffusion_checkpoint_path=None):
        self.config = config
        self.device = torch.device(config['device'])

        self.checkpoint_dir = Path(config['checkpoint_dir']) / 'mc_diffusion'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config['output_dir']) / 'mc_diffusion'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load & freeze multi-channel VAE ----
        print(f"Loading MC-VAE from: {mc_vae_checkpoint_path}")
        ckpt = torch.load(mc_vae_checkpoint_path, map_location=self.device, weights_only=False)
        self.vae = MultiChannelVAE(
            in_channels=5,
            latent_dim=config.get('latent_dim', 4),
            base_channels=config.get('vae_base_channels', 64),
        ).to(self.device)
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        self.latent_scale_factor = ckpt.get('latent_scale_factor', None)
        if self.latent_scale_factor is None:
            print("  ⚠ WARNING: latent_scale_factor NOT found in VAE checkpoint!")
            print("    Will recompute from data. Old checkpoints may lack this key.")
            self.latent_scale_factor = None  # will be set in train() before use
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.latent_clamp_range = (-5.0, 5.0)
        print(f"  ✓ MC-VAE frozen, scale={self.latent_scale_factor}")

        # ---- UNet (operates in 4-dim latent for 5-channel VAE) ----
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
            condition_img_size=config['img_size'],
        ).to(self.device)

        self.condition_downsample = nn.Sequential(
            nn.Conv2d(config['base_channels'], config['base_channels'], 3, 2, 1),
            nn.GroupNorm(8, config['base_channels']), nn.SiLU(),
            nn.Conv2d(config['base_channels'], config['base_channels'], 3, 2, 1),
            nn.GroupNorm(8, config['base_channels']), nn.SiLU(),
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
                config['num_timesteps'], config.get('beta_start', 1e-4),
                config.get('beta_end', 0.02), self.device
            )
            print(f"  Using linear noise schedule")

        self.optimizer = AdamW(
            list(self.unet.parameters()) + list(self.condition_downsample.parameters()),
            lr=config['learning_rate'], weight_decay=config['weight_decay'],
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.get('diffusion_epochs', 200),
            eta_min=config.get('min_lr', 1e-6),
        )

        self.ema = EMA(self.unet, decay=config.get('ema_decay', 0.9999))
        self.use_amp = config.get('use_amp', True)
        # Conservative init_scale to prevent float16 overflow with larger model
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1024, growth_interval=2000) if self.use_amp else None
        self.grad_accum = config.get('gradient_accumulation_steps', 1)
        
        # LR warmup
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.base_lr = config['learning_rate']

        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.train_losses, self.val_losses = [], []
        self.metrics_history = []

        # Rich per-epoch logger (CSV + .log)
        self.logger = EpochLogger(self.metrics_dir, run_tag='mc_diff')

        if diffusion_checkpoint_path:
            self._load_checkpoint(diffusion_checkpoint_path)

        print(f"MC-Diffusion Trainer ready. UNet params: {sum(p.numel() for p in self.unet.parameters()):,}")

    def _load_checkpoint(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.unet.load_state_dict(ckpt['unet_state_dict'])
            self.condition_downsample.load_state_dict(ckpt['condition_downsample_state_dict'])
            # coord_predictor removed — skip legacy keys silently
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.scaler and 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict']:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.start_epoch = ckpt.get('epoch', 0) + 1
            # Restore LR scheduler state directly
            if 'lr_scheduler_state_dict' in ckpt and ckpt['lr_scheduler_state_dict']:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
                print(f"  ✓ LR scheduler state restored directly")
            else:
                # Fallback: fast-forward if no saved state
                for _ in range(self.start_epoch - 1):
                    self.lr_scheduler.step()
                print(f"  ⚠ LR scheduler fast-forwarded {self.start_epoch - 1} steps (no saved state)")
            if 'ema_state_dict' in ckpt:
                self.ema.load_state_dict(ckpt['ema_state_dict'])
            # Restore metrics history for continuous training curves
            if 'metrics_history' in ckpt:
                self.metrics_history = ckpt['metrics_history']
                print(f"  ✓ Metrics history restored ({len(self.metrics_history)} entries)")
            if 'train_losses' in ckpt:
                self.train_losses = ckpt['train_losses']
                self.val_losses = ckpt['val_losses']
                print(f"  ✓ Loss history restored ({len(self.train_losses)} epochs)")
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  ✓ Resumed MC-Diffusion from epoch {self.start_epoch - 1}, LR={current_lr:.2e}, best_val={self.best_val_loss:.6f}")
        except Exception as e:
            print(f"  ✗ Checkpoint load failed: {e} — starting fresh")
            self.start_epoch = 1

    # ---- Latent encoding / decoding ----
    @torch.no_grad()
    def encode_to_latent(self, target_5ch):
        mu, logvar = self.vae.encode(target_5ch)
        z = self.vae.reparameterize(mu, logvar)
        return z * self.latent_scale_factor

    @torch.no_grad()
    def decode_from_latent(self, z_scaled):
        return self.vae.decode(z_scaled / self.latent_scale_factor)

    @torch.no_grad()
    def compute_latent_clamp_range(self, dataloader, num_batches=50):
        """Compute the actual scaled-latent range from data for dynamic DDIM clamping."""
        self.vae.eval()
        all_mins, all_maxs = [], []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            target_gs = batch['target'].to(self.device)
            target_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([target_gs, target_era5], dim=1)
            z_scaled = self.encode_to_latent(target_5ch)
            all_mins.append(z_scaled.min().item())
            all_maxs.append(z_scaled.max().item())
        
        z_min = min(all_mins)
        z_max = max(all_maxs)
        # Add 10% margin beyond observed range
        margin = (z_max - z_min) * 0.1
        clamp_min = z_min - margin
        clamp_max = z_max + margin
        
        self.latent_clamp_range = (clamp_min, clamp_max)
        print(f"  ✓ Latent clamp range computed: [{clamp_min:.2f}, {clamp_max:.2f}]")
        print(f"    (raw latent range: [{z_min:.3f}, {z_max:.3f}])")
        return self.latent_clamp_range

    # ---- Training ----
    def train_epoch(self, loader, epoch):
        self.unet.train(); self.condition_downsample.train(); self.vae.eval()
        total, n = 0, 0
        pbar = tqdm(loader, desc=f'MC-Diff Train Ep {epoch}')
        for bi, batch in enumerate(pbar):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            cur_coords = batch['current_coords'].to(self.device)   # (B, 2)
            # Build 5-channel target (GRIDSAT + ERA5 only)
            target_gs = batch['target'].to(self.device)
            target_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([target_gs, target_era5], dim=1)
            bs = target_5ch.shape[0]

            with torch.no_grad():
                z = self.encode_to_latent(target_5ch)

            enc_cond, ctx = self.unet.condition_encoder(gridsat, era5, timestamps, coords=cur_coords)
            enc_cond = self.condition_downsample(enc_cond)

            t = torch.randint(0, self.scheduler.num_timesteps, (bs,), device=self.device)
            noise = torch.randn_like(z)
            noisy = self.scheduler.add_noise(z, noise, t)

            if bi % self.grad_accum == 0:
                self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred = self.unet(noisy, gridsat, era5, timestamps, t,
                                 encoded_condition=enc_cond, context_emb=ctx)
                loss = F.mse_loss(pred, noise) / self.grad_accum

            # NaN protection
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
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.ema.update(self.unet)
                        last_grad_norm = float(grad_norm.item())
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.unet.parameters()) + list(self.condition_downsample.parameters()), 1.0)
                    self.optimizer.step()
                    self.ema.update(self.unet)
                    last_grad_norm = float(grad_norm.item())

            total += loss.item() * self.grad_accum; n += 1
            self.logger.log_batch(loss=float(loss.item() * self.grad_accum),
                                  grad_norm=last_grad_norm)
            pbar.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}")
            if bi % 50 == 0:
                torch.cuda.empty_cache()

        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.unet.eval(); self.vae.eval()
        total, n = 0, 0
        for batch in tqdm(loader, desc='MC-Diff Val'):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            cur_coords = batch['current_coords'].to(self.device)
            target_gs = batch['target'].to(self.device)
            target_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([target_gs, target_era5], dim=1)
            bs = target_5ch.shape[0]

            z = self.encode_to_latent(target_5ch)
            enc_cond, ctx = self.unet.condition_encoder(gridsat, era5, timestamps, coords=cur_coords)
            enc_cond = self.condition_downsample(enc_cond)

            t = torch.randint(0, self.scheduler.num_timesteps, (bs,), device=self.device)
            noise = torch.randn_like(z)
            noisy = self.scheduler.add_noise(z, noise, t)
            pred = self.unet(noisy, gridsat, era5, timestamps, t,
                             encoded_condition=enc_cond, context_emb=ctx)
            total += F.mse_loss(pred, noise).item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def sample(self, gridsat, era5, timestamps, coords=None, num_steps=50, clamp_range=None):
        """Generate 5-channel output."""
        self.unet.eval(); self.vae.eval()
        bs = gridsat.shape[0]
        lat_size = self.config['img_size'] // 4
        lat_dim = self.config.get('latent_dim', 4)
        z = torch.randn(bs, lat_dim, lat_size, lat_size, device=self.device)

        enc_cond, ctx = self.unet.condition_encoder(gridsat, era5, timestamps, coords=coords)
        enc_cond = self.condition_downsample(enc_cond)

        step_size = max(1, self.scheduler.num_timesteps // num_steps)
        ts = list(range(0, self.scheduler.num_timesteps, step_size))[::-1]
        for i, t_val in enumerate(ts):
            t_batch = torch.full((bs,), t_val, device=self.device, dtype=torch.long)
            noise_pred = self.unet(z, gridsat, era5, timestamps, t_batch,
                                   encoded_condition=enc_cond, context_emb=ctx)
            t_prev = ts[i+1] if i+1 < len(ts) else -1
            z = self.scheduler.sample_prev_timestep_ddim(z, noise_pred, t_val, t_prev=t_prev, eta=0.0, clamp_range=clamp_range)

        return self.decode_from_latent(z)  # (B, 5, H, W)

    # ---- Checkpoint / samples / metrics ----
    def save_checkpoint(self, epoch, val_loss):
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
            'metrics_history': self.metrics_history,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
        }
        torch.save(ckpt, self.checkpoint_dir / 'mc_diff_latest.pt')
        if is_best:
            self.ema.apply(self.unet)
            ckpt['unet_state_dict'] = self.unet.state_dict()
            torch.save(ckpt, self.checkpoint_dir / 'mc_diff_best.pt')
            self.ema.restore(self.unet)
            print(f"  ✓ New best MC-Diff (val_loss: {val_loss:.6f})")

    def save_samples(self, loader, epoch):
        batch = next(iter(loader))
        ns = min(6, len(batch['gridsat']))
        gridsat = batch['gridsat'][:ns].to(self.device)
        era5 = batch['era5'][:ns].to(self.device)
        target_gs = batch['target'][:ns]
        target_era5 = batch['target_era5'][:ns]
        timestamps = batch['timestamp'][:ns]
        cur_coords = batch['current_coords'][:ns].to(self.device)

        gen5 = self.sample(gridsat, era5, timestamps, coords=cur_coords, clamp_range=self.latent_clamp_range)
        gen_gs = gen5[:, 0:1].cpu().numpy()
        gen_era5 = gen5[:, 1:5].cpu().numpy()
        tgt_gs = target_gs.numpy()
        tgt_era5 = target_era5.numpy()
        inp_gs = batch['gridsat'][:ns].numpy()   # T gridsat input
        inp_era5 = batch['era5'][:ns].numpy()     # T era5 input

        # Channel names for image channels only (skip coords in visualization)
        ch_names = ['GRIDSAT', 'U-component', 'V-component', 'Temperature', 'Pressure']
        row_labels = ['Input', 'Target', 'Generated']
        rows_per_ch = 3  # T, T+1, Generated
        n_rows = rows_per_ch * len(ch_names)
        fig, axes = plt.subplots(n_rows, ns, figsize=(2.5 * ns, 2 * n_rows))
        if ns == 1:
            axes = axes.reshape(-1, 1)

        for ch_i, ch_name in enumerate(ch_names):
            r_t     = ch_i * rows_per_ch      # T input row
            r_t1    = ch_i * rows_per_ch + 1   # T+1 target row
            r_gen   = ch_i * rows_per_ch + 2   # Generated row
            cmap = 'gray' if ch_i == 0 else 'viridis'

            for s in range(ns):
                # -- T input --
                if ch_i == 0:
                    axes[r_t, s].imshow(inp_gs[s, 0], cmap=cmap)
                else:
                    axes[r_t, s].imshow(inp_era5[s, ch_i - 1], cmap=cmap)
                axes[r_t, s].set_xticks([])
                axes[r_t, s].set_yticks([])

                # -- T+1 target --
                if ch_i == 0:
                    axes[r_t1, s].imshow(tgt_gs[s, 0], cmap=cmap)
                else:
                    axes[r_t1, s].imshow(tgt_era5[s, ch_i - 1], cmap=cmap)
                axes[r_t1, s].set_xticks([])
                axes[r_t1, s].set_yticks([])

                # -- Generated --
                if ch_i == 0:
                    axes[r_gen, s].imshow(gen_gs[s, 0], cmap=cmap)
                else:
                    axes[r_gen, s].imshow(gen_era5[s, ch_i - 1], cmap=cmap)
                axes[r_gen, s].set_xticks([])
                axes[r_gen, s].set_yticks([])

            # Row labels (left side, first sample column) — visible because we use set_ylabel instead of axis('off')
            axes[r_t,   0].set_ylabel(f'Input {ch_name}',     fontsize=7, fontweight='bold')
            axes[r_t1,  0].set_ylabel(f'Target {ch_name}',    fontsize=7, fontweight='bold')
            axes[r_gen, 0].set_ylabel(f'Gen. {ch_name}',      fontsize=7, fontweight='bold')

        plt.suptitle(f'MC-Diff Samples — Epoch {epoch}', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'mc_diff_samples_ep{epoch}.png', dpi=150)
        plt.close()

    @torch.no_grad()
    def evaluate_era5(self, loader, num_samples=100):
        """Compute per-channel metrics + steering flow displacement accuracy."""
        self.ema.apply(self.unet)
        self.unet.eval(); self.vae.eval()
        ch_names = ['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']
        ch_mse = {c: [] for c in ch_names}
        ch_mae = {c: [] for c in ch_names}
        ch_psnr = {c: [] for c in ch_names}
        ch_ssim = {c: [] for c in ch_names}
        overall_mse, overall_mae, overall_psnr, overall_ssim, overall_sid = [], [], [], [], []
        # Steering flow tracking
        gt_displacements, gen_displacements = [], []
        n = 0
        calc = MetricsCalculator()

        # ERA5 normalization stats for un-normalizing U/V wind
        ds = loader.dataset
        era5_mean = ds.era5_mean.numpy().flatten()  # [u_mean, v_mean, t_mean, p_mean]
        era5_std = ds.era5_std.numpy().flatten()

        for batch in loader:
            if n >= num_samples:
                break
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            timestamps = batch['timestamp']
            cur_coords = batch['current_coords'].to(self.device)
            target_gs = batch['target'].to(self.device)
            target_era5 = batch['target_era5'].to(self.device)
            target_5ch = torch.cat([target_gs, target_era5], dim=1)

            gen5 = self.sample(gridsat, era5, timestamps, coords=cur_coords, clamp_range=self.latent_clamp_range)

            # Per-channel metrics
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

            # ---- Steering flow displacement (from U/V wind channels) ----
            roi = 64
            h, w = gen5.shape[2], gen5.shape[3]
            y0, x0 = (h - roi) // 2, (w - roi) // 2
            gt_u = target_5ch[:, 1, y0:y0+roi, x0:x0+roi].cpu().numpy() * era5_std[0] + era5_mean[0]
            gt_v = target_5ch[:, 2, y0:y0+roi, x0:x0+roi].cpu().numpy() * era5_std[1] + era5_mean[1]
            gen_u = gen5[:, 1, y0:y0+roi, x0:x0+roi].cpu().numpy() * era5_std[0] + era5_mean[0]
            gen_v = gen5[:, 2, y0:y0+roi, x0:x0+roi].cpu().numpy() * era5_std[1] + era5_mean[1]
            dt_sec = 3.0 * 3600  # 3-hour timestep
            for b in range(gt_u.shape[0]):
                lat_val = batch['raw_lat'][b] if isinstance(batch['raw_lat'][b], (int, float)) else float(batch['raw_lat'][b])
                cos_lat = max(np.cos(np.radians(lat_val)), 0.1)
                gt_displacements.append((gt_v[b].mean() * dt_sec / 111000.0,
                                          gt_u[b].mean() * dt_sec / (111000.0 * cos_lat)))
                gen_displacements.append((gen_v[b].mean() * dt_sec / 111000.0,
                                           gen_u[b].mean() * dt_sec / (111000.0 * cos_lat)))
            n += gridsat.shape[0]

        self.ema.restore(self.unet)
        results = {}
        for cn in ch_names:
            results[f'{cn}_mse'] = float(np.mean(ch_mse[cn])) if ch_mse[cn] else 0
            results[f'{cn}_mae'] = float(np.mean(ch_mae[cn])) if ch_mae[cn] else 0
            results[f'{cn}_psnr'] = float(np.mean(ch_psnr[cn])) if ch_psnr[cn] else 0
            results[f'{cn}_ssim'] = float(np.mean(ch_ssim[cn])) if ch_ssim[cn] else 0
        results['overall_mse'] = float(np.mean(overall_mse)) if overall_mse else 0
        results['overall_mae'] = float(np.mean(overall_mae)) if overall_mae else 0
        results['overall_psnr'] = float(np.mean(overall_psnr)) if overall_psnr else 0
        results['overall_ssim'] = float(np.mean(overall_ssim)) if overall_ssim else 0
        results['overall_sid'] = float(np.mean(overall_sid)) if overall_sid else 0

        # Steering flow displacement metrics
        if gt_displacements:
            gt_d = np.array(gt_displacements)
            gen_d = np.array(gen_displacements)
            disp_mae_lat = float(np.mean(np.abs(gt_d[:, 0] - gen_d[:, 0])))
            disp_mae_lon = float(np.mean(np.abs(gt_d[:, 1] - gen_d[:, 1])))
            disp_mae_total = float(np.mean(np.sqrt((gt_d[:, 0] - gen_d[:, 0])**2 +
                                                     (gt_d[:, 1] - gen_d[:, 1])**2)))
            results['steering_mae_lat_deg'] = disp_mae_lat
            results['steering_mae_lon_deg'] = disp_mae_lon
            results['steering_mae_total_deg'] = disp_mae_total
            print(f"  Steering Flow MAE: dlat={disp_mae_lat:.3f} dlon={disp_mae_lon:.3f} total={disp_mae_total:.3f} deg")
            self._plot_steering_flow(gt_d, gen_d)

        return results

    def _plot_steering_flow(self, gt_d, gen_d):
        """3-panel steering flow analysis: quiver, scatter, error histogram."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (0) Quiver: GT vs Generated steering vectors
        N = min(len(gt_d), 200)
        idx = np.random.choice(len(gt_d), N, replace=False) if len(gt_d) > N else np.arange(len(gt_d))
        origins = np.zeros(N)
        axes[0].quiver(origins, origins, gt_d[idx, 1], gt_d[idx, 0],
                        color='#4CAF50', alpha=0.4, scale=5, label='GT Steering')
        axes[0].quiver(origins, origins, gen_d[idx, 1], gen_d[idx, 0],
                        color='#F44336', alpha=0.4, scale=5, label='Gen Steering')
        axes[0].set_xlabel('dlon (deg)'); axes[0].set_ylabel('dlat (deg)')
        axes[0].set_title('Steering Vectors (3h displacement)')
        axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_aspect('equal')

        # (1) Scatter: GT vs Gen displacement components
        axes[1].scatter(gt_d[:, 0], gen_d[:, 0], alpha=0.4, s=15, c='#2196F3', label='dlat')
        axes[1].scatter(gt_d[:, 1], gen_d[:, 1], alpha=0.4, s=15, c='#FF9800', label='dlon')
        lim = max(np.abs(gt_d).max(), np.abs(gen_d).max()) * 1.2
        axes[1].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='Perfect')
        mae_lat = np.mean(np.abs(gt_d[:, 0] - gen_d[:, 0]))
        mae_lon = np.mean(np.abs(gt_d[:, 1] - gen_d[:, 1]))
        axes[1].set_xlabel('GT displacement (deg)'); axes[1].set_ylabel('Gen displacement (deg)')
        axes[1].set_title(f'Displacement Scatter (MAE: lat={mae_lat:.3f}, lon={mae_lon:.3f})')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        # (2) Error histogram
        errors = np.sqrt((gt_d[:, 0] - gen_d[:, 0])**2 + (gt_d[:, 1] - gen_d[:, 1])**2)
        axes[2].hist(errors, bins=30, color='#9C27B0', alpha=0.7, edgecolor='white')
        axes[2].axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean={np.mean(errors):.3f}')
        axes[2].axvline(np.median(errors), color='orange', linestyle='--', label=f'Median={np.median(errors):.3f}')
        axes[2].set_xlabel('Displacement Error (deg)'); axes[2].set_ylabel('Count')
        axes[2].set_title('Steering Flow Error Distribution')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'Steering Flow Analysis (N={len(gt_d)}, 3h from ERA5 U/V)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mc_diff_steering_flow.png', dpi=150)
        plt.close()
        print(f"  Steering flow plot saved")


    # ---- Main train loop ----
    def train(self, train_loader, val_loader):
        num_epochs = self.config.get('diffusion_epochs', 100)

        # Recompute scale factor if missing from VAE checkpoint
        if self.latent_scale_factor is None:
            print("  Recomputing latent_scale_factor from validation data...")
            all_z = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= 50:
                        break
                    gs = batch['gridsat'].to(self.device)
                    era5 = batch['era5'].to(self.device)
                    tgt_gs = batch['target'].to(self.device)
                    tgt_era5 = batch['target_era5'].to(self.device)
                    x5 = torch.cat([tgt_gs, tgt_era5], dim=1)
                    mu, logvar = self.vae.encode(x5)
                    z = self.vae.reparameterize(mu, logvar)
                    all_z.append(z.cpu())
            z_all = torch.cat(all_z, 0)
            z_std = z_all.std().item()
            self.latent_scale_factor = 1.0 / z_std
            print(f"  ✓ Recomputed scale_factor = {self.latent_scale_factor:.5f} (z_std={z_std:.4f})")

        # Compute dynamic latent clamp range from data
        self.compute_latent_clamp_range(val_loader, num_batches=50)

        print(f"\n{'='*60}\nSTAGE 2: MULTI-CHANNEL DIFFUSION TRAINING\n{'='*60}")
        print(f"Epochs: {self.start_epoch} → {num_epochs}")
        print(f"Latent scale factor: {self.latent_scale_factor:.5f}")

        for epoch in range(self.start_epoch, num_epochs + 1):
            # LR warmup
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
                    ch_metrics = self.evaluate_era5(val_loader, num_samples=self.config.get('eval_samples', 50))
                    eval_aggregate.update(ch_metrics)
                except Exception as e:
                    print(f"  Eval failed: {e}")

            is_best = val_loss < self.best_val_loss
            self.logger.end_epoch(
                epoch,
                learning_rate=lr,
                phase='mc_diff',
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

        print(f"\n{'='*60}\nMC-DIFFUSION TRAINING COMPLETE\nBest val loss: {self.best_val_loss:.6f}\n{'='*60}")

        # ===== Final comprehensive evaluation =====
        print("Running final comprehensive evaluation...")
        try:
            final_metrics = self.evaluate_era5(val_loader, num_samples=min(self.config.get('eval_samples', 50) * 2, 200))
            print(f"\nFinal Evaluation Metrics:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")
            # Save final metrics as JSON
            with open(self.output_dir / 'mc_diff_final_eval_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=4)
        except Exception as e:
            print(f"Warning: Final evaluation failed: {e}")
            final_metrics = {}

        # Save final metrics CSV
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(self.metrics_dir / 'mc_diff_final_metrics.csv', index=False)
        print(f"\n  Metrics CSV saved to: {self.metrics_dir / 'mc_diff_final_metrics.csv'}")
        print(f"  Columns: {list(metrics_df.columns)}")
        print(f"  Rows (epochs): {len(metrics_df)}")

        # Save config as JSON
        config_save = {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()}
        try:
            with open(self.output_dir / 'mc_diff_config.json', 'w') as f:
                json.dump(config_save, f, indent=4, default=str)
        except Exception as e:
            print(f"  Warning: Could not save config JSON: {e}")

        # ===== Plot training curves (loss + actual PSNR/SSIM) =====
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        loss_epochs = [m['epoch'] for m in self.metrics_history if 'train_loss' in m][:len(self.train_losses)]
        if not loss_epochs:
            loss_epochs = list(range(1, len(self.train_losses) + 1))
        plt.plot(loss_epochs, self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(loss_epochs, self.val_losses, label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('MC-Diffusion Training (Noise Prediction)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot evaluation metrics if available (actual computed values)
        eval_epochs = [m['epoch'] for m in self.metrics_history if 'overall_psnr' in m]
        eval_psnrs = [m['overall_psnr'] for m in self.metrics_history if 'overall_psnr' in m]
        eval_ssims = [m['overall_ssim'] for m in self.metrics_history if 'overall_ssim' in m]

        if eval_epochs and eval_psnrs:
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

            plt.title('Evaluation Metrics (Actual)')
            ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'mc_diff_training_curves.png', dpi=150)
        plt.close()

        # Per-channel metrics breakdown plot (MSE + PSNR + SSIM)
        ch_names_plot = ['GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres']
        ch_mse_data = {c: [] for c in ch_names_plot}
        ch_psnr_data = {c: [] for c in ch_names_plot}
        ch_ssim_data = {c: [] for c in ch_names_plot}
        ch_mae_data = {c: [] for c in ch_names_plot}
        ch_epochs = []
        overall_sid_data = []
        for m in self.metrics_history:
            if 'GRIDSAT_mse' in m:
                ch_epochs.append(m['epoch'])
                overall_sid_data.append(m.get('overall_sid', 0))
                for c in ch_names_plot:
                    ch_mse_data[c].append(m.get(f'{c}_mse', 0))
                    ch_psnr_data[c].append(m.get(f'{c}_psnr', 0))
                    ch_ssim_data[c].append(m.get(f'{c}_ssim', 0))
                    ch_mae_data[c].append(m.get(f'{c}_mae', 0))

        if ch_epochs:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # (0,0) Per-channel MSE
            for c in ch_names_plot:
                axes[0, 0].plot(ch_epochs, ch_mse_data[c], '-o', label=c, markersize=3)
            axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('MSE')
            axes[0, 0].set_title('Per-Channel MSE'); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

            # (0,1) Per-channel MAE
            for c in ch_names_plot:
                axes[0, 1].plot(ch_epochs, ch_mae_data[c], '-o', label=c, markersize=3)
            axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('MAE')
            axes[0, 1].set_title('Per-Channel MAE'); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

            # (0,2) Per-channel PSNR
            for c in ch_names_plot:
                axes[0, 2].plot(ch_epochs, ch_psnr_data[c], '-o', label=c, markersize=3)
            axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('PSNR (dB)')
            axes[0, 2].set_title('Per-Channel PSNR'); axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True, alpha=0.3)

            # (1,0) Per-channel SSIM
            for c in ch_names_plot:
                axes[1, 0].plot(ch_epochs, ch_ssim_data[c], '-o', label=c, markersize=3)
            axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('SSIM')
            axes[1, 0].set_title('Per-Channel SSIM'); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

            # (1,1) Overall SID
            axes[1, 1].plot(ch_epochs, overall_sid_data, 'g-o', markersize=4)
            axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('SID')
            axes[1, 1].set_title('Overall SID (Spectral Info Divergence)'); axes[1, 1].grid(True, alpha=0.3)

            # (1,2) Overall PSNR + SSIM (dual axis)
            if eval_epochs:
                ax_p = axes[1, 2]
                ax_p.plot(eval_epochs, eval_psnrs, 'b-o', label='PSNR (dB)', markersize=4)
                ax_p.set_xlabel('Epoch'); ax_p.set_ylabel('PSNR (dB)', color='b')
                ax_p.tick_params(axis='y', labelcolor='b')
                ax_s = ax_p.twinx()
                ax_s.plot(eval_epochs, eval_ssims, 'r-s', label='SSIM', markersize=4)
                ax_s.set_ylabel('SSIM', color='r')
                ax_s.tick_params(axis='y', labelcolor='r')
                ax_p.set_title('Overall PSNR & SSIM')
                ax_p.grid(True, alpha=0.3)

            plt.suptitle('MC-Diffusion Per-Channel Metrics', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'mc_diff_channel_metrics.png', dpi=150)
            plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Thin entry-point. The canonical config lives in scripts/assembly.py.

    For a full pipeline run use:    python scripts/assembly.py
    For MC-Diff only:                python scripts/multi_channel_diffusion.py
    """
    from assembly import CONFIG, build_vae_dataloaders

    config = dict(CONFIG)
    config["batch_size"] = config.get("p1_batch_size", 4)   # MC-Diff used the same single-frame batch as Phase-1 RF

    print("\n=== Multi-Channel Diffusion ===\n")
    train_loader, val_loader, _ = build_vae_dataloaders(config)

    mc_vae_path = config["mc_vae_checkpoint"]
    from pathlib import Path
    if not Path(mc_vae_path).exists():
        print(f"VAE checkpoint missing — training first.")
        vae_trainer = MultiChannelVAETrainer(config)
        vae_trainer.train(train_loader, val_loader)

    diff_trainer = MultiChannelDiffusionTrainer(config, mc_vae_path)
    diff_trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
