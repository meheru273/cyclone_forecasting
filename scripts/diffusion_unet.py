import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
# CONDITION ENCODING (from condition_encoding.py)
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
        pe = torch.zeros(d_model, h, w)
        y_pos = torch.arange(0, h).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(0, w).unsqueeze(0).repeat(h, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[0::4, :, :] = torch.sin(y_pos * div_term[0::2].unsqueeze(-1).unsqueeze(-1))
        pe[1::4, :, :] = torch.cos(y_pos * div_term[0::2].unsqueeze(-1).unsqueeze(-1))
        pe[2::4, :, :] = torch.sin(x_pos * div_term[1::2].unsqueeze(-1).unsqueeze(-1))
        pe[3::4, :, :] = torch.cos(x_pos * div_term[1::2].unsqueeze(-1).unsqueeze(-1))
        return pe

    def forward(self, x):
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
        dt = datetime.strptime(timestamp_str.split('.')[0], "%Y-%m-%d %H_%M_%S")
        return (dt.year - 2000, dt.month - 1, dt.day - 1, dt.hour)

    def forward(self, timestamps):
        if isinstance(timestamps, (list, tuple)) and isinstance(timestamps[0], str):
            years, months, days, hours = [], [], [], []
            for ts in timestamps:
                y, m, d, h = self.parse_timestamp(ts)
                years.append(y); months.append(m); days.append(d); hours.append(h)
            years = torch.tensor(years, dtype=torch.long)
            months = torch.tensor(months, dtype=torch.long)
            days = torch.tensor(days, dtype=torch.long)
            hours = torch.tensor(hours, dtype=torch.long)
        else:
            years, months = timestamps[:, 0].long(), timestamps[:, 1].long()
            days, hours = timestamps[:, 2].long(), timestamps[:, 3].long()
        device = self.year_embed.weight.device
        years, months = years.to(device), months.to(device)
        days, hours = days.to(device), hours.to(device)
        temporal_emb = torch.cat([
            self.year_embed(years), self.month_embed(months),
            self.day_embed(days), self.hour_embed(hours)
        ], dim=-1)
        return self.projection(temporal_emb)


class ConditionEncoder(nn.Module):
    def __init__(self, gridsat_channels=1, era5_channels=4, img_size=256,
                 embed_dim=128, output_channels=64):
        super().__init__()
        self.output_channels = output_channels
        emb_dim = embed_dim * 2
        self.spatial_pos_gridsat = SpatialPositionalEncoding(gridsat_channels, img_size, img_size)
        self.spatial_pos_era5 = SpatialPositionalEncoding(era5_channels, img_size, img_size)
        self.temporal_encoder = TemporalEncoding(embed_dim)
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

    def forward(self, gridsat, era5, timestamps):
        gridsat_pos = self.spatial_pos_gridsat(gridsat)
        era5_pos = self.spatial_pos_era5(era5)
        combined = torch.cat([gridsat_pos, era5_pos], dim=1)
        image_features = self.image_encoder(combined)
        temporal_emb = self.temporal_encoder(timestamps)
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
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod).to(device)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod).to(device)

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cum_prod = self.alpha_cum_prod.to(device)
        self.sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(device)
        self.sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(device)
        return self

    def add_noise(self, original, noise, t):
        b = original.shape[0]
        sqrt_alpha = self.sqrt_alpha_cum_prod[t].view(b, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cum_prod[t].view(b, 1, 1, 1)
        return sqrt_alpha * original + sqrt_one_minus * noise

    def sample_prev_timestep_ddim(self, xt, noise_pred, t, t_prev=None, eta=0.0):
        if isinstance(t, int):
            alpha_t = self.alpha_cum_prod[t]
        else:
            alpha_t = self.alpha_cum_prod[t[0]] if t.dim() > 0 else self.alpha_cum_prod[t]
        if t_prev is None or (isinstance(t_prev, int) and t_prev < 0):
            alpha_t_prev = torch.tensor(1.0, device=self.device)
        else:
            if isinstance(t_prev, int):
                alpha_t_prev = self.alpha_cum_prod[t_prev]
            else:
                alpha_t_prev = self.alpha_cum_prod[t_prev[0]] if t_prev.dim() > 0 else self.alpha_cum_prod[t_prev]

        x0_pred = (xt - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -3, 3)
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_t_prev - sigma_t ** 2, min=0)) * noise_pred
        x_prev = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt
        if eta > 0 and sigma_t > 0:
            x_prev = x_prev + sigma_t * torch.randn_like(xt)
        return x_prev


# ============================================================================
# DIFFUSION UNET BUILDING BLOCKS
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
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1) if up_sample else nn.Identity()

    def forward(self, x, t_emb, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        return x


# ============================================================================
# DIFFUSION UNET (pixel-space, 256x256)
# ============================================================================

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 channel_mults=(1, 2, 4), t_emb_dim=256, num_heads=8,
                 gridsat_channels=1, era5_channels=4, img_size=256,
                 condition_embed_dim=128):
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
            img_size=img_size, embed_dim=condition_embed_dim, output_channels=base_channels)

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

        decoder_cfgs = [
            (ch_list[-1], ch_list[-1], ch_list[-1], False, False),
            (ch_list[-1], ch_list[-2], ch_list[-2], True,  True),
            (ch_list[-2], ch_list[-3], ch_list[-3], True,  False),
        ]
        self.decoders = nn.ModuleList([
            UpBlock(in_ch, skip_ch, out_ch, emb_dim, up_sample=up, use_attention=attn, num_heads=num_heads)
            for in_ch, skip_ch, out_ch, up, attn in decoder_cfgs])

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch_list[0]), nn.SiLU(),
            nn.Conv2d(ch_list[0], out_channels, 3, padding=1))

    def forward(self, noisy_image, gridsat, era5, timestamps, t,
                encoded_condition=None, context_emb=None):

        t_emb = get_time_embeddings(t, self.t_proj[0].in_features)
        t_emb = self.t_proj(t_emb)

        if encoded_condition is None or context_emb is None:
            encoded_condition, context_emb = self.condition_encoder(gridsat, era5, timestamps)
        ctx_emb = self.context_proj(context_emb)
        comb = t_emb + ctx_emb

        x = torch.cat([noisy_image, encoded_condition], dim=1)
        x = self.init_conv(x)

        skips = []
        for enc in self.encoders:
            x, skip = enc(x, comb)
            skips.append(skip)

        x = self.mid_block(x, comb)

        for dec_idx, dec in enumerate(self.decoders):
            skip_idx = len(skips) - 1 - dec_idx
            x = dec(x, comb, skips[skip_idx])

        x = self.final_conv(x)
        return x


# ============================================================================
# DATA LOADER
# ============================================================================

# Hardcoded global normalization statistics computed from the full dataset:
#   GRIDSAT  mean=169.8992  std=122.2612
#   ERA5[0]  mean=-1.1655   std=5.3889
#   ERA5[1]  mean=0.3835    std=5.0042
#   ERA5[2]  mean=298.4892  std=4.2326
#   ERA5[3]  mean=100976.2891 std=521.1500
_GS_MEAN  = 169.8992
_GS_STD   = 122.2612
_E5_MEAN  = [-1.1655, 0.3835, 298.4892, 100976.2891]
_E5_STD   = [5.3889,  5.0042,   4.2326,    521.1500]


class CycloneDataset(Dataset):
    def __init__(self, root_dir, years, gridsat_type='GRIDSAT_data.npy',
                 era5_type='ERA5_data.npy', transform=None, max_samples=None,
                 test_storm=None, val_storms=None, mode='train', img_size=256):
        self.root_dir = Path(root_dir)
        self.gridsat_type = gridsat_type
        self.era5_type = era5_type
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        self.test_storm = test_storm if isinstance(test_storm, list) else [test_storm] if test_storm else []
        self.val_storms = val_storms if isinstance(val_storms, list) else [val_storms] if val_storms else []
        self.mode = mode
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
                if self.mode == 'train':
                    if storm_name in self.test_storm or storm_name in self.val_storms:
                        continue
                elif self.mode == 'val':
                    if storm_name not in self.val_storms:
                        continue
                elif self.mode == 'test':
                    if storm_name not in self.test_storm:
                        continue
                timestamps = sorted([t.name for t in cyclone_folder.iterdir() if t.is_dir()])
                for i in range(len(timestamps) - 1):
                    current_ts = timestamps[i]
                    next_ts = timestamps[i + 1]
                    current_gridsat = cyclone_folder / current_ts / self.gridsat_type
                    current_era5 = cyclone_folder / current_ts / self.era5_type
                    next_gridsat = cyclone_folder / next_ts / self.gridsat_type
                    if current_gridsat.exists() and current_era5.exists() and next_gridsat.exists():
                        self.samples.append({
                            'current_gridsat': str(current_gridsat),
                            'current_era5': str(current_era5),
                            'next_gridsat': str(next_gridsat),
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

        if current_gridsat.ndim == 3: current_gridsat = current_gridsat[0:1]
        elif current_gridsat.ndim == 2: current_gridsat = current_gridsat[np.newaxis, ...]
        if next_gridsat.ndim == 3: next_gridsat = next_gridsat[0:1]
        elif next_gridsat.ndim == 2: next_gridsat = next_gridsat[np.newaxis, ...]
        if current_era5.ndim == 3: current_era5 = current_era5[:4]
        elif current_era5.ndim == 2: current_era5 = current_era5[np.newaxis, ...]

        current_gridsat = torch.from_numpy(current_gridsat)
        current_era5 = torch.from_numpy(current_era5)
        next_gridsat = torch.from_numpy(next_gridsat)

        sz = (self.img_size, self.img_size)
        current_gridsat = F.interpolate(current_gridsat.unsqueeze(0), size=sz, mode='bilinear', align_corners=False).squeeze(0)
        current_era5    = F.interpolate(current_era5.unsqueeze(0),    size=sz, mode='bilinear', align_corners=False).squeeze(0)
        next_gridsat    = F.interpolate(next_gridsat.unsqueeze(0),    size=sz, mode='bilinear', align_corners=False).squeeze(0)

        # Hardcoded global normalization stats
        gs_mean = torch.tensor(_GS_MEAN, dtype=torch.float32)
        gs_std  = torch.tensor(_GS_STD,  dtype=torch.float32)
        current_gridsat = ((current_gridsat - gs_mean) / gs_std.clamp(min=1e-8)).clamp(-3, 3) / 3
        next_gridsat    = ((next_gridsat    - gs_mean) / gs_std.clamp(min=1e-8)).clamp(-3, 3) / 3

        era5_mean = torch.tensor(_E5_MEAN, dtype=torch.float32)
        era5_std  = torch.tensor(_E5_STD,  dtype=torch.float32)
        current_era5 = ((current_era5 - era5_mean[:, None, None]) /
                        era5_std[:, None, None].clamp(min=1e-8)).clamp(-3, 3) / 3

        return {
            'gridsat': current_gridsat, 'era5': current_era5, 'target': next_gridsat,
            'timestamp': sample_info['timestamp'], 'storm_name': sample_info['storm_name'],
            'year': sample_info['year']
        }


def get_all_storms(root_dir, years):
    root_dir = Path(root_dir)
    all_storms = []
    for year_folder in years:
        year_path = root_dir / year_folder
        if not year_path.exists(): continue
        nested_path = year_path / year_folder
        if nested_path.exists(): year_path = nested_path
        for cf in year_path.iterdir():
            if cf.is_dir(): all_storms.append(cf.name)
    return all_storms


def get_dataloaders(root_dir, batch_size=4, num_workers=2,
                    train_years=['2006','2007','2008','2009','2010','2011','2012'],
                    test_years=['2011'], gridsat_type='GRIDSAT_data.npy',
                    era5_type='ERA5_data.npy', max_samples=None,
                    test_storm=["2022349N13068"], num_val_storms=5,
                    img_size=256, random_seed=42):

    all_storms = get_all_storms(root_dir, train_years)
    available = [s for s in all_storms if s not in test_storm]
    random.seed(random_seed)
    val_storms = random.sample(available, min(num_val_storms, len(available)))
    print(f"Val storms: {len(val_storms)}, Test storms: {len(test_storm)}")

    train_ds = CycloneDataset(root_dir=root_dir, years=train_years, gridsat_type=gridsat_type,
        era5_type=era5_type, max_samples=max_samples, test_storm=test_storm,
        val_storms=val_storms, mode='train', img_size=img_size)
    val_ds = CycloneDataset(root_dir=root_dir, years=train_years, gridsat_type=gridsat_type,
        era5_type=era5_type, test_storm=test_storm, val_storms=val_storms, mode='val',
        img_size=img_size)
    test_ds = CycloneDataset(root_dir=root_dir, years=test_years, gridsat_type=gridsat_type,
        era5_type=era5_type, test_storm=test_storm, val_storms=val_storms, mode='test',
        img_size=img_size)

    print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ============================================================================
# METRICS
# ============================================================================

class MetricsCalculator:
    @staticmethod
    def mae(pred, target): return torch.mean(torch.abs(pred - target)).item()
    @staticmethod
    def mse(pred, target): return F.mse_loss(pred, target).item()
    @staticmethod
    def rmse(pred, target): return torch.sqrt(F.mse_loss(pred, target)).item()
    @staticmethod
    def psnr(pred, target, max_val=2.0):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return 100.0
        return (20 * torch.log10(torch.tensor(max_val, device=mse.device) / torch.sqrt(mse))).item()
    @staticmethod
    def ssim(pred, target, window_size=11):
        # Scale constants to data range L (SSIM standard: C1=(K1*L)^2, C2=(K2*L)^2)
        L = (target.max() - target.min()).item()
        L = max(L, 1e-8)
        C1, C2, sigma = (0.01 * L)**2, (0.03 * L)**2, 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        window = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(pred.device)
        ch = pred.size(1)
        window = window.expand(ch, 1, window_size, window_size).contiguous()
        pad = window_size // 2
        mu1 = F.conv2d(pred, window, padding=pad, groups=ch)
        mu2 = F.conv2d(target, window, padding=pad, groups=ch)
        s1  = F.conv2d(pred*pred,     window, padding=pad, groups=ch) - mu1*mu1
        s2  = F.conv2d(target*target, window, padding=pad, groups=ch) - mu2*mu2
        s12 = F.conv2d(pred*target,   window, padding=pad, groups=ch) - mu1*mu2
        ssim_map = ((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))
        return ssim_map.mean().item()


class FID(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity(); inception.eval()
        for p in inception.parameters(): p.requires_grad = False
        self.inception = inception.to(device)

    def preprocess(self, imgs):
        imgs = F.interpolate(imgs, size=(299,299), mode='bilinear', align_corners=False)
        if imgs.shape[1]==1: imgs = imgs.repeat(1,3,1,1)
        imgs = (imgs+1)/2.0
        m = torch.tensor([.485,.456,.406]).view(1,3,1,1).to(imgs.device)
        s = torch.tensor([.229,.224,.225]).view(1,3,1,1).to(imgs.device)
        return (imgs-m)/s

    @torch.no_grad()
    def get_features(self, imgs):
        self.inception.eval()
        f = self.inception(self.preprocess(imgs))
        if isinstance(f, tuple): f = f[0]
        if f.dim()>2: f = f.view(f.size(0),-1)
        return f.cpu().numpy()

    def calculate_fid(self, real_f, fake_f):
        mu_r, mu_f = np.mean(real_f, 0), np.mean(fake_f, 0)
        s_r, s_f = np.cov(real_f, rowvar=False), np.cov(fake_f, rowvar=False)
        diff = mu_r - mu_f
        cm, _ = linalg.sqrtm(s_r.dot(s_f), disp=False)
        if np.iscomplexobj(cm): cm = cm.real
        return float(diff.dot(diff) + np.trace(s_r + s_f - 2*cm))


# ============================================================================
# EVALUATION 
# ============================================================================

def evaluate_model(model, dataloader, scheduler, device,
                   num_samples=100, compute_advanced_metrics=True,
                   eval_seed=42):
    # Save RNG state so evaluation doesn't corrupt training randomness
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    np_rng_state = np.random.get_state()
    py_rng_state = random.getstate()
    # Seed for reproducible eval
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    model.eval(); calc = MetricsCalculator()
    metrics = {k: [] for k in ['mae','mse','rmse','psnr','ssim','fid']}
    fid_m = None; rf, ff = [], []
    if compute_advanced_metrics:
        try: fid_m = FID(device=device)
        except: pass

    total_seen = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if total_seen >= num_samples: break
            gs = batch['gridsat'].to(device); e5 = batch['era5'].to(device)
            tgt = batch['target'].to(device)
            ts = batch['timestamp']; bs = tgt.shape[0]
            total_seen += bs
            ec, ce = model.condition_encoder(gs, e5, ts)
            gen = torch.randn_like(tgt)
            step = max(1, scheduler.num_timesteps // 50)
            steps = list(range(0, scheduler.num_timesteps, step))[::-1]
            for si, tv in enumerate(steps):
                tb = torch.full((bs,), tv, device=device, dtype=torch.long)
                np_ = model(gen, gs, e5, ts, tb, encoded_condition=ec, context_emb=ce)
                tp = steps[si+1] if si+1<len(steps) else -1
                gen = scheduler.sample_prev_timestep_ddim(gen, np_, tv, t_prev=tp, eta=0.0)
            metrics['mae'].append(calc.mae(gen, tgt))
            metrics['mse'].append(calc.mse(gen, tgt))
            metrics['rmse'].append(calc.rmse(gen, tgt))
            metrics['psnr'].append(calc.psnr(gen, tgt))
            metrics['ssim'].append(calc.ssim(gen, tgt))
            if compute_advanced_metrics and fid_m:
                try: rf.append(fid_m.get_features(tgt)); ff.append(fid_m.get_features(gen))
                except: pass
    avg = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    if fid_m and rf and ff:
        try: avg['fid'] = fid_m.calculate_fid(np.concatenate(rf), np.concatenate(ff))
        except: pass
    print(f"  [Eval] MAE={avg['mae']:.4f} MSE={avg['mse']:.4f} PSNR={avg['psnr']:.2f} "
          f"SSIM={avg['ssim']:.4f} FID={avg['fid']:.4f}")
    # Restore RNG state
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    np.random.set_state(np_rng_state)
    random.setstate(py_rng_state)
    return avg


def save_comparison_images(model, dataloader, scheduler, device, save_dir, num_samples=8):
    model.eval(); save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        batch = next(iter(dataloader)); n = min(num_samples, len(batch['gridsat']))
        gs = batch['gridsat'][:n].to(device); e5 = batch['era5'][:n].to(device)
        tgt = batch['target'][:n].to(device)
        ts = batch['timestamp'][:n]
        ec, ce = model.condition_encoder(gs, e5, ts)
        gen = torch.randn_like(tgt)
        step = max(1, scheduler.num_timesteps // 50)
        steps = list(range(0, scheduler.num_timesteps, step))[::-1]
        for si, tv in enumerate(steps):
            tb = torch.full((n,), tv, device=device, dtype=torch.long)
            np_ = model(gen, gs, e5, ts, tb, encoded_condition=ec, context_emb=ce)
            tp = steps[si+1] if si+1<len(steps) else -1
            gen = scheduler.sample_prev_timestep_ddim(gen, np_, tv, t_prev=tp, eta=0.0)
        g_np, t_np, gen_np = gs.cpu().numpy(), tgt.cpu().numpy(), gen.cpu().numpy()
        fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
        if n==1: axes = axes.reshape(1,-1)
        for i in range(n):
            axes[i,0].imshow(g_np[i,0], cmap='gray', vmin=-1, vmax=1); axes[i,0].set_title('Input'); axes[i,0].axis('off')
            axes[i,1].imshow(t_np[i,0], cmap='gray', vmin=-1, vmax=1); axes[i,1].set_title('GT'); axes[i,1].axis('off')
            axes[i,2].imshow(gen_np[i,0], cmap='gray', vmin=-1, vmax=1); axes[i,2].set_title('Pred'); axes[i,2].axis('off')
        plt.tight_layout(); plt.savefig(save_dir/'comparison.png', dpi=150); plt.close()


def plot_training_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Train', alpha=0.7); plt.plot(val_losses, label='Val', alpha=0.7)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Curves'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ============================================================================
# DIFFUSION TRAINER
# ============================================================================

class DiffusionTrainer:
    """Train pixel-space DiffusionUNet with MSE(predicted_noise, true_noise)."""
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.device = torch.device(config['device'])
        self.checkpoint_dir = Path(config['checkpoint_dir']) / 'diffusion'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config['output_dir']) / 'diffusion'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.unet = DiffusionUNet(
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            base_channels=config['base_channels'],
            channel_mults=tuple(config['channel_mults']),
            t_emb_dim=config['t_emb_dim'],
            num_heads=config['num_heads'],
            gridsat_channels=1, era5_channels=4,
            img_size=config['img_size'],
            condition_embed_dim=config['embed_dim'],
        ).to(self.device)

        self.scheduler = LinearNoiseScheduler(
            config['num_timesteps'], config['beta_start'], config['beta_end'], self.device)

        self.optimizer = AdamW(self.unet.parameters(), lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer,
            T_max=config.get('diffusion_epochs', 100), eta_min=config.get('min_lr', 1e-6))

        self.train_losses, self.val_losses = [], []
        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.metrics_history = []
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.grad_accum = config.get('gradient_accumulation_steps', 1)

        # EMA for stable evaluation
        self.ema = EMA(self.unet, decay=config.get('ema_decay', 0.9999))

        if checkpoint_path: self._load_checkpoint(checkpoint_path)
        n_params = sum(p.numel() for p in self.unet.parameters())
        print(f"\nDiffusion Trainer initialized:")
        print(f"  Params: {n_params:,} | AMP: {self.use_amp} | GradAccum: {self.grad_accum}")
        print(f"  EMA decay: {config.get('ema_decay', 0.9999)}")
        print(f"  Resume epoch: {self.start_epoch}")

    def _load_checkpoint(self, path):
        print(f"Loading checkpoint: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.unet.load_state_dict(ckpt['unet_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.scaler and 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict']:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            if 'lr_scheduler_state_dict' in ckpt:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            if 'ema_state_dict' in ckpt:
                self.ema.load_state_dict(ckpt['ema_state_dict'])
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            ep = ckpt.get('epoch', 0); self.start_epoch = ep + 1
            print(f"  Resumed epoch {ep}, best_val={self.best_val_loss:.6f}")
        except Exception as e:
            print(f"  Checkpoint load failed: {e}"); self.start_epoch = 1

    def train_epoch(self, dataloader, epoch):
        self.unet.train()
        total, n = 0.0, 0
        debug = (epoch <= 2) or (epoch % 5 == 0)
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for bi, batch in enumerate(pbar):
            gs  = batch['gridsat'].to(self.device)
            e5  = batch['era5'].to(self.device)
            tgt = batch['target'].to(self.device)
            ts  = batch['timestamp']
            bs  = tgt.shape[0]
            t     = torch.randint(0, self.scheduler.num_timesteps, (bs,), device=self.device)
            noise = torch.randn_like(tgt)
            noisy = self.scheduler.add_noise(tgt, noise, t)
            if bi % self.grad_accum == 0: self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred = self.unet(noisy, gs, e5, ts, t)
                loss = F.mse_loss(pred, noise) / self.grad_accum
            if self.use_amp: self.scaler.scale(loss).backward()
            else: loss.backward()
            if (bi + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    self.scaler.step(self.optimizer); self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    self.optimizer.step()
                self.ema.update(self.unet)  # update EMA after each optimizer step
            raw = loss.item() * self.grad_accum; total += raw; n += 1
            if bi % 50 == 0: torch.cuda.empty_cache()
            if n == 1 and debug:
                print(f"\n  [Debug] tgt shape={tgt.shape} range=[{tgt.min():.3f},{tgt.max():.3f}]")
                print(f"  [Debug] pred mean={pred.mean():.4f} std={pred.std():.4f} loss={raw:.6f}")
            pbar.set_postfix({'loss': f'{raw:.4f}'})
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, dataloader):
        self.unet.eval(); total, n = 0.0, 0
        for batch in tqdm(dataloader, desc='Val'):
            gs  = batch['gridsat'].to(self.device)
            e5  = batch['era5'].to(self.device)
            tgt = batch['target'].to(self.device)
            ts  = batch['timestamp']; bs = tgt.shape[0]
            t     = torch.randint(0, self.scheduler.num_timesteps, (bs,), device=self.device)
            noise = torch.randn_like(tgt)
            noisy = self.scheduler.add_noise(tgt, noise, t)
            pred  = self.unet(noisy, gs, e5, ts, t)
            total += F.mse_loss(pred, noise).item(); n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def sample(self, gs, e5, ts, num_steps=50):
        self.unet.eval(); bs = gs.shape[0]; sz = self.config['img_size']
        x = torch.randn(bs, 1, sz, sz, device=self.device)
        ec, ce = self.unet.condition_encoder(gs, e5, ts)
        step  = max(1, self.scheduler.num_timesteps // num_steps)
        steps = list(range(0, self.scheduler.num_timesteps, step))[::-1]
        for i, tv in enumerate(steps):
            tb  = torch.full((bs,), tv, device=self.device, dtype=torch.long)
            np_ = self.unet(x, gs, e5, ts, tb, encoded_condition=ec, context_emb=ce)
            tp  = steps[i+1] if i+1 < len(steps) else -1
            x   = self.scheduler.sample_prev_timestep_ddim(x, np_, tv, t_prev=tp, eta=0.0)
        return x

    def save_checkpoint(self, epoch, val_loss):
        is_best = val_loss < self.best_val_loss
        if is_best: self.best_val_loss = val_loss
        ckpt = {'epoch': epoch, 'unet_state_dict': self.unet.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'ema_state_dict': self.ema.state_dict(),
                'best_val_loss': self.best_val_loss, 'config': self.config}
        torch.save(ckpt, self.checkpoint_dir / 'diffusion_latest.pt')
        if is_best:
            # Save best checkpoint with EMA weights
            self.ema.apply(self.unet)
            ema_ckpt = dict(ckpt)
            ema_ckpt['unet_state_dict'] = self.unet.state_dict()
            torch.save(ema_ckpt, self.checkpoint_dir / 'diffusion_best.pt')
            self.ema.restore(self.unet)
            print(f"  ** New best (val_loss={val_loss:.6f})")
        if epoch % 10 == 0:
            torch.save(ckpt, self.checkpoint_dir / f'diffusion_epoch_{epoch}.pt')

    def save_samples(self, dataloader, epoch):
        self.unet.eval()
        batch = next(iter(dataloader)); n = min(8, len(batch['gridsat']))
        gs  = batch['gridsat'][:n].to(self.device)
        e5  = batch['era5'][:n].to(self.device)
        tgt = batch['target'][:n].to(self.device)
        ts  = batch['timestamp'][:n]
        with torch.no_grad(): gen = self.sample(gs, e5, ts)
        g_np, t_np, gen_np = gs.cpu().numpy(), tgt.cpu().numpy(), gen.cpu().numpy()
        fig, axes = plt.subplots(3, n, figsize=(2*n, 6))
        if n == 1: axes = axes.reshape(3, 1)
        for i in range(n):
            axes[0,i].imshow(g_np[i,0], cmap='viridis'); axes[0,i].axis('off'); axes[0,i].set_title('Input')
            axes[1,i].imshow(t_np[i,0], cmap='viridis'); axes[1,i].axis('off'); axes[1,i].set_title('Target')
            axes[2,i].imshow(gen_np[i,0], cmap='viridis'); axes[2,i].axis('off'); axes[2,i].set_title('Gen')
        plt.tight_layout(); plt.savefig(self.output_dir / f'samples_epoch_{epoch}.png', dpi=150); plt.close()

    def train(self, train_loader, val_loader):
        num_epochs   = self.config.get('diffusion_epochs', 100)
        eval_samples = self.config.get('eval_samples', 50)
        print(f"Epochs: {num_epochs} (from {self.start_epoch})\n{'='*60}\n")

        for epoch in range(self.start_epoch, num_epochs + 1):
            print(f"\n{'='*50}\nEpoch {epoch}/{num_epochs}\n{'='*50}")
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            print(f"  Train: {train_loss:.6f}")
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            print(f"  Val:   {val_loss:.6f}")
            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            print(f"  LR: {lr:.1e}")

            eval_metrics = None
            try:
                print(f"\n  Evaluating ({eval_samples} samples) with EMA weights...")
                self.ema.apply(self.unet)
                eval_metrics = evaluate_model(self.unet, val_loader, self.scheduler,
                    self.device, num_samples=eval_samples, compute_advanced_metrics=True)
                self.ema.restore(self.unet)
            except Exception as e:
                self.ema.restore(self.unet)  # always restore even on error
                print(f"  Eval failed: {e}")

            row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr}
            if eval_metrics:
                for k, v in eval_metrics.items(): row[f'eval_{k}'] = v
            self.metrics_history.append(row)
            pd.DataFrame(self.metrics_history).to_csv(
                self.metrics_dir / 'training_metrics.csv', index=False)
            self.save_checkpoint(epoch, val_loss)
            if epoch % 10 == 0 or (epoch > num_epochs//2 and epoch % 5 == 0):
                self.ema.apply(self.unet)
                self.save_samples(val_loader, epoch)
                self.ema.restore(self.unet)

        print(f"\n{'='*60}\nFinal eval (EMA weights)...")
        try:
            self.ema.apply(self.unet)
            final = evaluate_model(self.unet, val_loader, self.scheduler, self.device,
                num_samples=min(eval_samples*2, 200), compute_advanced_metrics=True)
            with open(self.output_dir / 'final_metrics.json', 'w') as f: json.dump(final, f, indent=4)
            self.ema.restore(self.unet)
        except Exception as e:
            self.ema.restore(self.unet)
            print(f"Final eval failed: {e}")
        print(f"\n{'='*60}\nTRAINING COMPLETE | Best val: {self.best_val_loss:.6f}")
        print(f"Checkpoints: {self.checkpoint_dir}\n{'='*60}\n")
        pd.DataFrame(self.metrics_history).to_csv(self.metrics_dir / 'final_metrics.csv', index=False)
        try:
            cfg = {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()}
            with open(self.output_dir / 'config.json', 'w') as f: json.dump(cfg, f, indent=4, default=str)
        except: pass
        plot_training_curves(self.train_losses, self.val_losses, self.output_dir / 'training_curves.png')


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = {
        'root_dir': r'/kaggle/input/datasets/meheruzannat/cyclone-dataset',
        'train_years': ['2006','2007','2008','2009','2010','2011','2012'],
        'test_storm': ['2022349N13068'],
        'num_val_storms': 35,
        'batch_size': 4,
        'num_workers': 4,
        'img_size': 64,
        'in_channels': 1,
        'out_channels': 1,
        'base_channels': 64,
        'channel_mults': (1, 2, 4),
        'num_heads': 4,
        'embed_dim': 64,
        't_emb_dim': 128,
        'num_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'diffusion_epochs': 50,
        'learning_rate': 2e-4,
        'min_lr': 1e-5,
        'weight_decay': 1e-4,
        'dropout': 0.1,
        'ema_decay': 0.9999,
        'use_amp': True,
        'gradient_accumulation_steps': 4,
        'eval_samples': 200,
        'checkpoint_dir': './checkpoints',
        'output_dir': './outputs',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=config['root_dir'], batch_size=config['batch_size'],
        num_workers=config['num_workers'], train_years=config['train_years'],
        test_storm=config['test_storm'], img_size=config['img_size'],
        num_val_storms=config['num_val_storms'])

    checkpoint_path = None  # set to path string to resume
    trainer = DiffusionTrainer(config, checkpoint_path=checkpoint_path)
    trainer.train(train_loader, val_loader)
    print("\nDONE!")


if __name__ == '__main__':
    main()