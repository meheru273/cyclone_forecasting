"""
Refinement-GAN: Stage 3 of the Satellite Diffusion Model pipeline.

Architecture (from diagram):
    Ground Truth ─────────────────────────────────────────────────┐
                                                                  │ (real)
    Input ──▶ [Frozen VAE + Diffusion UNet] ──▶ Generated ──▶ RefinementNetwork ──▶ Refined
                                                                  │
                                                         PatchGAN Discriminator

    Loss = L1(refined, ground_truth) + λ_adv * adversarial_loss

All pipeline components are defined directly in this file (fully self-contained).
"""

import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================================
# DATASET
# ============================================================================

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
        self.test_storm = test_storm if isinstance(test_storm, list) else ([test_storm] if test_storm else [])
        self.val_storms = val_storms if isinstance(val_storms, list) else ([val_storms] if val_storms else [])
        self.mode = mode
        self._load_samples(years, max_samples)

    def _load_samples(self, years, max_samples):
        for year_folder in years:
            year_path = self.root_dir / year_folder
            if not year_path.exists():
                continue
            nested = year_path / year_folder
            if nested.exists():
                year_path = nested
            for cyclone_folder in year_path.iterdir():
                if not cyclone_folder.is_dir():
                    continue
                storm_name = cyclone_folder.name
                if self.mode == 'train' and (storm_name in self.test_storm or storm_name in self.val_storms):
                    continue
                if self.mode == 'val' and storm_name not in self.val_storms:
                    continue
                if self.mode == 'test' and storm_name not in self.test_storm:
                    continue
                timestamps = sorted([t.name for t in cyclone_folder.iterdir() if t.is_dir()])
                for i in range(len(timestamps) - 1):
                    cur_ts, nxt_ts = timestamps[i], timestamps[i + 1]
                    cg = cyclone_folder / cur_ts / self.gridsat_type
                    ce = cyclone_folder / cur_ts / self.era5_type
                    ng = cyclone_folder / nxt_ts / self.gridsat_type
                    if cg.exists() and ce.exists() and ng.exists():
                        self.samples.append({'current_gridsat': str(cg), 'current_era5': str(ce),
                                             'next_gridsat': str(ng), 'timestamp': cur_ts,
                                             'storm_name': storm_name, 'year': year_folder})
                        if max_samples and len(self.samples) >= max_samples:
                            return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cg = np.nan_to_num(np.load(s['current_gridsat']).astype(np.float32), nan=0.0)
        ce = np.nan_to_num(np.load(s['current_era5']).astype(np.float32),    nan=0.0)
        ng = np.nan_to_num(np.load(s['next_gridsat']).astype(np.float32),    nan=0.0)

        cg = cg[0:1]  if cg.ndim == 3 else cg[np.newaxis]
        ng = ng[0:1]  if ng.ndim == 3 else ng[np.newaxis]
        ce = ce[:4]   if ce.ndim == 3 else ce[np.newaxis]

        cg = torch.from_numpy(cg)
        ce = torch.from_numpy(ce)
        ng = torch.from_numpy(ng)

        sz = (self.img_size, self.img_size)
        cg = F.interpolate(cg.unsqueeze(0), size=sz, mode='bilinear', align_corners=False).squeeze(0)
        ce = F.interpolate(ce.unsqueeze(0), size=sz, mode='bilinear', align_corners=False).squeeze(0)
        ng = F.interpolate(ng.unsqueeze(0), size=sz, mode='bilinear', align_corners=False).squeeze(0)

        eps = 1e-8
        mn, mx = cg.min(), cg.max()
        if mx - mn > eps:
            cg = (cg - mn) / (mx - mn) * 2 - 1
        for c in range(ce.shape[0]):
            mn, mx = ce[c].min(), ce[c].max()
            if mx - mn > eps:
                ce[c] = (ce[c] - mn) / (mx - mn) * 2 - 1
        mn, mx = ng.min(), ng.max()
        if mx - mn > eps:
            ng = (ng - mn) / (mx - mn) * 2 - 1

        return {'gridsat': cg, 'era5': ce, 'target': ng,
                'timestamp': s['timestamp'], 'storm_name': s['storm_name'], 'year': s['year']}


def get_all_storms(root_dir, years):
    root_dir = Path(root_dir)
    storms = []
    for year_folder in years:
        year_path = root_dir / year_folder
        if not year_path.exists():
            continue
        nested = year_path / year_folder
        if nested.exists():
            year_path = nested
        for cf in year_path.iterdir():
            if cf.is_dir():
                storms.append(cf.name)
    return storms


def get_dataloaders(root_dir, batch_size=4, num_workers=2,
                    train_years=None, test_years=None,
                    gridsat_type='GRIDSAT_data.npy', era5_type='ERA5_data.npy',
                    max_samples=None, test_storm=None,
                    num_val_storms=5, img_size=256, random_seed=42):
    if train_years is None:
        train_years = ['2007', '2008', '2009', '2010', '2011']
    if test_years is None:
        test_years = ['2011']
    if test_storm is None:
        test_storm = ['2022349N13068']

    all_storms = get_all_storms(root_dir, train_years)
    available  = [s for s in all_storms if s not in test_storm]
    random.seed(random_seed)
    val_storms = random.sample(available, min(num_val_storms, len(available)))
    print(f"Val storms: {len(val_storms)}, Test storms: {len(test_storm)}")

    kw = dict(gridsat_type=gridsat_type, era5_type=era5_type,
              test_storm=test_storm, val_storms=val_storms, img_size=img_size)
    train_ds = CycloneDataset(root_dir, train_years, max_samples=max_samples, mode='train', **kw)
    val_ds   = CycloneDataset(root_dir, train_years, mode='val',  **kw)
    test_ds  = CycloneDataset(root_dir, test_years,  mode='test', **kw)
    print(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ============================================================================
# VAE
# ============================================================================

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4, base_channels=64):
        super().__init__()
        bc = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bc, 4, 2, 1), nn.GroupNorm(8, bc), nn.SiLU(),
            nn.Conv2d(bc, bc*2, 4, 2, 1),        nn.GroupNorm(8, bc*2), nn.SiLU(),
            nn.Conv2d(bc*2, bc*2, 3, 1, 1),      nn.GroupNorm(8, bc*2), nn.SiLU(),
        )
        self.fc_mu     = nn.Conv2d(bc*2, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(bc*2, latent_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=4, out_channels=1, base_channels=64):
        super().__init__()
        bc = base_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, bc*2, 1), nn.GroupNorm(8, bc*2), nn.SiLU(),
            nn.ConvTranspose2d(bc*2, bc, 4, 2, 1), nn.GroupNorm(8, bc), nn.SiLU(),
            nn.ConvTranspose2d(bc,   bc, 4, 2, 1), nn.GroupNorm(8, bc), nn.SiLU(),
            nn.Conv2d(bc, out_channels, 3, 1, 1),  nn.Tanh(),
        )

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4, base_channels=64):
        super().__init__()
        self.encoder    = VAEEncoder(in_channels, latent_dim, base_channels)
        self.decoder    = VAEDecoder(latent_dim, in_channels, base_channels)
        self.latent_dim = latent_dim

    def encode(self, x):         return self.encoder(x)
    def decode(self, z):         return self.decoder(z)
    def reparameterize(self, mu, logvar): return self.encoder.reparameterize(mu, logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


# ============================================================================
# CONDITION ENCODING
# ============================================================================

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        pe = torch.zeros(channels, height, width)
        y  = torch.arange(height, dtype=torch.float32).unsqueeze(1).repeat(1, width)
        x  = torch.arange(width,  dtype=torch.float32).unsqueeze(0).repeat(height, 1)
        div = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * -(math.log(10000.0) / channels))
        for i in range(0, channels, 4):
            idx = i // 2
            if idx < len(div):
                pe[i] = torch.sin(y * div[idx])
                if i+1 < channels: pe[i+1] = torch.cos(y * div[idx])
                if i+2 < channels: pe[i+2] = torch.sin(x * div[idx])
                if i+3 < channels: pe[i+3] = torch.cos(x * div[idx])
        self.register_buffer('pos_encoding', pe)
        self.height, self.width = height, width

    def forward(self, x):
        _, _, h, w = x.shape
        pos = self.pos_encoding
        if h != self.height or w != self.width:
            pos = F.interpolate(pos.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        return x + pos.unsqueeze(0)


class TemporalEncoding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        q = embed_dim // 4
        self.year_embed  = nn.Embedding(50, q)
        self.month_embed = nn.Embedding(12, q)
        self.day_embed   = nn.Embedding(31, q)
        self.hour_embed  = nn.Embedding(24, q)
        self.projection  = nn.Linear(embed_dim, embed_dim)

    def _parse(self, ts):
        ts = ts.split('.')[0]
        dt = datetime.strptime(ts, "%Y-%m-%d %H_%M_%S")
        return dt.year - 2000, dt.month - 1, dt.day - 1, dt.hour

    def forward(self, timestamps):
        if isinstance(timestamps[0], str):
            p = [self._parse(t) for t in timestamps]
            years  = torch.tensor([x[0] for x in p], dtype=torch.long)
            months = torch.tensor([x[1] for x in p], dtype=torch.long)
            days   = torch.tensor([x[2] for x in p], dtype=torch.long)
            hours  = torch.tensor([x[3] for x in p], dtype=torch.long)
        else:
            years, months, days, hours = timestamps
        dev = self.year_embed.weight.device
        emb = torch.cat([self.year_embed(years.to(dev)),  self.month_embed(months.to(dev)),
                         self.day_embed(days.to(dev)),    self.hour_embed(hours.to(dev))], dim=-1)
        return self.projection(emb)


class StormEncoding(nn.Module):
    def __init__(self, max_storms=1000, embed_dim=128):
        super().__init__()
        self.storm_embed = nn.Embedding(max_storms, embed_dim)
        self._map, self._next = {}, 0

    def _idx(self, name):
        if name not in self._map:
            self._map[name] = self._next; self._next += 1
        return self._map[name]

    def forward(self, storm_names):
        if isinstance(storm_names[0], str):
            indices = torch.tensor([self._idx(n) for n in storm_names], dtype=torch.long)
        else:
            indices = storm_names
        return self.storm_embed(indices.to(self.storm_embed.weight.device))


class ConditionEncoder(nn.Module):
    def __init__(self, gridsat_channels=1, era5_channels=4, img_size=256,
                 embed_dim=128, output_channels=64, dropout=0.1):
        super().__init__()
        self.output_channels = output_channels
        self.spatial_pos_gridsat = SpatialPositionalEncoding(gridsat_channels, img_size, img_size)
        self.spatial_pos_era5 = SpatialPositionalEncoding(era5_channels,    img_size, img_size)
        self.temporal_encoder = TemporalEncoding(embed_dim)
        self.storm_encoder    = StormEncoding(embed_dim=embed_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim)
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(gridsat_channels+era5_channels, output_channels, 3, 1, 1),
            nn.GroupNorm(8, output_channels), nn.SiLU(), nn.Dropout2d(dropout),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1),
        )
        self.context_to_spatial = nn.Sequential(nn.Linear(embed_dim, output_channels), nn.SiLU())

    def forward(self, gridsat, era5, timestamps, storm_names):
        B, _, H, W = gridsat.shape
        feats = self.image_encoder(torch.cat([self.spatial_pos_gridsat(gridsat), self.spatial_pos_era5(era5)], dim=1))
        ctx   = self.context_projection(torch.cat([self.temporal_encoder(timestamps),
                                             self.storm_encoder(storm_names)], dim=-1))
        spatial = F.interpolate(self.context_to_spatial(ctx).view(B, -1, 1, 1),
                                size=(H, W), mode='bilinear', align_corners=False)
        return feats + spatial, ctx


# ============================================================================
# NOISE SCHEDULER
# ============================================================================

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        acp    = torch.cumprod(alphas, dim=0)
        self.betas      = betas
        self.alphas     = alphas
        self.alpha_cumprod              = acp
        self.sqrt_alpha_cumprod         = torch.sqrt(acp)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - acp)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), acp[:-1]])

    def to(self, device):
        self.device = device
        for attr in ('betas','alphas','alpha_cumprod','alpha_cumprod_prev',
                     'sqrt_alpha_cumprod','sqrt_one_minus_alpha_cumprod'):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(self, original, noise, t):
        B = original.shape[0]
        dev = original.device
        sa  = self.sqrt_alpha_cumprod.to(dev)[t].reshape(B)
        s1m = self.sqrt_one_minus_alpha_cumprod.to(dev)[t].reshape(B)
        for _ in range(original.dim() - 1):
            sa = sa.unsqueeze(-1); s1m = s1m.unsqueeze(-1)
        return sa * original + s1m * noise

    def sample_prev_timestep_ddim(self, xt, noise_pred, t, t_prev=None, eta=0.0):
        dev = xt.device
        if t_prev is None: t_prev = t - 1
        acp_t  = self.alpha_cumprod.to(dev)[t]
        sqt    = self.sqrt_alpha_cumprod.to(dev)[t]
        s1mt   = self.sqrt_one_minus_alpha_cumprod.to(dev)[t]
        acp_p  = self.alpha_cumprod.to(dev)[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=dev)
        x0     = torch.clamp((xt - s1mt * noise_pred) / sqt, -3.0, 3.0)
        sq_p   = torch.sqrt(acp_p)
        sigma  = (eta * torch.sqrt((1-acp_p)/(1-acp_t)) * torch.sqrt(1-acp_t/acp_p)
                  if eta > 0 and (1-acp_t) > 1e-8 else torch.tensor(0.0, device=dev))
        x_prev = sq_p * x0 + torch.sqrt(torch.clamp(1-acp_p-sigma**2, min=0.0)) * noise_pred
        if eta > 0 and t > 0:
            x_prev = x_prev + sigma * torch.randn_like(xt)
        return x_prev


# ============================================================================
# UNET FOR DIFFUSION
# ============================================================================

def get_time_embeddings(time_steps, t_emb_dim):
    factor = 10000 ** (torch.arange(0, t_emb_dim//2, device=time_steps.device) / (t_emb_dim//2))
    t_emb  = time_steps[:, None].repeat(1, t_emb_dim//2) / factor
    return torch.cat((torch.sin(t_emb), torch.cos(t_emb)), dim=-1)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.resnet_conv1 = nn.Sequential(nn.GroupNorm(8, in_channels), nn.SiLU(),
                                          nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.t_emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
        self.resnet_conv2 = nn.Sequential(nn.GroupNorm(8, out_channels), nn.SiLU(),
                                          nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention      = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv  = nn.Conv2d(in_channels, out_channels, 1)
        self.down_sample_conv      = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = self.resnet_conv1(x) + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv2(out) + self.residual_input_conv(x)
        B, C, H, W = out.shape
        ai = self.attention_norm(out).permute(0,2,3,1).reshape(B, H*W, C)
        ao, _ = self.attention(ai, ai, ai)
        out = out + ao.reshape(B, H, W, C).permute(0,3,1,2)
        return self.down_sample_conv(out), out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        self.resnet_conv1 = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, out_channels, 3,1,1))
                                 for c in (in_channels, out_channels)])
        self.t_emb_layers  = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(2)])
        self.resnet_conv2 = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, out_channels), nn.SiLU(),
                                               nn.Conv2d(out_channels, out_channels, 3,1,1)) for _ in range(2)])
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention      = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv      = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in (in_channels, out_channels)])

    def forward(self, x, t_emb):
        for i in range(2):
            res = x
            x   = self.resnet_conv1[i](x) + self.t_emb_layers[i](t_emb)[:,:,None,None]
            x   = self.resnet_conv2[i](x) + self.residual_input_conv[i](res)
            if i == 0:
                B,C,H,W = x.shape
                ai = self.attention_norm(x).permute(0,2,3,1).reshape(B,H*W,C)
                ao,_ = self.attention(ai,ai,ai)
                x = x + ao.reshape(B,H,W,C).permute(0,3,1,2)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads, skip_channels=None):
        super().__init__()
        self.up_sample = up_sample
        actual_in = in_channels + (skip_channels or 0)
        self.resnet_conv1 = nn.Sequential(nn.GroupNorm(8, actual_in), nn.SiLU(),
                                          nn.Conv2d(actual_in, out_channels, 3,1,1))
        self.t_emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
        self.resnet_conv2 = nn.Sequential(nn.GroupNorm(8, out_channels), nn.SiLU(),
                                          nn.Conv2d(out_channels, out_channels, 3,1,1))
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention      = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv  = nn.Conv2d(actual_in, out_channels, 1)

    def forward(self, x, t_emb, skip=None):
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        out = self.resnet_conv1(x) + self.t_emb_layers(t_emb)[:,:,None,None]
        out = self.resnet_conv2(out) + self.residual_input_conv(x)
        B, C, H, W = out.shape
        ai = self.attention_norm(out).permute(0,2,3,1).reshape(B,H*W,C)
        ao,_ = self.attention(ai,ai,ai)
        return out + ao.reshape(B,H,W,C).permute(0,3,1,2)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 channel_mults=(1,2,4,8), t_emb_dim=128, num_heads=4,
                 gridsat_channels=1, era5_channels=4, img_size=64,
                 condition_embed_dim=128, condition_img_size=256):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.condition_encoder = ConditionEncoder(
            gridsat_channels=gridsat_channels, era5_channels=era5_channels,
            img_size=condition_img_size, embed_dim=condition_embed_dim, output_channels=base_channels)
        self.time_embed     = nn.Sequential(nn.Linear(t_emb_dim, t_emb_dim*4), nn.SiLU(),
                                            nn.Linear(t_emb_dim*4, t_emb_dim*4))
        self.combined_embed = nn.Sequential(nn.Linear(t_emb_dim*4+condition_embed_dim, t_emb_dim*4), nn.SiLU(),
                                            nn.Linear(t_emb_dim*4, t_emb_dim*4))
        self.init_conv      = nn.Conv2d(base_channels+in_channels, base_channels, 3, 1, 1)
        self.encoders = nn.ModuleList()
        curr = base_channels
        for i, m in enumerate(channel_mults):
            oc = base_channels * m
            self.encoders.append(DownBlock(curr, oc, t_emb_dim*4, down_sample=(i<len(channel_mults)-1), num_heads=num_heads))
            curr = oc
        self.mid_block = MidBlock(curr, curr, t_emb_dim*4, num_heads)
        enc_ch = [base_channels*m for m in channel_mults]
        self.decoders = nn.ModuleList()
        for di, m in enumerate(list(reversed(channel_mults))[1:]):
            oc = base_channels * m
            si = len(channel_mults)-2-di
            self.decoders.append(UpBlock(curr, oc, t_emb_dim*4, up_sample=True, num_heads=num_heads,
                                         skip_channels=enc_ch[si] if si>=0 else None))
            curr = oc
        self.final_conv = nn.Sequential(nn.GroupNorm(8, base_channels), nn.SiLU(),
                                        nn.Conv2d(base_channels, out_channels, 3, 1, 1))

    def forward(self, noisy_latent, gridsat, era5, timestamps, storm_names, t,
                encoded_condition=None, context_emb=None):
        if encoded_condition is None or context_emb is None:
            encoded_condition, context_emb = self.condition_encoder(gridsat, era5, timestamps, storm_names)
        x    = torch.cat([encoded_condition, noisy_latent], dim=1)
        temb = self.combined_embed(torch.cat([self.time_embed(get_time_embeddings(t, self.t_emb_dim)), context_emb], dim=-1))
        x    = self.init_conv(x)
        skips = []
        for enc in self.encoders:
            x, sk = enc(x, temb); skips.append(sk)
        x = self.mid_block(x, temb)
        for i, dec in enumerate(self.decoders):
            si = len(skips)-2-i
            x  = dec(x, temb, skips[si] if si>=0 else None)
        return self.final_conv(x)



# ============================================================================
# REFINEMENT NETWORK (Generator)
# ============================================================================

class ResidualBlock(nn.Module):
    """Two-conv residual block with GroupNorm + SiLU."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class RefinementNetwork(nn.Module):
    """
    U-Net style generator for image-to-image refinement.

    Uses residual learning:  refined = generated + network(generated)
    so the network only has to learn the correction delta.

    Input:  1-channel generated image  (B, 1, 256, 256)
    Output: 1-channel refined image    (B, 1, 256, 256)
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        bc = base_channels

        # ----- Encoder (3 down-blocks) -----
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, bc, 3, 1, 1),
            nn.GroupNorm(8, bc),
            nn.SiLU(),
            ResidualBlock(bc),
        )
        self.down1 = nn.Conv2d(bc, bc, 4, 2, 1)          # 256 -> 128

        self.enc2 = nn.Sequential(
            nn.Conv2d(bc, bc * 2, 3, 1, 1),
            nn.GroupNorm(8, bc * 2),
            nn.SiLU(),
            ResidualBlock(bc * 2),
        )
        self.down2 = nn.Conv2d(bc * 2, bc * 2, 4, 2, 1)  # 128 -> 64

        self.enc3 = nn.Sequential(
            nn.Conv2d(bc * 2, bc * 4, 3, 1, 1),
            nn.GroupNorm(8, bc * 4),
            nn.SiLU(),
            ResidualBlock(bc * 4),
        )
        self.down3 = nn.Conv2d(bc * 4, bc * 4, 4, 2, 1)  # 64 -> 32

        # ----- Bottleneck -----
        self.bottleneck = nn.Sequential(
            ResidualBlock(bc * 4),
            ResidualBlock(bc * 4),
        )

        # ----- Decoder (3 up-blocks with skip connections) -----
        self.up3 = nn.ConvTranspose2d(bc * 4, bc * 4, 4, 2, 1)  # 32 -> 64
        self.dec3 = nn.Sequential(
            nn.Conv2d(bc * 4 + bc * 4, bc * 4, 3, 1, 1),  # skip concat
            nn.GroupNorm(8, bc * 4),
            nn.SiLU(),
            ResidualBlock(bc * 4),
        )

        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, 4, 2, 1)  # 64 -> 128
        self.dec2 = nn.Sequential(
            nn.Conv2d(bc * 2 + bc * 2, bc * 2, 3, 1, 1),  # skip concat
            nn.GroupNorm(8, bc * 2),
            nn.SiLU(),
            ResidualBlock(bc * 2),
        )

        self.up1 = nn.ConvTranspose2d(bc * 2, bc, 4, 2, 1)       # 128 -> 256
        self.dec1 = nn.Sequential(
            nn.Conv2d(bc + bc, bc, 3, 1, 1),              # skip concat
            nn.GroupNorm(8, bc),
            nn.SiLU(),
            ResidualBlock(bc),
        )

        # Final projection to 1-channel residual
        self.final = nn.Sequential(
            nn.Conv2d(bc, bc, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(bc, in_channels, 1),
            nn.Tanh(),  # residual bounded to [-1, 1]
        )

    def forward(self, x):
        """
        Args:
            x: generated image (B, 1, H, W)
        Returns:
            refined image (B, 1, H, W) = x + learned_residual
        """
        # Encoder
        e1 = self.enc1(x)               # (B, bc, 256, 256)
        e2 = self.enc2(self.down1(e1))   # (B, bc*2, 128, 128)
        e3 = self.enc3(self.down2(e2))   # (B, bc*4, 64, 64)

        # Bottleneck
        b = self.bottleneck(self.down3(e3))  # (B, bc*4, 32, 32)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # (B, bc*4, 64, 64)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, bc*2, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, bc, 256, 256)

        residual = self.final(d1)  # (B, 1, 256, 256)

        # Residual learning: output = input + delta
        refined = x + residual
        return refined


# ============================================================================
# PATCHGAN DISCRIMINATOR
# ============================================================================

class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for refinement quality assessment."""
    def __init__(self, in_channels=2, base_channels=64, n_layers=3):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf = base_channels
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, base_channels * 8)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, 2, 1),
                nn.GroupNorm(8, nf),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers += [
            nn.Conv2d(nf, nf, 4, 1, 1),
            nn.GroupNorm(8, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, 4, 1, 1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, condition, target):
        x = torch.cat([condition, target], dim=1)
        return self.model(x)


# ============================================================================
# REFINEMENT-GAN TRAINER
# ============================================================================

class RefinementGANTrainer:
    """
    Stage 3: Train a Refinement Network (generator) + PatchGAN Discriminator
    on diffusion outputs vs ground truth.

    Network 1 (Frozen VAE + Diffusion UNet) produces generated images.
    Network 2 (RefinementNetwork + Discriminator) is trained to refine them.

    Generator loss:       L = L1(refined, GT) + λ_adv * hinge_gen
    Discriminator loss:   hinge loss on real vs refined
    """

    def __init__(self, config, vae_checkpoint_path, diffusion_checkpoint_path,
                 refinement_checkpoint_path=None):
        self.config = config
        self.device = torch.device(config['device'])

        # Directories
        self.checkpoint_dir = Path(config['checkpoint_dir']) / 'refinement_gan'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = Path(config['output_dir']) / 'refinement_gan'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # LOAD AND FREEZE VAE
        # =====================================================================
        print(f"\nLoading pretrained VAE from: {vae_checkpoint_path}")
        vae_ckpt = torch.load(vae_checkpoint_path, map_location=self.device)

        self.vae = VAE(
            in_channels=1,
            latent_dim=config.get('latent_dim', 4),
            base_channels=config.get('vae_base_channels', 64)
        ).to(self.device)
        self.vae.load_state_dict(vae_ckpt['vae_state_dict'])
        self.latent_scale_factor = vae_ckpt.get('latent_scale_factor', 0.18215)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        print(f"  ✓ VAE loaded and FROZEN (scale factor: {self.latent_scale_factor})")

        # =====================================================================
        # LOAD AND FREEZE DIFFUSION UNET + CONDITION DOWNSAMPLE
        # =====================================================================
        latent_size = config['img_size'] // 4
        print(f"\nLoading pretrained Diffusion UNet from: {diffusion_checkpoint_path}")
        diff_ckpt = torch.load(diffusion_checkpoint_path, map_location=self.device)

        self.unet = ConditionalUNet(
            in_channels=config.get('latent_dim', 4),
            out_channels=config.get('latent_dim', 4),
            base_channels=config['base_channels'],
            channel_mults=config['channel_mults'],
            t_emb_dim=config['t_emb_dim'],
            num_heads=config['num_heads'],
            gridsat_channels=1,
            era5_channels=4,
            img_size=latent_size,
            condition_embed_dim=config['embed_dim'],
            condition_img_size=config['img_size']
        ).to(self.device)
        self.unet.load_state_dict(diff_ckpt['unet_state_dict'])
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

        self.condition_downsample = nn.Sequential(
            nn.Conv2d(config['base_channels'], config['base_channels'], 3, 2, 1),
            nn.GroupNorm(8, config['base_channels']),
            nn.SiLU(),
            nn.Conv2d(config['base_channels'], config['base_channels'], 3, 2, 1),
            nn.GroupNorm(8, config['base_channels']),
            nn.SiLU(),
        ).to(self.device)
        self.condition_downsample.load_state_dict(diff_ckpt['condition_downsample_state_dict'])
        self.condition_downsample.eval()
        for p in self.condition_downsample.parameters():
            p.requires_grad = False

        print(f"  ✓ UNet and condition_downsample loaded and FROZEN")

        # Noise scheduler (for DDIM sampling)
        self.scheduler = LinearNoiseScheduler(
            num_timesteps=config['num_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            device=self.device
        )

        # =====================================================================
        # TRAINABLE NETWORKS
        # =====================================================================
        # Refinement Network (Generator)
        self.generator = RefinementNetwork(
            in_channels=1,
            base_channels=config.get('refine_base_channels', 64)
        ).to(self.device)

        # PatchGAN Discriminator
        self.discriminator = PatchGANDiscriminator(
            in_channels=2,  # condition (gridsat) + target/refined
            base_channels=config.get('disc_base_channels', 64),
            n_layers=config.get('disc_n_layers', 3)
        ).to(self.device)

        # =====================================================================
        # OPTIMIZERS
        # =====================================================================
        self.optimizer_g = AdamW(
            self.generator.parameters(),
            lr=config.get('refine_lr', 1e-4),
            betas=(0.5, 0.999),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.optimizer_d = AdamW(
            self.discriminator.parameters(),
            lr=config.get('disc_lr', 2e-4),
            betas=(0.5, 0.999),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # LR schedulers
        num_epochs = config.get('refine_epochs', 50)
        self.lr_scheduler_g = CosineAnnealingLR(
            self.optimizer_g, T_max=num_epochs, eta_min=config.get('min_lr', 1e-6)
        )
        self.lr_scheduler_d = CosineAnnealingLR(
            self.optimizer_d, T_max=num_epochs, eta_min=config.get('min_lr', 1e-6)
        )

        # Loss weight
        self.lambda_adv = config.get('lambda_adv', 0.01)
        self.lambda_l1 = config.get('lambda_l1', 1.0)

        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 1
        self.metrics_history = []

        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler_g = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.scaler_d = torch.cuda.amp.GradScaler() if self.use_amp else None

        # =====================================================================
        # RESUME CHECKPOINT
        # =====================================================================
        if refinement_checkpoint_path is not None:
            self._load_checkpoint(refinement_checkpoint_path)

        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"\nRefinement-GAN Trainer initialized:")
        print(f"  Generator params:     {gen_params:,}")
        print(f"  Discriminator params: {disc_params:,}")
        print(f"  λ_adv: {self.lambda_adv}, λ_L1: {self.lambda_l1}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Resuming from epoch: {self.start_epoch}")

    # -----------------------------------------------------------------
    # Checkpoint loading / saving
    # -----------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_path):
        """Load a previously saved refinement-GAN checkpoint to resume training."""
        print(f"\nLoading refinement checkpoint from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)

            self.generator.load_state_dict(ckpt['generator_state_dict'])
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            print(f"  ✓ Generator and Discriminator weights restored")

            self.optimizer_g.load_state_dict(ckpt['optimizer_g_state_dict'])
            self.optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
            print(f"  ✓ Optimizer states restored")

            if self.scaler_g is not None and 'scaler_g_state_dict' in ckpt and ckpt['scaler_g_state_dict'] is not None:
                self.scaler_g.load_state_dict(ckpt['scaler_g_state_dict'])
            if self.scaler_d is not None and 'scaler_d_state_dict' in ckpt and ckpt['scaler_d_state_dict'] is not None:
                self.scaler_d.load_state_dict(ckpt['scaler_d_state_dict'])
            print(f"  ✓ AMP scaler states restored")

            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            resumed_epoch = ckpt.get('epoch', 0)
            self.start_epoch = resumed_epoch + 1

            for _ in range(resumed_epoch):
                self.lr_scheduler_g.step()
                self.lr_scheduler_d.step()

            print(f"  ✓ Resuming from epoch {resumed_epoch} → starting epoch {self.start_epoch}")
            print(f"  ✓ Best val loss so far: {self.best_val_loss:.6f}")

        except Exception as e:
            print(f"  ✗ Failed to load checkpoint: {e}")
            print(f"  → Starting from scratch")
            self.start_epoch = 1

    def save_checkpoint(self, epoch, val_loss):
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scaler_g_state_dict': self.scaler_g.state_dict() if self.scaler_g is not None else None,
            'scaler_d_state_dict': self.scaler_d.state_dict() if self.scaler_d is not None else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        torch.save(checkpoint, self.checkpoint_dir / 'refinement_latest.pt')

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'refinement_best.pt')
            print(f"  ✓ New best refinement model saved (val_loss: {val_loss:.4f})")

        if epoch % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'refinement_epoch_{epoch}.pt')

    # -----------------------------------------------------------------
    # Frozen diffusion generation
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate_diffusion(self, gridsat, era5, timestamps, storm_names, num_steps=50):
        """Generate images using frozen VAE + Diffusion UNet (Network 1)."""
        self.unet.eval()
        self.vae.eval()

        batch_size = gridsat.shape[0]
        latent_size = self.config['img_size'] // 4
        latent_dim = self.config.get('latent_dim', 4)

        # Start from random noise in latent space
        z = torch.randn(batch_size, latent_dim, latent_size, latent_size, device=self.device)

        # Encode conditions once
        encoded_cond, context_emb = self.unet.condition_encoder(
            gridsat, era5, timestamps, storm_names
        )
        encoded_cond = self.condition_downsample(encoded_cond)

        # DDIM sampling
        step_size = max(1, self.scheduler.num_timesteps // num_steps)
        timesteps = list(range(0, self.scheduler.num_timesteps, step_size))[::-1]

        for i, t_idx in enumerate(timesteps):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            noise_pred = self.unet(
                z, gridsat, era5, timestamps, storm_names, t,
                encoded_condition=encoded_cond, context_emb=context_emb
            )
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            z = self.scheduler.sample_prev_timestep_ddim(z, noise_pred, t_idx, t_prev=t_prev, eta=0.0)

        # Decode from latent to pixel space
        z_unscaled = z / self.latent_scale_factor
        images = self.vae.decode(z_unscaled)
        return images

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def train_epoch(self, dataloader, epoch):
        self.generator.train()
        self.discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_l1_loss = 0
        epoch_adv_loss = 0
        num_batches = 0
        debug_this_epoch = (epoch <= 2) or (epoch % 5 == 0)

        pbar = tqdm(dataloader, desc=f'Refinement-GAN - Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            target = batch['target'].to(self.device)
            timestamps = batch['timestamp']
            storm_names = batch['storm_name']

            # =========================================================
            # STEP 1: Generate with frozen diffusion model
            # =========================================================
            with torch.no_grad():
                generated = self.generate_diffusion(
                    gridsat, era5, timestamps, storm_names,
                    num_steps=self.config.get('refine_ddim_steps', 50)
                )

            # =========================================================
            # STEP 2: Train Discriminator
            # =========================================================
            self.optimizer_d.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Refine (detach from generator graph for D training)
                refined = self.generator(generated).detach()

                # Real vs refined
                real_pred = self.discriminator(gridsat, target)
                fake_pred = self.discriminator(gridsat, refined)

                # Hinge loss
                d_loss_real = F.relu(1.0 - real_pred).mean()
                d_loss_fake = F.relu(1.0 + fake_pred).mean()
                d_loss = d_loss_real + d_loss_fake

            if self.use_amp:
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.optimizer_d.step()

            # =========================================================
            # STEP 3: Train Generator (Refinement Network)
            # =========================================================
            self.optimizer_g.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                refined = self.generator(generated)

                # L1 reconstruction loss
                l1_loss = F.l1_loss(refined, target)

                # Adversarial loss (generator wants D to predict "real")
                fake_pred_for_g = self.discriminator(gridsat, refined)
                adv_loss = -fake_pred_for_g.mean()  # hinge-based generator loss

                # Combined generator loss
                g_loss = self.lambda_l1 * l1_loss + self.lambda_adv * adv_loss

            if self.use_amp:
                self.scaler_g.scale(g_loss).backward()
                self.scaler_g.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.optimizer_g.step()

            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_adv_loss += adv_loss.item()
            num_batches += 1

            # Debug first batch
            if batch_idx == 0 and debug_this_epoch:
                print(f"\n  ===== DEBUG: Refinement Epoch {epoch}, Batch 0 =====")
                print(f"  [Input]     target: range=[{target.min().item():.4f}, {target.max().item():.4f}]")
                print(f"  [Diffusion] generated: range=[{generated.min().item():.4f}, {generated.max().item():.4f}]")
                print(f"  [Refined]   refined: range=[{refined.min().item():.4f}, {refined.max().item():.4f}]")
                print(f"  [Loss]      G: {g_loss.item():.6f} (L1: {l1_loss.item():.6f}, Adv: {adv_loss.item():.6f})")
                print(f"  [Loss]      D: {d_loss.item():.6f} (Real: {d_loss_real.item():.6f}, Fake: {d_loss_fake.item():.6f})")
                mse_before = F.mse_loss(generated, target).item()
                mse_after = F.mse_loss(refined, target).item()
                print(f"  [MSE]       Before refine: {mse_before:.6f}, After refine: {mse_after:.6f}")

            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            pbar.set_postfix({
                'G': f"{g_loss.item():.4f}",
                'D': f"{d_loss.item():.4f}",
                'L1': f"{l1_loss.item():.4f}"
            })

        avg_g = epoch_g_loss / max(num_batches, 1)
        avg_d = epoch_d_loss / max(num_batches, 1)
        avg_l1 = epoch_l1_loss / max(num_batches, 1)
        avg_adv = epoch_adv_loss / max(num_batches, 1)

        return {
            'g_loss': avg_g,
            'd_loss': avg_d,
            'l1_loss': avg_l1,
            'adv_loss': avg_adv,
        }

    @torch.no_grad()
    def validate(self, dataloader):
        self.generator.eval()
        self.discriminator.eval()

        total_l1 = 0
        total_mse_before = 0
        total_mse_after = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc='Validation'):
            gridsat = batch['gridsat'].to(self.device)
            era5 = batch['era5'].to(self.device)
            target = batch['target'].to(self.device)
            timestamps = batch['timestamp']
            storm_names = batch['storm_name']

            generated = self.generate_diffusion(
                gridsat, era5, timestamps, storm_names,
                num_steps=self.config.get('refine_ddim_steps', 50)
            )
            refined = self.generator(generated)

            total_l1 += F.l1_loss(refined, target).item()
            total_mse_before += F.mse_loss(generated, target).item()
            total_mse_after += F.mse_loss(refined, target).item()
            num_batches += 1

        n = max(num_batches, 1)
        avg_l1 = total_l1 / n
        avg_mse_before = total_mse_before / n
        avg_mse_after = total_mse_after / n

        print(f"  Validation: L1={avg_l1:.6f}, MSE(before)={avg_mse_before:.6f}, MSE(after)={avg_mse_after:.6f}")
        return avg_l1

    def save_samples(self, dataloader, epoch):
        """Save comparison: input, diffusion output, refined, ground truth."""
        self.generator.eval()
        self.vae.eval()
        self.unet.eval()

        batch = next(iter(dataloader))
        batch_size = len(batch['gridsat'])
        num_samples = min(8, batch_size)

        gridsat = batch['gridsat'][:num_samples].to(self.device)
        era5 = batch['era5'][:num_samples].to(self.device)
        target = batch['target'][:num_samples].to(self.device)
        timestamps = batch['timestamp'][:num_samples]
        storm_names = batch['storm_name'][:num_samples]

        with torch.no_grad():
            generated = self.generate_diffusion(gridsat, era5, timestamps, storm_names, num_steps=50)
            refined = self.generator(generated)

        gridsat_np = gridsat.cpu().numpy()
        target_np = target.cpu().numpy()
        generated_np = generated.cpu().numpy()
        refined_np = refined.cpu().numpy()

        fig, axes = plt.subplots(4, num_samples, figsize=(2.5 * num_samples, 10))
        if num_samples == 1:
            axes = axes.reshape(4, 1)

        titles = ['Input (GridSAT)', 'Ground Truth', 'Diffusion Output', 'Refined']
        data = [gridsat_np, target_np, generated_np, refined_np]

        for i in range(num_samples):
            for row, (title, arr) in enumerate(zip(titles, data)):
                axes[row, i].imshow(arr[i, 0], cmap='viridis')
                axes[row, i].axis('off')
                if i == 0:
                    axes[row, i].set_ylabel(title, fontsize=9)

        plt.suptitle(f'Refinement-GAN — Epoch {epoch}', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'refinement_samples_epoch_{epoch}.png', dpi=150)
        plt.close()

        # Log sample-level metrics
        mse_before = F.mse_loss(generated, target).item()
        mse_after = F.mse_loss(refined, target).item()
        print(f"  [Samples] MSE before refine: {mse_before:.6f}, after: {mse_after:.6f}")

    def train(self, train_loader, val_loader):
        num_epochs = self.config.get('refine_epochs', 50)
        eval_every = self.config.get('eval_every', 5)

        print(f"\n{'='*60}")
        print("STAGE 3: REFINEMENT-GAN TRAINING")
        print(f"{'='*60}")
        print(f"Training for {num_epochs} epochs (starting from epoch {self.start_epoch})")
        print(f"Loss = {self.lambda_l1}*L1 + {self.lambda_adv}*adversarial")
        print(f"VAE and Diffusion UNet are FROZEN")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"  Train — G: {train_metrics['g_loss']:.6f}, "
                  f"D: {train_metrics['d_loss']:.6f}, "
                  f"L1: {train_metrics['l1_loss']:.6f}, "
                  f"Adv: {train_metrics['adv_loss']:.6f}")

            # Validate
            val_loss = self.validate(val_loader)
            self.train_losses.append(train_metrics['g_loss'])
            self.val_losses.append(val_loss)

            # LR step
            self.lr_scheduler_g.step()
            self.lr_scheduler_d.step()
            current_lr_g = self.optimizer_g.param_groups[0]['lr']
            print(f"  LR(G): {current_lr_g:.1e}")

            # CSV logging
            epoch_metrics = {
                'epoch': epoch,
                'g_loss': train_metrics['g_loss'],
                'd_loss': train_metrics['d_loss'],
                'l1_loss': train_metrics['l1_loss'],
                'adv_loss': train_metrics['adv_loss'],
                'val_l1': val_loss,
                'lr_g': current_lr_g,
            }
            self.metrics_history.append(epoch_metrics)
            pd.DataFrame(self.metrics_history).to_csv(
                self.metrics_dir / 'refinement_training_metrics.csv', index=False
            )

            # Checkpoint
            self.save_checkpoint(epoch, val_loss)

            # Save sample images
            if epoch % eval_every == 0 or epoch == num_epochs:
                print(f"\n  Saving sample images...")
                self.save_samples(val_loader, epoch)

        # Final training curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train G Loss')
        plt.plot(self.val_losses, label='Val L1 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Refinement-GAN Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'refinement_training_curves.png', dpi=150)
        plt.close()

        print(f"\n{'='*60}")
        print("REFINEMENT-GAN TRAINING COMPLETE")
        print(f"Best Val L1 Loss: {self.best_val_loss:.6f}")
        print(f"Checkpoints:  {self.checkpoint_dir}")
        print(f"Outputs:      {self.output_dir}")
        print(f"{'='*60}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    config = {
        # Data
        'root_dir': r'/kaggle/input/setcd-dataset',
        'train_years': ['2007', '2008', '2009', '2010', '2011'],
        'test_storm': ['2022349N13068'],
        'num_val_storms': 15,
        'batch_size': 4,
        'num_workers': 4,
        'img_size': 256,

        # Architecture params (must match frozen checkpoints)
        'latent_dim': 4,
        'vae_base_channels': 64,
        'base_channels': 64,
        'channel_mults': (1, 2, 4, 8),
        'num_heads': 4,
        'embed_dim': 128,
        't_emb_dim': 128,
        'num_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,

        # Refinement Network (Generator)
        'refine_base_channels': 64,
        'refine_lr': 1e-4,
        'refine_epochs': 50,
        'refine_ddim_steps': 50,

        # Discriminator
        'disc_base_channels': 64,
        'disc_n_layers': 3,
        'disc_lr': 2e-4,

        # Loss weights
        'lambda_adv': 0.01,
        'lambda_l1': 1.0,

        # Training
        'weight_decay': 1e-5,
        'min_lr': 1e-6,
        'use_amp': True,
        'eval_every': 5,

        # Output
        'checkpoint_dir': './checkpoints',
        'output_dir': './outputs',

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # Checkpoint paths
    vae_checkpoint_path = "/kaggle/input/models/kawsaralam5811/vae/tensorflow2/default/1/vae_best.pt"
    diffusion_checkpoint_path = "/kaggle/input/models/kawsaralam5811/unet/tensorflow2/default/1/diffusion_best.pt"
    refinement_checkpoint_path = None  # Set to path to resume training

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=config['root_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_years=config['train_years'],
        test_storm=config['test_storm'],
        img_size=config['img_size'],
        num_val_storms=config['num_val_storms']
    )

    # Train
    trainer = RefinementGANTrainer(
        config,
        vae_checkpoint_path,
        diffusion_checkpoint_path,
        refinement_checkpoint_path=refinement_checkpoint_path
    )
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
