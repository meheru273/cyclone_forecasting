"""
MultiFrameCycloneDataset: yields windows of K_in past frames + K_out future frames.

Coexists with the single-frame `CycloneDataset` in multi_channel_diffusion.py —
Phase-1 single-frame remains reproducible.

Phase-2 default: K_in = 3, K_out = 3 at 3-hour cadence (9-hour horizon).

Sample dict layout (returned by __getitem__):
    'gridsat_in'      : (K_in, 1, H, W)
    'era5_in'         : (K_in, 4, H, W)
    'target_gs'       : (K_out, 1, H, W)
    'target_era5'     : (K_out, 4, H, W)
    'coords_in'       : (K_in, 2)               normalised lat/lon
    'target_coords'   : (K_out, 2)              normalised lat/lon (supervision)
    'raw_lat_in'      : (K_in,)                 raw degrees
    'raw_lon_in'      : (K_in,)                 raw degrees
    'raw_target_lat'  : (K_out,)                raw degrees
    'raw_target_lon'  : (K_out,)                raw degrees
    'timestamps_in'   : list[str] of length K_in
    'target_timestamps': list[str] of length K_out
    'storm_name'      : str
    'year'            : str
    (when use_cache=True, additionally:)
    'target_latents'  : (K_out, latent_dim, H/4, W/4)  pre-encoded VAE mu
    'input_latents'   : (K_in,  latent_dim, H/4, W/4)  (optional, only if needed)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from multi_channel_diffusion import parse_storm_coordinates  # SID parser


class MultiFrameCycloneDataset(Dataset):
    """Multi-frame variant of CycloneDataset.

    Args:
        root_dir: SETCD root.
        years:    list of year folder names (e.g. ['2012',...,'2020']).
        k_in:     number of conditioning frames.
        k_out:    number of prediction frames.
        mode:     'train' / 'val' / 'test' — controls augmentation only.
        img_size: spatial resolution.
        coord_lookup_path: per-timestep IBTrACS JSON (built by build_coordinate_lookup.py)
        latent_cache_dir:  if provided, returns target_latents pre-encoded (mu)
        return_input_latents: if True (and cache provided), also returns input_latents
    """

    GS_MEAN   = torch.tensor([169.8992]).view(-1, 1, 1)
    GS_STD    = torch.tensor([122.2612]).view(-1, 1, 1)
    ERA5_MEAN = torch.tensor([-1.1655, 0.3835, 298.4892, 100976.2891]).view(-1, 1, 1)
    ERA5_STD  = torch.tensor([5.3889,  5.0042,   4.2326,    521.1500]).view(-1, 1, 1)

    def __init__(self, root_dir, years, k_in=3, k_out=3,
                 mode='train', img_size=256,
                 coord_lookup_path: Optional[str] = None,
                 latent_cache_dir: Optional[str] = None,
                 return_input_latents: bool = False,
                 max_samples: Optional[int] = None,
                 gridsat_name='GRIDSAT_data.npy',
                 era5_name='ERA5_data.npy'):
        self.root_dir = Path(root_dir)
        self.years = years
        self.k_in = int(k_in)
        self.k_out = int(k_out)
        self.mode = mode
        self.img_size = img_size
        self.gridsat_name = gridsat_name
        self.era5_name = era5_name
        self.latent_cache_dir = Path(latent_cache_dir) if latent_cache_dir else None
        self.return_input_latents = bool(return_input_latents)
        # publish normalisation stats with the same attribute names as the
        # single-frame CycloneDataset, so eval helpers that pull from
        # `loader.dataset.era5_mean/std` keep working.
        self.gs_mean   = self.GS_MEAN
        self.gs_std    = self.GS_STD
        self.era5_mean = self.ERA5_MEAN
        self.era5_std  = self.ERA5_STD

        # Coord lookup
        self.coord_lookup = {}
        self.coord_mean = torch.tensor([0.0, 180.0])
        self.coord_std  = torch.tensor([15.0, 100.0])
        if coord_lookup_path and Path(coord_lookup_path).exists():
            with open(coord_lookup_path, 'r') as f:
                data = json.load(f)
            self.coord_lookup = data.get('lookup', {})
            stats = data.get('coord_stats', {})
            self.coord_mean = torch.tensor([stats.get('lat_mean', 0.0),
                                            stats.get('lon_mean', 180.0)])
            self.coord_std  = torch.tensor([stats.get('lat_std', 15.0),
                                            stats.get('lon_std', 100.0)])

        self.samples = self._build_index(max_samples)

    def _build_index(self, max_samples):
        out = []
        win = self.k_in + self.k_out
        for yr in self.years:
            year_path = self.root_dir / yr
            if not year_path.exists():
                continue
            nested = year_path / yr
            if nested.exists() and nested.is_dir():
                year_path = nested
            for storm_folder in sorted(p for p in year_path.iterdir() if p.is_dir()):
                storm = storm_folder.name
                tsteps = sorted([t.name for t in storm_folder.iterdir() if t.is_dir()])
                if len(tsteps) < win:
                    continue
                for i in range(len(tsteps) - win + 1):
                    window = tsteps[i:i + win]
                    # check all files exist
                    ok = True
                    for t in window:
                        if not ((storm_folder / t / self.gridsat_name).exists()
                                and (storm_folder / t / self.era5_name).exists()):
                            ok = False
                            break
                    if not ok:
                        continue
                    out.append({
                        'storm_folder': storm_folder,
                        'storm_name':   storm,
                        'year':         yr,
                        'window':       window,
                    })
                    if max_samples and len(out) >= max_samples:
                        return out
        return out

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    def _load_frame(self, storm_folder: Path, ts_name: str):
        gs = np.load(storm_folder / ts_name / self.gridsat_name).astype(np.float32)
        e5 = np.load(storm_folder / ts_name / self.era5_name).astype(np.float32)
        gs = np.nan_to_num(gs); e5 = np.nan_to_num(e5)
        if gs.ndim == 2:
            gs = gs[np.newaxis, ...]
        else:
            gs = gs[0:1]
        if e5.ndim == 2:
            e5 = e5[np.newaxis, ...]
        else:
            e5 = e5[:4]
        gs_t = torch.from_numpy(gs)
        e5_t = torch.from_numpy(e5)
        # resize
        target_size = (self.img_size, self.img_size)
        gs_t = F.interpolate(gs_t.unsqueeze(0), size=target_size,
                             mode='bilinear', align_corners=False).squeeze(0)
        e5_t = F.interpolate(e5_t.unsqueeze(0), size=target_size,
                             mode='bilinear', align_corners=False).squeeze(0)
        # z-score
        gs_t = (gs_t - self.gs_mean) / self.gs_std
        e5_t = (e5_t - self.era5_mean) / self.era5_std
        return gs_t, e5_t

    def _lookup_coords(self, storm, ts):
        key = f"{storm}/{ts}"
        if key in self.coord_lookup:
            return self.coord_lookup[key]['lat'], self.coord_lookup[key]['lon']
        return parse_storm_coordinates(storm)   # fallback to genesis

    def _normalise_coords(self, lat, lon):
        return torch.tensor([
            (lat - float(self.coord_mean[0])) / float(self.coord_std[0]),
            (lon - float(self.coord_mean[1])) / float(self.coord_std[1]),
        ], dtype=torch.float32)

    def _load_latent(self, storm, ts):
        if self.latent_cache_dir is None:
            return None
        p = self.latent_cache_dir / storm / f"{ts}.npy"
        if p.exists():
            return torch.from_numpy(np.load(p)).float()
        return None

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        s = self.samples[idx]
        storm = s['storm_name']
        storm_folder = s['storm_folder']
        window = s['window']
        in_ts  = window[:self.k_in]
        out_ts = window[self.k_in:]

        # Load frames
        gs_in_list, e5_in_list = [], []
        for ts in in_ts:
            gs_t, e5_t = self._load_frame(storm_folder, ts)
            gs_in_list.append(gs_t); e5_in_list.append(e5_t)
        gs_out_list, e5_out_list = [], []
        for ts in out_ts:
            gs_t, e5_t = self._load_frame(storm_folder, ts)
            gs_out_list.append(gs_t); e5_out_list.append(e5_t)

        gridsat_in = torch.stack(gs_in_list, dim=0)      # (K_in, 1, H, W)
        era5_in    = torch.stack(e5_in_list, dim=0)       # (K_in, 4, H, W)
        target_gs  = torch.stack(gs_out_list, dim=0)
        target_era5 = torch.stack(e5_out_list, dim=0)

        # Consistent left-right flip augmentation across all frames in this sample
        if self.mode == 'train' and random.random() > 0.5:
            gridsat_in  = torch.flip(gridsat_in,  [-1])
            era5_in     = torch.flip(era5_in,     [-1])
            target_gs   = torch.flip(target_gs,   [-1])
            target_era5 = torch.flip(target_era5, [-1])

        # Coords
        coords_in_norm, raw_lats_in, raw_lons_in = [], [], []
        for ts in in_ts:
            lat, lon = self._lookup_coords(storm, ts)
            coords_in_norm.append(self._normalise_coords(lat, lon))
            raw_lats_in.append(lat); raw_lons_in.append(lon)

        target_coords_norm, raw_target_lats, raw_target_lons = [], [], []
        for ts in out_ts:
            lat, lon = self._lookup_coords(storm, ts)
            target_coords_norm.append(self._normalise_coords(lat, lon))
            raw_target_lats.append(lat); raw_target_lons.append(lon)

        item = {
            'gridsat_in':       gridsat_in,                          # (K_in, 1, H, W)
            'era5_in':          era5_in,                              # (K_in, 4, H, W)
            'target_gs':        target_gs,                            # (K_out, 1, H, W)
            'target_era5':      target_era5,                          # (K_out, 4, H, W)
            'coords_in':        torch.stack(coords_in_norm, dim=0),   # (K_in, 2)
            'target_coords':    torch.stack(target_coords_norm, dim=0),
            'raw_lat_in':       torch.tensor(raw_lats_in,  dtype=torch.float32),
            'raw_lon_in':       torch.tensor(raw_lons_in,  dtype=torch.float32),
            'raw_target_lat':   torch.tensor(raw_target_lats, dtype=torch.float32),
            'raw_target_lon':   torch.tensor(raw_target_lons, dtype=torch.float32),
            'timestamps_in':    in_ts,
            'target_timestamps': out_ts,
            'storm_name':       storm,
            'year':             s['year'],
        }

        # Latent cache lookups (target frames are always needed during training;
        # input frames optionally, for V2 temporal attention etc.)
        if self.latent_cache_dir is not None:
            tgt_lat = []
            ok_tgt = True
            for ts in out_ts:
                z = self._load_latent(storm, ts)
                if z is None:
                    ok_tgt = False
                    break
                tgt_lat.append(z)
            if ok_tgt:
                item['target_latents'] = torch.stack(tgt_lat, dim=0)   # (K_out, latent_dim, h, w)

            if self.return_input_latents:
                in_lat = []
                ok_in = True
                for ts in in_ts:
                    z = self._load_latent(storm, ts)
                    if z is None:
                        ok_in = False
                        break
                    in_lat.append(z)
                if ok_in:
                    item['input_latents'] = torch.stack(in_lat, dim=0)

        return item


def get_multi_frame_dataloaders(root_dir, train_years, val_years, test_years,
                                k_in=3, k_out=3,
                                batch_size=2, num_workers=2,
                                img_size=256,
                                coord_lookup_path=None,
                                latent_cache_dir=None,
                                return_input_latents=False):
    train_ds = MultiFrameCycloneDataset(
        root_dir, train_years, k_in=k_in, k_out=k_out, mode='train',
        img_size=img_size, coord_lookup_path=coord_lookup_path,
        latent_cache_dir=latent_cache_dir, return_input_latents=return_input_latents)
    val_ds = MultiFrameCycloneDataset(
        root_dir, val_years, k_in=k_in, k_out=k_out, mode='val',
        img_size=img_size, coord_lookup_path=coord_lookup_path,
        latent_cache_dir=latent_cache_dir, return_input_latents=return_input_latents)
    test_ds = MultiFrameCycloneDataset(
        root_dir, test_years, k_in=k_in, k_out=k_out, mode='test',
        img_size=img_size, coord_lookup_path=coord_lookup_path,
        latent_cache_dir=latent_cache_dir, return_input_latents=return_input_latents)

    print(f"Multi-frame (K_in={k_in}, K_out={k_out})")
    print(f"  Train years {train_years}: {len(train_ds)} samples")
    print(f"  Val   years {val_years}:  {len(val_ds)} samples")
    print(f"  Test  years {test_years}: {len(test_ds)} samples")

    # persistent_workers=True keeps workers alive across epochs (each multi-frame
    # sample loads 6 frames × 5 channels × 256×256 npy = ~7.5 MB; respawning
    # workers per epoch was the main cause of GPU sitting idle between epochs).
    # prefetch_factor=4 has each worker pre-load 4 batches so the GPU never waits.
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              **loader_kwargs)
    return train_loader, val_loader, test_loader
