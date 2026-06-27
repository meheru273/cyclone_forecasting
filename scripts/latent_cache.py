"""
Latent caching for the frozen MC-VAE.

When training a multi-frame model, every sample needs 6 frames encoded by the
VAE. Across an 80-epoch run on ~25 000 training samples that's millions of VAE
forwards, dominating wall-clock time. Pre-encoding every (storm, timestamp)
once and saving the result to disk reduces epoch time by ~3×.

Design:
  - One precompute script walks the dataset, calls vae.encode → mu, saves as
    `{cache_dir}/{storm}/{timestamp}.npy` (deterministic; uses mu, not sampled z).
  - At training time, the model uses `LatentCache.get_or_encode(...)` which
    either returns the cached mu or falls back to live encoding.
  - The user is responsible for re-running the precompute when the VAE
    checkpoint changes (cache dir convention: include checkpoint hash).

Usage (precompute):
    python scripts/latent_cache.py \\
        --vae-checkpoint ./checkpoints/mc_vae/mc_vae_best.pt \\
        --cache-dir       ./cache/latents_mc_vae \\
        --root            /path/to/SETCD \\
        --years 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022

Usage (training): see how multi_frame_flow.py loads it.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


def vae_fingerprint(checkpoint_path: str) -> str:
    """Short hash of the VAE checkpoint file — used to namespace cache dirs."""
    h = hashlib.sha1()
    with open(checkpoint_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:10]


class LatentCache:
    """Per-(storm, timestamp) cache of VAE encodings."""

    def __init__(self, cache_dir: str, vae=None, device: str = 'cuda'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vae = vae           # may be None if cache is fully precomputed
        self.device = device
        self.hits = 0
        self.misses = 0

    def _path(self, storm: str, timestamp: str) -> Path:
        return self.cache_dir / storm / f"{timestamp}.npy"

    def has(self, storm: str, timestamp: str) -> bool:
        return self._path(storm, timestamp).exists()

    @torch.no_grad()
    def get_or_encode(self, storm: str, timestamp: str,
                      x_5ch: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the cached (mu) for the given key, or encode-and-cache if missing.

        Args:
            storm, timestamp: the unique key
            x_5ch: (1, 5, H, W) tensor (single sample); only needed on cache miss
        Returns:
            mu tensor on `self.device` with shape (1, latent_dim, H/4, W/4)
        """
        p = self._path(storm, timestamp)
        if p.exists():
            arr = np.load(p)
            self.hits += 1
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)
        if self.vae is None or x_5ch is None:
            raise FileNotFoundError(
                f"Cache miss for {storm}/{timestamp} and no VAE available to encode.")
        self.misses += 1
        mu, _ = self.vae.encode(x_5ch.to(self.device))
        arr = mu.squeeze(0).detach().cpu().numpy().astype(np.float32)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        return mu

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(total, 1),
        }


# ---------------------------------------------------------------------------
# Precompute script
# ---------------------------------------------------------------------------

def _build_index(root: Path, years):
    """Return list of (storm_path, timestamp_folder) tuples to encode."""
    items = []
    for yr in years:
        year_path = root / yr
        if not year_path.exists():
            continue
        nested = year_path / yr
        if nested.exists() and nested.is_dir():
            year_path = nested
        for storm_folder in sorted(p for p in year_path.iterdir() if p.is_dir()):
            for ts_folder in sorted(p for p in storm_folder.iterdir() if p.is_dir()):
                if ((ts_folder / 'GRIDSAT_data.npy').exists()
                        and (ts_folder / 'ERA5_data.npy').exists()):
                    items.append((storm_folder.name, ts_folder.name, ts_folder))
    return items


def precompute_main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--cache-dir', required=True,
                        help='Output directory for cached latents')
    parser.add_argument('--root', required=True, help='SETCD root directory')
    parser.add_argument('--years', nargs='+', required=True,
                        help='Year folders to encode, e.g. 2012 2013 ... 2022')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=4)
    parser.add_argument('--vae-base-channels', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-encode even if cache file exists.')
    args = parser.parse_args()

    # Lazy import to keep this module light when only LatentCache class is needed
    from multi_channel_diffusion import MultiChannelVAE
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading VAE from {args.vae_checkpoint}")
    ckpt = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    vae = MultiChannelVAE(
        in_channels=5,
        latent_dim=args.latent_dim,
        base_channels=args.vae_base_channels,
    ).to(device).eval()
    vae.load_state_dict(ckpt['vae_state_dict'])
    for p in vae.parameters():
        p.requires_grad = False

    fp = vae_fingerprint(args.vae_checkpoint)
    cache_root = Path(args.cache_dir) / fp
    print(f"Cache dir: {cache_root}  (VAE fingerprint = {fp})")
    cache = LatentCache(str(cache_root), vae=vae, device=str(device))

    # Normalisation stats (must match CycloneDataset)
    gs_mean = torch.tensor([169.8992], dtype=torch.float32).view(1, 1, 1, 1).to(device)
    gs_std  = torch.tensor([122.2612], dtype=torch.float32).view(1, 1, 1, 1).to(device)
    era5_mean = torch.tensor([-1.1655, 0.3835, 298.4892, 100976.2891],
                             dtype=torch.float32).view(1, 4, 1, 1).to(device)
    era5_std  = torch.tensor([5.3889, 5.0042, 4.2326, 521.1500],
                              dtype=torch.float32).view(1, 4, 1, 1).to(device)

    items = _build_index(Path(args.root), args.years)
    print(f"Found {len(items)} (storm, timestamp) pairs to encode.")

    pbar = tqdm(items, desc='Encoding')
    encoded = 0
    skipped = 0
    for storm, ts_name, ts_folder in pbar:
        out_path = cache._path(storm, ts_name)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        gs = np.load(ts_folder / 'GRIDSAT_data.npy').astype(np.float32)
        e5 = np.load(ts_folder / 'ERA5_data.npy').astype(np.float32)
        gs = np.nan_to_num(gs); e5 = np.nan_to_num(e5)
        # Normalise dims
        if gs.ndim == 2:
            gs = gs[np.newaxis, ...]
        else:
            gs = gs[0:1]
        if e5.ndim == 2:
            e5 = e5[np.newaxis, ...]
        else:
            e5 = e5[:4]
        gs_t = torch.from_numpy(gs).unsqueeze(0).to(device)         # (1, 1, H, W)
        e5_t = torch.from_numpy(e5).unsqueeze(0).to(device)          # (1, 4, H, W)
        # Resize to img_size
        gs_t = F.interpolate(gs_t, size=(args.img_size, args.img_size),
                             mode='bilinear', align_corners=False)
        e5_t = F.interpolate(e5_t, size=(args.img_size, args.img_size),
                             mode='bilinear', align_corners=False)
        # z-score
        gs_t = (gs_t - gs_mean) / gs_std
        e5_t = (e5_t - era5_mean) / era5_std
        x5   = torch.cat([gs_t, e5_t], dim=1)
        mu, _ = vae.encode(x5)
        arr = mu.squeeze(0).detach().cpu().numpy().astype(np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)
        encoded += 1
        if encoded % 200 == 0:
            pbar.set_postfix(encoded=encoded, skipped=skipped)

    print(f"\nDone. Encoded {encoded}, skipped {skipped}.")
    print(f"Cache size on disk: {cache_root}")


if __name__ == '__main__':
    precompute_main()
