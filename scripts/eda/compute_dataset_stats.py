"""
compute_dataset_stats.py — Precompute global mean/std for GRIDSAT and ERA5 data.

Uses Welford's online algorithm to compute running mean and variance
without loading the entire dataset into memory.

Run this ONCE before training:
    python compute_dataset_stats.py --root_dir /kaggle/input/setcd-dataset \
        --years 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 \
        --output dataset_stats.json

The output JSON is consumed by CycloneDataset in diffusion_unet_train.py
and latent_diffusion.py for z-score normalization.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_stats(root_dir, years, gridsat_type='GRIDSAT_data.npy',
                  era5_type='ERA5_data.npy'):
    """
    Compute global mean/std for GRIDSAT (1-ch) and ERA5 (4-ch) using
    Welford's online algorithm.
    """
    root_dir = Path(root_dir)

    # Welford accumulators — GRIDSAT (1 channel)
    gs_count = 0
    gs_mean = 0.0
    gs_m2 = 0.0

    # Welford accumulators — ERA5 (per-channel, up to 4)
    era5_n_channels = 4
    era5_count = np.zeros(era5_n_channels)
    era5_mean = np.zeros(era5_n_channels)
    era5_m2 = np.zeros(era5_n_channels)

    # Collect all file paths first
    gs_files = []
    era5_files = []

    for year_folder in years:
        year_path = root_dir / year_folder
        if not year_path.exists():
            print(f"  [WARN] Year folder not found: {year_path}")
            continue
        # Handle nested year structure
        nested_path = year_path / year_folder
        if nested_path.exists():
            year_path = nested_path

        for cyclone_folder in year_path.iterdir():
            if not cyclone_folder.is_dir():
                continue
            for ts_folder in cyclone_folder.iterdir():
                if not ts_folder.is_dir():
                    continue
                gs_file = ts_folder / gridsat_type
                era5_file = ts_folder / era5_type
                if gs_file.exists():
                    gs_files.append(gs_file)
                if era5_file.exists():
                    era5_files.append(era5_file)

    print(f"Found {len(gs_files)} GRIDSAT files, {len(era5_files)} ERA5 files")

    # --- GRIDSAT stats ---
    print("\nComputing GRIDSAT statistics...")
    for fpath in tqdm(gs_files, desc="GRIDSAT"):
        data = np.load(str(fpath)).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        # Flatten — treat every pixel as a sample
        pixels = data.ravel()
        n = len(pixels)
        for val in [pixels]:  # batch update
            batch_mean = val.mean()
            batch_var = val.var()
            batch_count = len(val)

            delta = batch_mean - gs_mean
            total = gs_count + batch_count
            new_mean = gs_mean + delta * batch_count / total
            # Parallel Welford update
            gs_m2 += batch_var * batch_count + delta ** 2 * gs_count * batch_count / total
            gs_mean = new_mean
            gs_count = total

    gs_std = np.sqrt(gs_m2 / gs_count) if gs_count > 0 else 1.0

    # --- ERA5 stats (per-channel) ---
    print("\nComputing ERA5 statistics...")
    for fpath in tqdm(era5_files, desc="ERA5"):
        data = np.load(str(fpath)).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            data = data[:era5_n_channels]

        n_ch = min(data.shape[0], era5_n_channels)
        for c in range(n_ch):
            pixels = data[c].ravel()
            batch_mean = pixels.mean()
            batch_var = pixels.var()
            batch_count = len(pixels)

            delta = batch_mean - era5_mean[c]
            total = era5_count[c] + batch_count
            new_mean = era5_mean[c] + delta * batch_count / total
            era5_m2[c] += batch_var * batch_count + delta ** 2 * era5_count[c] * batch_count / total
            era5_mean[c] = new_mean
            era5_count[c] = total

    era5_std = np.sqrt(era5_m2 / np.maximum(era5_count, 1))
    # Guard against zero std (constant channels)
    era5_std = np.where(era5_std < 1e-8, 1.0, era5_std)
    if gs_std < 1e-8:
        gs_std = 1.0

    stats = {
        'gridsat': {
            'mean': [float(gs_mean)],
            'std': [float(gs_std)]
        },
        'era5': {
            'mean': [float(m) for m in era5_mean],
            'std': [float(s) for s in era5_std]
        },
        'num_gridsat_files': len(gs_files),
        'num_era5_files': len(era5_files),
        'gridsat_pixel_count': int(gs_count),
        'era5_pixel_counts': [int(c) for c in era5_count]
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Compute global dataset statistics for normalization')
    parser.add_argument('--root_dir', type=str,
                        default=r'/kaggle/input/setcd-dataset',
                        help='Root directory of the SETCD dataset')
    parser.add_argument('--years', nargs='+',
                        default=['2006','2007','2008','2009','2010',
                                 '2011','2012','2013','2014','2015'],
                        help='Year folders to scan')
    parser.add_argument('--output', type=str, default='dataset_stats.json',
                        help='Output JSON path')
    parser.add_argument('--gridsat_type', type=str, default='GRIDSAT_data.npy')
    parser.add_argument('--era5_type', type=str, default='ERA5_data.npy')
    args = parser.parse_args()

    print(f"Root dir: {args.root_dir}")
    print(f"Years: {args.years}")
    print(f"Output: {args.output}")

    stats = compute_stats(args.root_dir, args.years,
                          args.gridsat_type, args.era5_type)

    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print("Dataset Statistics")
    print(f"{'='*50}")
    print(f"GRIDSAT  mean={stats['gridsat']['mean'][0]:.4f}  "
          f"std={stats['gridsat']['std'][0]:.4f}")
    for c in range(len(stats['era5']['mean'])):
        print(f"ERA5[{c}]  mean={stats['era5']['mean'][c]:.4f}  "
              f"std={stats['era5']['std'][c]:.4f}")
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
