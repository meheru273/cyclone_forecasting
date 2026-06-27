"""
Count storms and GRIDSAT/ERA5 file pairs per year in the SETCD dataset.
Read-only exploration script. Run once to validate the proposed split
(2012-2020 train / 2021 val / 2022 test).

Usage:
    # Use default Kaggle path:
    python count_storms_per_year.py

    # Override path:
    python count_storms_per_year.py --root /path/to/SETCD

    # Custom output files:
    python count_storms_per_year.py --out summary.csv

Output:
    - dataset_year_summary.csv : year, num_storms, num_timestamps, num_complete_pairs
    - dataset_per_storm.csv    : year, storm, timestamps, complete_pairs
    - Console: per-year table + projected split sizes
"""

import argparse
from pathlib import Path
import pandas as pd

# Default dataset root on Kaggle
DEFAULT_ROOT = '/kaggle/input/datasets/meheruzannat/cyclone-dataset'


def count_year(year_path: Path, gridsat_name: str, era5_name: str):
    """Return (n_storms, n_timestamps, n_complete_pairs, per_storm_breakdown)."""
    # Handle nested {year}/{year}/ structure some SETCD dumps use
    nested = year_path / year_path.name
    if nested.exists() and nested.is_dir():
        year_path = nested

    storms = sorted([p for p in year_path.iterdir() if p.is_dir()])
    total_ts = 0
    total_pairs = 0
    breakdown = []

    for storm in storms:
        timestamps = sorted([p for p in storm.iterdir() if p.is_dir()])
        n_ts = len(timestamps)
        n_pairs = sum(
            1 for ts in timestamps
            if (ts / gridsat_name).exists() and (ts / era5_name).exists()
        )
        total_ts += n_ts
        total_pairs += n_pairs
        breakdown.append({
            'storm': storm.name,
            'timestamps': n_ts,
            'complete_pairs': n_pairs,
        })

    return len(storms), total_ts, total_pairs, breakdown


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--root',
        default=DEFAULT_ROOT,
        help=f'SETCD root directory (contains year folders). '
             f'Default: {DEFAULT_ROOT}',
    )
    parser.add_argument(
        '--out',
        default='dataset_year_summary.csv',
        help='Year-level CSV output path',
    )
    parser.add_argument(
        '--per-storm-out',
        default='dataset_per_storm.csv',
        help='Per-storm CSV output path',
    )
    parser.add_argument(
        '--gridsat-name',
        default='GRIDSAT_data.npy',
        help='Expected GRIDSAT filename inside each timestamp folder',
    )
    parser.add_argument(
        '--era5-name',
        default='ERA5_data.npy',
        help='Expected ERA5 filename inside each timestamp folder',
    )
    parser.add_argument(
        '--train-years',
        nargs='+',
        default=[str(y) for y in range(2012, 2021)],
        help='Years to sum as the training split',
    )
    parser.add_argument(
        '--val-years',
        nargs='+',
        default=['2021'],
        help='Years for the validation split',
    )
    parser.add_argument(
        '--test-years',
        nargs='+',
        default=['2022'],
        help='Years for the test split',
    )
    args, _ = parser.parse_known_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(
            f"Dataset root not found: {root}\n"
            f"Make sure the Kaggle dataset is attached, or pass --root <path>."
        )

    year_folders = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    )
    if not year_folders:
        raise SystemExit(f"No numeric year folders found under {root}")

    print(f"Dataset root : {root}")
    print(f"Years found  : {[p.name for p in year_folders]}")
    print()
    print(f"{'Year':<6} {'Storms':>7} {'Timestamps':>12} {'Complete pairs':>15}")
    print("-" * 44)

    rows = []
    per_storm_rows = []

    for yp in year_folders:
        n_storms, n_ts, n_pairs, breakdown = count_year(
            yp, args.gridsat_name, args.era5_name
        )
        rows.append({
            'year': yp.name,
            'num_storms': n_storms,
            'num_timestamps': n_ts,
            'num_complete_pairs': n_pairs,
        })
        for b in breakdown:
            per_storm_rows.append({'year': yp.name, **b})
        print(f"{yp.name:<6} {n_storms:>7} {n_ts:>12} {n_pairs:>15}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.DataFrame(per_storm_rows).to_csv(args.per_storm_out, index=False)

    print()
    print(f"Year-level CSV : {args.out}")
    print(f"Per-storm CSV  : {args.per_storm_out}")

    # Totals row
    print()
    print(f"{'TOTAL':<6} "
          f"{df['num_storms'].sum():>7} "
          f"{df['num_timestamps'].sum():>12} "
          f"{df['num_complete_pairs'].sum():>15}")

    # Split summary
    print()
    print("----- Proposed split -----")
    for split_name, years in [
        ('train', args.train_years),
        ('val',   args.val_years),
        ('test',  args.test_years),
    ]:
        sub = df[df['year'].isin(years)]
        if sub.empty:
            print(f"  {split_name:5s} ({','.join(years)}): NO YEARS FOUND")
            continue
        n_storms  = sub['num_storms'].sum()
        n_pairs   = sub['num_complete_pairs'].sum()
        # Each consecutive (t, t+1) pair needs 2 timestamps in the same storm;
        # a storm with N timestamps yields at most N-1 consecutive pairs.
        n_consec  = n_pairs - n_storms
        print(
            f"  {split_name:5s} "
            f"({len(years)} year{'s' if len(years) > 1 else ''}: "
            f"{','.join(years)}): "
            f"{n_storms:>3} storms, "
            f"{n_pairs:>6} complete pairs, "
            f"~{n_consec:>6} consecutive (t→t+1) candidates"
        )

    # Extra: per-year breakdown of the proposed splits for quick sanity check
    print()
    print("----- Per-year detail -----")
    all_split_years = set(args.train_years + args.val_years + args.test_years)
    detail = df[df['year'].isin(all_split_years)].copy()
    detail['split'] = detail['year'].apply(
        lambda y: 'train' if y in args.train_years
        else ('val' if y in args.val_years else 'test')
    )
    print(detail[['split', 'year', 'num_storms',
                  'num_timestamps', 'num_complete_pairs']].to_string(index=False))


if __name__ == '__main__':
    main()