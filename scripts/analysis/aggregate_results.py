"""
Phase-3 results aggregator.

Walks the per-run metric CSVs, builds the canonical Table-1
(`main_results_table.csv`) used in the paper, and the
`per_leadtime_summary.csv` for the lead-time × metric figures.

Each registered run points to:
    - run_id      (paper row label, e.g. 'R6_mf_2rf_no_phys')
    - description (human-readable)
    - per_leadtime_csv (full per-leadtime CSV produced by the trainer)
    - sampler_steps_csv (the sweep CSV; optional)
    - param_count (model parameter count; optional, displayed in the table)

The result CSV has one row per (run_id, lead, metric).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


HEADLINE_METRICS = [
    'overall_psnr', 'overall_ssim', 'overall_mse',
    'gridsat_lpips', 'gridsat_rapsd_l1',
    'csi_220', 'pod_220', 'far_220',
    'mslp_err_hpa', 'max_wind_err',
    'track_km', 'crps',
]


def _read_run(spec: dict) -> Optional[pd.DataFrame]:
    p = Path(spec['per_leadtime_csv'])
    if not p.exists():
        print(f"  [skip] {spec['run_id']}: {p} not found")
        return None
    df = pd.read_csv(p)
    df.insert(0, 'run_id', spec['run_id'])
    df.insert(1, 'description', spec.get('description', ''))
    return df


def aggregate(registry_path: str, out_dir: str):
    """`registry_path` is a JSON file: list of dicts with keys run_id,
    description, per_leadtime_csv [, sampler_steps_csv, param_count]."""
    with open(registry_path, 'r') as f:
        registry = json.load(f)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for spec in registry:
        df = _read_run(spec)
        if df is not None:
            frames.append(df)
    if not frames:
        raise SystemExit("No runs found — check the registry paths.")

    combined = pd.concat(frames, ignore_index=True)

    # Main table: one row per (run_id, lead) with the headline columns
    keep = ['run_id', 'description', 'lead'] + [c for c in HEADLINE_METRICS
                                                  if c in combined.columns]
    main = combined[keep].copy()
    main.to_csv(out_dir / 'main_results_table.csv', index=False)

    # Per-leadtime pivot (one row per run_id, columns multi-indexed on metric × lead)
    long = main.melt(id_vars=['run_id', 'description', 'lead'],
                     var_name='metric', value_name='value')
    pivot = long.pivot_table(index=['run_id', 'description'],
                             columns=['metric', 'lead'], values='value')
    pivot.to_csv(out_dir / 'per_leadtime_summary.csv')

    # Sampler-step sweeps (optional, only if at least one run has it)
    sweep_rows = []
    for spec in registry:
        sp = spec.get('sampler_steps_csv')
        if sp and Path(sp).exists():
            s = pd.read_csv(sp)
            s.insert(0, 'run_id', spec['run_id'])
            sweep_rows.append(s)
    if sweep_rows:
        pd.concat(sweep_rows, ignore_index=True).to_csv(
            out_dir / 'sampler_steps_combined.csv', index=False)
        print(f"  Sampler-step sweeps combined → {out_dir / 'sampler_steps_combined.csv'}")

    print(f"\nWrote:\n  {out_dir / 'main_results_table.csv'}")
    print(f"  {out_dir / 'per_leadtime_summary.csv'}")
    print(f"\nRows in main table: {len(main)}")
    print(main.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--registry', required=True,
                        help='JSON file listing runs to aggregate')
    parser.add_argument('--out', default='./paper_results',
                        help='Output directory for aggregated tables')
    args = parser.parse_args()
    aggregate(args.registry, args.out)


if __name__ == '__main__':
    main()
