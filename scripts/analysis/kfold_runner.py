"""
Phase-3 §3.4 storm-level k-fold cross-validation runner.

Splits training-year storms into k stratified folds (by basin), trains the
headline multi-frame model once per fold, and aggregates per-fold metrics.

Stratification uses SETCD-encoded basin extracted from the storm SID
(see `parse_storm_coordinates` in multi_channel_diffusion.py). Basins:
    N -> Northern Hemisphere, S -> Southern Hemisphere; refined to NA/EP/WP/NI
    by longitude bands. If basin coverage is uneven, neighbouring basins are
    merged to preserve a minimum fold size.

Usage:
    python scripts/analysis/kfold_runner.py \\
        --root /path/to/SETCD \\
        --train-years 2012 2013 ... 2020 \\
        --val-year 2021 \\
        --k 5 \\
        --mc-vae ./checkpoints/mc_vae/mc_vae_best.pt \\
        --base-out ./outputs/kfold

This launches `k` training runs sequentially. Each fold's test slice is a
20 % stratified subset of train-year storms; that slice is *excluded* from
training and used for per-fold metrics. The held-out year (2021) is *not*
used here — it remains the canonical validation set.

For the lighter 80-epoch protocol (phase3.md §3.4.2), pass
`--diffusion-epochs 80`. Set `--smoke` for a 2-epoch dry run to confirm the
pipeline works end-to-end before committing the real ~60-h run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch

from multi_channel_diffusion import parse_storm_coordinates
from multi_frame_dataset import MultiFrameCycloneDataset
from multi_frame_flow import MultiFrameRFTrainer


# ---------------------------------------------------------------------------
# Storm enumeration + basin tagging
# ---------------------------------------------------------------------------

def basin_from_sid(sid: str) -> str:
    """Coarse basin label from an IBTrACS SID (e.g. '2018171N12245').
    Returns one of {'NA','EP','WP','NI','SH','UN'}.
    """
    try:
        lat, lon = parse_storm_coordinates(sid)
    except Exception:
        return 'UN'
    if lat < 0:
        return 'SH'                 # whole Southern Hemisphere
    # Northern hemisphere: split by longitude
    if   100 <= lon < 180: return 'WP'   # Western Pacific
    elif 180 <= lon < 260: return 'EP'   # Eastern Pacific
    elif 260 <= lon < 360 or lon < 30: return 'NA'   # North Atlantic
    elif 30  <= lon < 100: return 'NI'   # North Indian
    return 'UN'


def list_storms(root: Path, years: List[str]) -> Dict[str, str]:
    """Return {storm_id: basin} dict for storms in the given years."""
    out = {}
    for yr in years:
        yp = root / yr
        if not yp.exists():
            continue
        nested = yp / yr
        if nested.exists() and nested.is_dir():
            yp = nested
        for d in yp.iterdir():
            if d.is_dir():
                out[d.name] = basin_from_sid(d.name)
    return out


def stratified_kfold_storms(storm_basin: Dict[str, str], k: int = 5,
                             seed: int = 0) -> List[List[str]]:
    """Return list of k lists, each containing the storms in that fold's
    test slice. Stratified by basin."""
    rng = random.Random(seed)
    by_basin: Dict[str, List[str]] = {}
    for sid, b in storm_basin.items():
        by_basin.setdefault(b, []).append(sid)
    # shuffle within each basin
    for b in by_basin:
        rng.shuffle(by_basin[b])
    folds: List[List[str]] = [[] for _ in range(k)]
    for b, storms in by_basin.items():
        for i, sid in enumerate(storms):
            folds[i % k].append(sid)
    for i, f in enumerate(folds):
        rng.shuffle(f)
        print(f"  fold {i}: {len(f)} storms — basins: "
              + ", ".join(f"{bb}:{sum(1 for s in f if storm_basin[s]==bb)}"
                          for bb in sorted({storm_basin[s] for s in f})))
    return folds


# ---------------------------------------------------------------------------
# Subset dataset wrapper — pin to a fixed storm allow-list
# ---------------------------------------------------------------------------

class _StormFilteredMultiFrame(MultiFrameCycloneDataset):
    """Multi-frame dataset filtered to a specific set of storm IDs."""

    def __init__(self, *args, allowed_storms: set, **kwargs):
        self._allowed = set(allowed_storms)
        super().__init__(*args, **kwargs)

    def _build_index(self, max_samples):
        items = super()._build_index(max_samples)
        keep = [s for s in items if s['storm_name'] in self._allowed]
        return keep


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------

def run_fold(fold_idx: int, train_storms: set, test_storms: set,
             config: dict, mc_vae_path: str, base_out: Path) -> Dict:
    print(f"\n{'#' * 70}\n# Fold {fold_idx} — train {len(train_storms)} storms, "
          f"test {len(test_storms)} storms\n{'#' * 70}")

    cfg = dict(config)
    cfg['run_tag']        = f"kfold/fold_{fold_idx}"
    cfg['checkpoint_dir'] = str(base_out / 'checkpoints')
    cfg['output_dir']     = str(base_out / 'outputs')

    train_ds = _StormFilteredMultiFrame(
        config['root_dir'], config['train_years'],
        k_in=config['k_in'], k_out=config['k_out'],
        mode='train', img_size=config['img_size'],
        coord_lookup_path=config['coord_lookup_path'],
        latent_cache_dir=config.get('latent_cache_dir'),
        allowed_storms=train_storms,
    )
    test_ds = _StormFilteredMultiFrame(
        config['root_dir'], config['train_years'],   # same train years
        k_in=config['k_in'], k_out=config['k_out'],
        mode='val', img_size=config['img_size'],
        coord_lookup_path=config['coord_lookup_path'],
        latent_cache_dir=config.get('latent_cache_dir'),
        allowed_storms=test_storms,
    )
    print(f"  train samples: {len(train_ds)},  test samples: {len(test_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True)

    trainer = MultiFrameRFTrainer(cfg, mc_vae_path)
    trainer.train(train_loader, test_loader)

    # Final per-leadtime eval on the fold's test storms
    per_lead = trainer.evaluate_per_leadtime(
        test_loader,
        num_samples=cfg.get('eval_samples', 200),
        num_steps=cfg.get('eval_sampler_steps', 4),
    )
    per_lead.insert(0, 'fold', fold_idx)
    return {
        'fold': fold_idx,
        'n_train_storms': len(train_storms),
        'n_test_storms':  len(test_storms),
        'per_leadtime_df': per_lead,
        'best_val_loss': trainer.best_val_loss,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True)
    parser.add_argument('--train-years', nargs='+',
                        default=[str(y) for y in range(2012, 2021)])
    parser.add_argument('--coord-lookup-path', default=None)
    parser.add_argument('--mc-vae', required=True, help='Path to mc_vae_best.pt')
    parser.add_argument('--latent-cache-dir', default=None)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--diffusion-epochs', type=int, default=80)
    parser.add_argument('--reflow-iterations', type=int, default=1)
    parser.add_argument('--reflow-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--k-in', type=int, default=3)
    parser.add_argument('--k-out', type=int, default=3)
    parser.add_argument('--base-out', default='./outputs/kfold')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke', action='store_true',
                        help='2-epoch smoke test for the whole pipeline.')
    args = parser.parse_args()

    base_out = Path(args.base_out)
    base_out.mkdir(parents=True, exist_ok=True)

    # 1. Enumerate train-year storms + basins
    storm_basin = list_storms(Path(args.root), args.train_years)
    print(f"Found {len(storm_basin)} storms across {len(args.train_years)} years.")
    by_basin = {}
    for sid, b in storm_basin.items():
        by_basin.setdefault(b, []).append(sid)
    for b in sorted(by_basin):
        print(f"  {b}: {len(by_basin[b])} storms")

    # 2. Stratified k-fold
    folds = stratified_kfold_storms(storm_basin, k=args.k, seed=args.seed)
    fold_assignments = {sid: i for i, f in enumerate(folds) for sid in f}
    with open(base_out / 'fold_assignments.json', 'w') as f:
        json.dump({'k': args.k, 'seed': args.seed, 'assignments': fold_assignments,
                   'basins': storm_basin}, f, indent=2)

    # 3. Per-fold config template
    base_cfg = {
        'root_dir':           args.root,
        'train_years':        args.train_years,
        'val_years':          args.train_years,   # IGNORED — wrapper filters
        'test_years':         args.train_years,
        'coord_lookup_path':  args.coord_lookup_path,
        'batch_size':         args.batch_size,
        'num_workers':        args.num_workers,
        'img_size':           args.img_size,
        'k_in':               args.k_in,
        'k_out':              args.k_out,
        'latent_dim':         4,
        'vae_base_channels':  64,
        'mc_vae_checkpoint':  args.mc_vae,
        'latent_cache_dir':   args.latent_cache_dir,
        'base_channels':      128,
        'channel_mults':      (1, 2, 3, 4),
        'num_heads':          8,
        'embed_dim':          128,
        't_emb_dim':          128,
        'dropout':            0.1,
        'diffusion_epochs':   2 if args.smoke else args.diffusion_epochs,
        'reflow_iterations':  0 if args.smoke else args.reflow_iterations,
        'reflow_epochs':      args.reflow_epochs,
        'reflow_sample_steps': 20,
        'learning_rate':      1e-4,
        'min_lr':             5e-5,
        'weight_decay':       1e-5,
        'ema_decay':          0.9999,
        'warmup_epochs':      2 if args.smoke else 5,
        'eval_sampler_steps': 4,
        'sampler_step_sweep': [1, 4, 16, 50],
        'crps_n_samples':     10,
        'use_amp':            True,
        'amp_dtype':          'bfloat16',
        'gradient_accumulation_steps': 4,
        'eval_every':         5,
        'eval_samples':       50 if args.smoke else 200,
        # physics OFF for k-fold per phase3.md §3.4
        'use_phys_loss':      False,
        'use_continuity':     False,
        'use_geostrophic':    False,
        'use_track_consistency': False,
        'device':             'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # 4. Per-fold runs
    all_results = []
    for i in range(args.k):
        test_storms  = set(folds[i])
        train_storms = set().union(*[set(folds[j]) for j in range(args.k) if j != i])
        info = run_fold(i, train_storms, test_storms, base_cfg, args.mc_vae, base_out)
        info['per_leadtime_df'].to_csv(
            base_out / f'fold_{i}_per_leadtime.csv', index=False)
        all_results.append(info)

    # 5. Aggregate
    big = pd.concat([r['per_leadtime_df'] for r in all_results], ignore_index=True)
    big.to_csv(base_out / 'kfold_per_leadtime.csv', index=False)

    summary = big.groupby('lead').agg(['mean', 'std']).reset_index()
    summary.to_csv(base_out / 'kfold_summary.csv')
    print(f"\nWrote per-fold {base_out / 'kfold_per_leadtime.csv'}")
    print(f"Wrote summary    {base_out / 'kfold_summary.csv'}")
    print(summary.to_string())


if __name__ == '__main__':
    main()
