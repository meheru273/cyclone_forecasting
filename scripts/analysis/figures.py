"""
Phase-3 figures generator.

Produces the 13 plots listed in docs/phase3.md §3.7 from the aggregated CSVs:

  - lead_time_metrics.png     (Fig 3)
  - rapsd_loglog.png          (Fig 4)
  - sampler_budget.png        (Fig 5)
  - kfold_box.png             (Fig 6)
  - track_overlay_basemap.png (Fig 7) — requires cartopy; falls back to a plain plot
  - reliability_diagram.png   (Fig 8)
  - spatial_error_heatmap.png (Fig 9)
  - per_channel_ecdf.png      (Fig 10)
  - failure_cases.png         (Fig 11)
  - cross_basin_bars.png      (Fig 12)
  - physics_loss_ablation.png (Fig 13)

Each figure is a separate function so the user can call them individually:
    python scripts/analysis/figures.py --kind lead_time \\
        --in paper_results/per_leadtime_summary.csv --out paper_results/figs/

Or generate all at once:
    python scripts/analysis/figures.py --all --in-dir paper_results/ \\
        --out paper_results/figs/

For figures that need per-sample data not in the aggregated CSVs (failure
cases, spatial heatmaps, ECDFs), the script also accepts a `--samples-npz`
file produced by a separate dump step in the trainer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fig 3: Lead-time × metric curves (PSNR, SSIM, CRPS, track-km)
# ---------------------------------------------------------------------------

def lead_time_metrics(per_lead_csv: str, out_path: str,
                      metrics=('overall_psnr', 'overall_ssim', 'crps', 'track_km')):
    df = pd.read_csv(per_lead_csv)
    runs = df['run_id'].unique()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        for r in runs:
            sub = df[df['run_id'] == r].sort_values('lead')
            if m not in sub.columns:
                continue
            ax.plot(sub['lead'], sub[m], marker='o', label=r)
        ax.set_xlabel('Lead time (× 3 h)')
        ax.set_ylabel(m)
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Fig 5: Sampler budget × quality
# ---------------------------------------------------------------------------

def sampler_budget(sweep_csv: str, out_path: str):
    df = pd.read_csv(sweep_csv)
    runs = df['run_id'].unique() if 'run_id' in df.columns else ['(only run)']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for r in runs:
        sub = df if 'run_id' not in df.columns else df[df['run_id'] == r]
        sub = sub.sort_values('n_steps')
        axes[0].plot(sub['n_steps'], sub['overall_psnr'], marker='o', label=r)
        axes[1].plot(sub['n_steps'], sub['wallclock_ms_per_sample'],
                     marker='s', label=r)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('ODE steps')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Overall PSNR')
    axes[0].set_title('Quality vs sampler steps')
    axes[1].set_ylabel('Wallclock (ms / sample)')
    axes[1].set_title('Cost vs sampler steps')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Fig 6: k-fold box plot
# ---------------------------------------------------------------------------

def kfold_box(kfold_csv: str, out_path: str,
              metrics=('overall_psnr', 'overall_ssim', 'crps', 'track_km')):
    df = pd.read_csv(kfold_csv)   # expected cols: fold, metric, value
    if 'metric' in df.columns and 'value' in df.columns:
        wide = df.pivot_table(index='fold', columns='metric', values='value')
    else:
        wide = df.set_index('fold')[list(metrics)]
    fig, ax = plt.subplots(figsize=(2.5 * len(metrics), 4))
    cols = [m for m in metrics if m in wide.columns]
    ax.boxplot([wide[m].dropna().to_numpy() for m in cols], labels=cols)
    ax.set_ylabel('Per-fold value')
    ax.set_title(f'Storm-level k-fold (k={len(wide)})')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Fig 4 / 10: RAPSD log-log + per-channel ECDFs
# ---------------------------------------------------------------------------

def rapsd_loglog(samples_npz: str, out_path: str, channels=('GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres')):
    """Expects an .npz with keys 'rapsd_pred', 'rapsd_target' each shape
    (n_channels, n_radial_bins)."""
    data = np.load(samples_npz)
    p = data['rapsd_pred']      # (C, R)
    t = data['rapsd_target']    # (C, R)
    n_c = p.shape[0]
    fig, axes = plt.subplots(1, n_c, figsize=(3.2 * n_c, 3.5))
    if n_c == 1:
        axes = [axes]
    radii = np.arange(1, p.shape[1] + 1)
    for c, ax in enumerate(axes):
        ax.loglog(radii, t[c] + 1e-12, 'k-',  label='Target', linewidth=2)
        ax.loglog(radii, p[c] + 1e-12, 'r--', label='Predicted', linewidth=2)
        ax.set_title(channels[c] if c < len(channels) else f'ch{c}')
        ax.set_xlabel('radial bin')
        ax.set_ylabel('Power')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


def per_channel_ecdf(samples_npz: str, out_path: str,
                     channels=('GRIDSAT', 'U-wind', 'V-wind', 'Temp', 'Pres')):
    """Expects .npz with 'pred_values' and 'target_values' each (C, N)."""
    d = np.load(samples_npz)
    p = d['pred_values']; t = d['target_values']
    n_c = p.shape[0]
    fig, axes = plt.subplots(1, n_c, figsize=(3.2 * n_c, 3.5))
    if n_c == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        for arr, lab, sty in [(t[c], 'Target', '-'), (p[c], 'Pred', '--')]:
            x = np.sort(arr.flatten())
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, sty, label=lab)
        ax.set_title(channels[c] if c < len(channels) else f'ch{c}')
        ax.set_xlabel('value')
        ax.set_ylabel('ECDF')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Fig 8: reliability diagram (for the headline IR threshold)
# ---------------------------------------------------------------------------

def reliability_diagram(reliability_csv: str, out_path: str):
    """Expects CSV with columns 'pred_prob_bin', 'observed_freq', 'count'."""
    df = pd.read_csv(reliability_csv)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.plot(df['pred_prob_bin'], df['observed_freq'], 'o-', label='Model')
    # bar inset for count distribution
    ax.bar(df['pred_prob_bin'], df['count'] / df['count'].max() * 0.2,
           width=0.04, alpha=0.3, color='gray', label='Frequency')
    ax.set_xlabel('Predicted CDF')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Reliability diagram (IR < 220 K)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Fig 9: spatial error heatmap (val-averaged)
# ---------------------------------------------------------------------------

def spatial_error_heatmap(samples_npz: str, out_path: str):
    """Expects 'spatial_mae' of shape (C, H, W) averaged over val samples."""
    d = np.load(samples_npz)
    e = d['spatial_mae']   # (C, H, W)
    n_c = e.shape[0]
    fig, axes = plt.subplots(1, n_c, figsize=(3.0 * n_c, 3.0))
    if n_c == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        im = ax.imshow(e[c], cmap='magma')
        ax.set_title(f'ch{c} val-averaged |err|')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Fig 11: failure-case gallery
# ---------------------------------------------------------------------------

def failure_cases(samples_npz: str, out_path: str, n_show=6):
    """Expects 'worst_pred', 'worst_target', 'worst_input' each (N, H, W),
    'worst_storm_ids' (N,), 'worst_psnr' (N,) — sorted ascending PSNR."""
    d = np.load(samples_npz, allow_pickle=True)
    pred = d['worst_pred'][:n_show]
    tgt  = d['worst_target'][:n_show]
    inp  = d['worst_input'][:n_show]
    sids = d['worst_storm_ids'][:n_show]
    psnr = d['worst_psnr'][:n_show]
    fig, axes = plt.subplots(3, n_show, figsize=(2.5 * n_show, 7.5))
    for s in range(n_show):
        axes[0, s].imshow(inp[s],  cmap='gray'); axes[0, s].set_title(f"{sids[s]}", fontsize=8)
        axes[1, s].imshow(tgt[s],  cmap='gray')
        axes[2, s].imshow(pred[s], cmap='gray'); axes[2, s].set_title(f"PSNR={psnr[s]:.2f}", fontsize=8)
        for r in range(3):
            axes[r, s].set_xticks([]); axes[r, s].set_yticks([])
    axes[0, 0].set_ylabel('Input',     fontsize=9, fontweight='bold')
    axes[1, 0].set_ylabel('Target',    fontsize=9, fontweight='bold')
    axes[2, 0].set_ylabel('Predicted', fontsize=9, fontweight='bold')
    plt.suptitle('Worst-PSNR predictions', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Fig 7: track overlay (tries cartopy, falls back to plain)
# ---------------------------------------------------------------------------

def track_overlay(track_csv: str, out_path: str):
    """Expects CSV with columns 'storm', 'lead', 'gt_lat', 'gt_lon', 'pred_lat', 'pred_lon'."""
    df = pd.read_csv(track_csv)
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)
        for storm, g in df.groupby('storm'):
            g = g.sort_values('lead')
            ax.plot(g['gt_lon'],   g['gt_lat'],   'k-o', markersize=4, transform=ccrs.PlateCarree())
            ax.plot(g['pred_lon'], g['pred_lat'], 'r--s', markersize=4, transform=ccrs.PlateCarree())
        ax.set_title('Predicted vs Truth Tracks (red = predicted)')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Wrote {out_path} (cartopy)")
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 7))
        for storm, g in df.groupby('storm'):
            g = g.sort_values('lead')
            ax.plot(g['gt_lon'],   g['gt_lat'],   'k-o', markersize=4, label='GT'   if storm == df['storm'].iloc[0] else None)
            ax.plot(g['pred_lon'], g['pred_lat'], 'r--s', markersize=4, label='Pred' if storm == df['storm'].iloc[0] else None)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title('Predicted vs Truth Tracks (cartopy unavailable — plain axes)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Wrote {out_path} (plain axes, cartopy missing)")


# ---------------------------------------------------------------------------
# Fig 12: cross-basin degradation bars
# ---------------------------------------------------------------------------

def cross_basin_bars(cross_basin_csv: str, out_path: str,
                     metric='overall_psnr'):
    """CSV: columns 'basin', 'split' (in-basin / OOD), 'value'."""
    df = pd.read_csv(cross_basin_csv)
    basins = sorted(df['basin'].unique())
    in_vals  = [df[(df['basin'] == b) & (df['split'] == 'in-basin')][metric].iloc[0]
                if not df[(df['basin'] == b) & (df['split'] == 'in-basin')].empty else 0
                for b in basins]
    ood_vals = [df[(df['basin'] == b) & (df['split'] == 'OOD')][metric].iloc[0]
                if not df[(df['basin'] == b) & (df['split'] == 'OOD')].empty else 0
                for b in basins]
    x = np.arange(len(basins))
    w = 0.35
    fig, ax = plt.subplots(figsize=(2 + 0.8 * len(basins), 4))
    ax.bar(x - w/2, in_vals,  w, label='In-basin')
    ax.bar(x + w/2, ood_vals, w, label='Held-out basin')
    ax.set_xticks(x)
    ax.set_xticklabels(basins)
    ax.set_ylabel(metric)
    ax.set_title('Cross-basin generalisation')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Fig 13: physics-loss ablation
# ---------------------------------------------------------------------------

def physics_loss_ablation(phys_csv: str, out_path: str,
                          metrics=('overall_psnr', 'overall_ssim', 'track_km',
                                   'wind_divergence_peak')):
    df = pd.read_csv(phys_csv)   # columns: ablation, metric, value
    if 'metric' in df.columns and 'value' in df.columns:
        wide = df.pivot_table(index='ablation', columns='metric', values='value')
    else:
        wide = df.set_index('ablation')[[m for m in metrics if m in df.columns]]
    cols = [m for m in metrics if m in wide.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, m in zip(axes, cols):
        wide[m].plot(kind='bar', ax=ax)
        ax.set_title(m)
        ax.set_ylabel(m)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

KIND_TO_FN = {
    'lead_time':      lead_time_metrics,
    'sampler':        sampler_budget,
    'kfold':          kfold_box,
    'rapsd':          rapsd_loglog,
    'ecdf':           per_channel_ecdf,
    'reliability':    reliability_diagram,
    'spatial':        spatial_error_heatmap,
    'failure':        failure_cases,
    'track_overlay':  track_overlay,
    'cross_basin':    cross_basin_bars,
    'physics':        physics_loss_ablation,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', choices=list(KIND_TO_FN.keys()),
                        help='Which figure to generate')
    parser.add_argument('--in', dest='input', help='Input CSV / NPZ path')
    parser.add_argument('--out', help='Output PNG path')
    args = parser.parse_args()

    if args.kind is None or args.input is None or args.out is None:
        raise SystemExit("--kind, --in, --out all required.")
    fn = KIND_TO_FN[args.kind]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fn(args.input, args.out)


if __name__ == '__main__':
    main()
