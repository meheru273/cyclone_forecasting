"""
Phase-3 statistical significance tests for forecast-model comparison.

Three procedures (all cluster-aware — per-sample metrics on consecutive frames
of the same storm are NOT independent):

  - Paired storm-level bootstrap CIs
  - Diebold-Mariano test with Newey-West (HAC) variance
  - Wilcoxon signed-rank (per-storm) as a sanity check

Input format (CSV per run):
    storm, lead, channel, mse, psnr, ssim, ... <other metric columns>
The 'storm' column groups observations for cluster bootstrap and Wilcoxon.

Usage:
    python scripts/analysis/significance_tests.py \\
        --a outputs/mf_kin3_kout3_2rf_no_phys/metrics/per_storm.csv \\
        --b outputs/nath_oursetting/metrics/per_storm.csv \\
        --metric overall_psnr \\
        --lead 1 \\
        --out significance_A_vs_B_lead1_psnr.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Per-storm aggregation
# ---------------------------------------------------------------------------

def per_storm_means(df: pd.DataFrame, metric: str, lead: Optional[int] = None) -> pd.Series:
    """Reduce a per-sample CSV to one row per storm by averaging `metric`
    over that storm's val frames. If `lead` is set, filter to that lead first."""
    sub = df if lead is None else df[df['lead'] == lead]
    if metric not in sub.columns:
        raise KeyError(f"metric '{metric}' not in CSV columns: {list(sub.columns)}")
    return sub.groupby('storm')[metric].mean()


# ---------------------------------------------------------------------------
# 1. Paired storm-level bootstrap CI
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(diffs: np.ndarray,
                        n_bootstrap: int = 10_000,
                        ci: float = 0.95,
                        rng_seed: int = 0) -> dict:
    """Cluster-aware (storm-level) bootstrap of paired metric differences.

    `diffs` is a 1-D array of within-storm metric differences
    `M_A(storm) - M_B(storm)`. We resample storms (with replacement),
    compute the mean of each resample, and report the empirical CI of those
    bootstrap means.

    Returns dict {mean, ci_low, ci_high, p_value_two_sided, n_storms}.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2:
        return {'mean': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'),
                'p_value_two_sided': float('nan'), 'n_storms': int(n)}

    rng = np.random.default_rng(rng_seed)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1.0 - alpha))

    # Two-sided p-value via bootstrap inversion: 2 * min(P(mean<=0), P(mean>=0))
    p_lo = float(np.mean(boot_means <= 0.0))
    p_hi = float(np.mean(boot_means >= 0.0))
    p_two = float(2.0 * min(p_lo, p_hi))
    return {
        'mean':              float(diffs.mean()),
        'ci_low':            lo,
        'ci_high':           hi,
        'p_value_two_sided': p_two,
        'n_storms':          int(n),
    }


# ---------------------------------------------------------------------------
# 2. Diebold-Mariano with Newey-West HAC variance
# ---------------------------------------------------------------------------

def _newey_west_variance(d: np.ndarray, max_lag: int) -> float:
    """HAC variance estimator (Newey-West, Bartlett kernel)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return float('nan')
    mean = d.mean()
    centred = d - mean
    gamma0 = float(np.mean(centred ** 2))
    var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)         # Bartlett weight
        gamma_l = float(np.mean(centred[lag:] * centred[:-lag]))
        var += 2.0 * w * gamma_l
    return max(var, 1e-12) / n


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray,
                    autocorr_lag: int = 2) -> dict:
    """Diebold-Mariano forecast-comparison test.

    H0: E[loss_a - loss_b] = 0 (equal forecast accuracy).

    Args:
        loss_a, loss_b: per-sample losses (1-D arrays of same length).
        autocorr_lag:   max lag for the Newey-West variance estimator.
                        For 3-hourly frames within a storm, lag=1 or 2
                        usually suffices.

    Returns dict {dm_stat, p_value_two_sided, mean_diff, n}.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b))
    a = a[mask]; b = b[mask]
    d = a - b
    n = len(d)
    if n < 2:
        return {'dm_stat': float('nan'), 'p_value_two_sided': float('nan'),
                'mean_diff': float('nan'), 'n': int(n)}
    var_hac = _newey_west_variance(d, autocorr_lag)
    dm = float(d.mean() / np.sqrt(var_hac))
    p  = float(2.0 * (1.0 - stats.norm.cdf(abs(dm))))
    return {'dm_stat': dm, 'p_value_two_sided': p,
            'mean_diff': float(d.mean()), 'n': int(n)}


# ---------------------------------------------------------------------------
# 3. Wilcoxon signed-rank (storm-level)
# ---------------------------------------------------------------------------

def wilcoxon_storm_level(diffs: np.ndarray) -> dict:
    """Non-parametric paired test on per-storm metric differences."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return {'statistic': float('nan'), 'p_value': float('nan'),
                'n_storms': int(len(diffs))}
    try:
        stat, p = stats.wilcoxon(diffs, zero_method='wilcox',
                                 alternative='two-sided')
        return {'statistic': float(stat), 'p_value': float(p),
                'n_storms': int(len(diffs))}
    except ValueError as e:
        # raised if all diffs are zero
        return {'statistic': float('nan'), 'p_value': float('nan'),
                'n_storms': int(len(diffs)), 'note': str(e)}


# ---------------------------------------------------------------------------
# Convenience top-level: one function = one (A vs B, metric, lead) result block
# ---------------------------------------------------------------------------

def compare_runs(df_a: pd.DataFrame, df_b: pd.DataFrame,
                 metric: str, lead: Optional[int] = None,
                 n_bootstrap: int = 10_000,
                 autocorr_lag: int = 2,
                 rng_seed: int = 0) -> dict:
    """Run all three tests on one (metric, lead) pair from two per-storm CSVs."""
    storm_a = per_storm_means(df_a, metric, lead)
    storm_b = per_storm_means(df_b, metric, lead)
    common = storm_a.index.intersection(storm_b.index)
    if len(common) == 0:
        raise ValueError("No overlapping storms between the two runs.")
    a = storm_a.loc[common].to_numpy()
    b = storm_b.loc[common].to_numpy()
    diffs = a - b

    out = {
        'metric': metric,
        'lead': lead,
        'mean_a': float(np.nanmean(a)),
        'mean_b': float(np.nanmean(b)),
        'bootstrap': paired_bootstrap_ci(diffs, n_bootstrap=n_bootstrap,
                                          rng_seed=rng_seed),
        'wilcoxon':  wilcoxon_storm_level(diffs),
    }

    # DM on the per-sample (frame-level) loss if the CSVs have it.
    # Lower-is-better metrics (mse, lpips, crps, *err*, *km*) are losses directly;
    # higher-is-better metrics (psnr, ssim, csi, pod) are negated.
    higher_is_better_keys = ('psnr', 'ssim', 'csi', 'pod')
    sign = -1.0 if any(k in metric.lower() for k in higher_is_better_keys) else 1.0
    try:
        sub_a = df_a if lead is None else df_a[df_a['lead'] == lead]
        sub_b = df_b if lead is None else df_b[df_b['lead'] == lead]
        # align by (storm, lead) — outer join + per-storm fill not handled here.
        # Simpler: sort by storm + lead in both, take aligned vectors of equal length
        # by inner-join on (storm, lead) where possible.
        joined = pd.merge(sub_a[['storm', 'lead', metric]],
                          sub_b[['storm', 'lead', metric]],
                          on=['storm', 'lead'], suffixes=('_a', '_b'))
        loss_a = sign * joined[f'{metric}_a'].to_numpy()
        loss_b = sign * joined[f'{metric}_b'].to_numpy()
        out['diebold_mariano'] = diebold_mariano(loss_a, loss_b,
                                                  autocorr_lag=autocorr_lag)
    except Exception as e:
        out['diebold_mariano'] = {'note': f'DM skipped: {e}'}

    return out


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', required=True, help='Per-storm CSV for model A')
    parser.add_argument('--b', required=True, help='Per-storm CSV for model B')
    parser.add_argument('--metric', required=True,
                        help='Column name to compare (e.g. overall_psnr, gridsat_lpips, track_km)')
    parser.add_argument('--lead', type=int, default=None,
                        help='Lead time to filter to (1, 2, or 3). Omit = pool all leads.')
    parser.add_argument('--n-bootstrap', type=int, default=10_000)
    parser.add_argument('--autocorr-lag', type=int, default=2,
                        help='Newey-West lag for DM test (3-hourly frames within a storm).')
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--out', required=True, help='Output JSON path')
    args = parser.parse_args()

    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)

    result = compare_runs(df_a, df_b, args.metric, lead=args.lead,
                          n_bootstrap=args.n_bootstrap,
                          autocorr_lag=args.autocorr_lag,
                          rng_seed=args.rng_seed)
    result['paths'] = {'a': str(args.a), 'b': str(args.b)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
