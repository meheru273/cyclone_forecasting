"""
EpochLogger — shared per-epoch metric logger for every trainer in the pipeline.

Outputs per training run:

    {run_tag}_metrics.csv   one row per epoch, rich training stats + val metrics
    {run_tag}_detailed.log  human-readable summary line per epoch

The CSV is rewritten in full every epoch so a mid-training crash still leaves
a complete record up to the last completed epoch.

Standard fields written per epoch:

    epoch, phase, wall_time_sec, n_batches, n_nan_skipped, lr
    train_loss_mean, train_loss_std, train_loss_min, train_loss_max
    grad_norm_mean, grad_norm_max
    [val_*]  any keys in the dict passed to end_epoch(val_metrics=...)
    [extra]  any keyword args passed to end_epoch()

Usage pattern in a trainer:

    self.logger = EpochLogger(self.metrics_dir, run_tag='vae')

    def train(self, train_loader, val_loader):
        for epoch in range(num_epochs):
            self.logger.start_epoch()
            self.train_epoch(...)
            val_metrics = self.validate(...)
            self.logger.end_epoch(
                epoch,
                val_metrics=val_metrics,
                lr=self.optimizer.param_groups[0]['lr'],
                phase='initial',          # or 'reflow_1', 'vae', etc.
                # any other scalar columns you want recorded
            )

    def train_epoch(self, loader, epoch):
        for batch in loader:
            ...
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            self.optimizer.step()
            self.logger.log_batch(loss=loss.item(), grad_norm=grad_norm.item())
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _safe_stat(arr, fn, default=float('nan')):
    if not arr:
        return default
    try:
        return float(fn(arr))
    except (TypeError, ValueError):
        return default


def _fmt_value(v):
    if v is None:
        return 'None'
    if isinstance(v, float):
        if np.isnan(v):
            return 'nan'
        return f'{v:.4g}'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


class EpochLogger:
    """One-instance-per-trainer metric logger."""

    def __init__(self, metrics_dir, run_tag: str):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.run_tag = run_tag
        self.csv_path = self.metrics_dir / f'{run_tag}_metrics.csv'
        self.txt_path = self.metrics_dir / f'{run_tag}_detailed.log'
        self.history: list[dict] = []
        self._batch_losses: list[float] = []
        self._batch_grads:  list[float] = []
        self._n_nan_skipped = 0
        self._epoch_start: Optional[float] = None

    # ----- batch-level ingest -----
    def start_epoch(self):
        self._epoch_start = time.time()
        self._batch_losses = []
        self._batch_grads = []
        self._n_nan_skipped = 0

    def log_batch(self, loss: Optional[float] = None,
                  grad_norm: Optional[float] = None):
        if loss is not None:
            if not np.isfinite(loss):
                self._n_nan_skipped += 1
            else:
                self._batch_losses.append(float(loss))
        if grad_norm is not None and np.isfinite(grad_norm):
            self._batch_grads.append(float(grad_norm))

    def log_nan_skip(self):
        """Use when the trainer skips a batch *before* computing loss
        (e.g. detects bad input)."""
        self._n_nan_skipped += 1

    # ----- epoch-level ingest -----
    def end_epoch(self, epoch: int, *,
                  eval_metrics: Optional[dict] = None,
                  val_metrics:  Optional[dict] = None,    # alias, kept for callers
                  learning_rate: Optional[float] = None,
                  lr:            Optional[float] = None,  # alias
                  phase:         Optional[str] = None,
                  is_best:       Optional[bool] = None,
                  **extra: Any) -> dict:
        """Append one row to the per-epoch CSV + .log.

        Standard columns (always present):
            epoch, phase, wall_time_sec, n_batches, n_nan_skipped,
            learning_rate, is_best,
            train_loss_mean/std/min/max, grad_norm_mean/max
        Plus:
            eval_<key> for every key in eval_metrics (or val_metrics alias)
            <key>       for every kwarg in **extra (e.g. latent_scale_factor)
        """
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        bl = self._batch_losses
        gn = self._batch_grads
        # Aliases: accept both lr/learning_rate and val_metrics/eval_metrics
        lr_value = learning_rate if learning_rate is not None else lr
        emetrics = eval_metrics if eval_metrics is not None else val_metrics

        row: dict[str, Any] = {
            'epoch':           epoch,
            'phase':           phase,
            'wall_time_sec':   round(elapsed, 2),
            'n_batches':       len(bl),
            'n_nan_skipped':   self._n_nan_skipped,
            'learning_rate':   lr_value,
            'is_best':         is_best,
            'train_loss_mean': _safe_stat(bl, np.mean),
            'train_loss_std':  _safe_stat(bl, np.std),
            'train_loss_min':  _safe_stat(bl, np.min),
            'train_loss_max':  _safe_stat(bl, np.max),
            'grad_norm_mean':  _safe_stat(gn, np.mean),
            'grad_norm_max':   _safe_stat(gn, np.max),
        }
        if emetrics:
            for k, v in emetrics.items():
                key = k if k.startswith('eval_') else f'eval_{k}'
                row[key] = v
        for k, v in extra.items():
            row[k] = v

        self.history.append(row)

        # Per-epoch CSV write (overwrites; new row appended to history).
        # Use a temp file + rename for crash-safety on Windows.
        tmp = self.csv_path.with_suffix('.csv.tmp')
        pd.DataFrame(self.history).to_csv(tmp, index=False)
        tmp.replace(self.csv_path)

        # Append a one-line summary to the text log.
        try:
            with open(self.txt_path, 'a', encoding='utf-8') as f:
                f.write(self._format_line(row) + '\n')
        except OSError:
            pass

        return row

    def _format_line(self, row):
        # Build "key=val | key=val" in a deterministic, readable order.
        priority = [
            'epoch', 'phase', 'wall_time_sec', 'n_batches', 'n_nan_skipped',
            'learning_rate', 'is_best',
            'train_loss_mean', 'train_loss_std',
            'train_loss_min',  'train_loss_max',
            'grad_norm_mean',  'grad_norm_max',
        ]
        seen = set()
        chunks = []
        for k in priority:
            if k in row and row[k] is not None:
                chunks.append(f'{k}={_fmt_value(row[k])}')
                seen.add(k)
        for k, v in row.items():
            if k in seen:
                continue
            if v is None:
                continue
            chunks.append(f'{k}={_fmt_value(v)}')
        return ' | '.join(chunks)
