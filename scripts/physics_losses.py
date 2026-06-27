"""
Phase-2 physics-soft losses for the multi-frame rectified-flow model.

All three losses are differentiable, finite-difference-based, and operate on
*decoded* (post-VAE) predictions in their normalised z-score space. They are
denormalised internally where physical units matter (geostrophic, track).

Loss components (each independently flag-toggleable):
  - L_continuity : ∂U/∂x + ∂V/∂y ≈ 0   (mass-continuity, 2-D barotropic)
  - L_geostrophic: f·U ≈ -∂P/∂y/ρ, f·V ≈ ∂P/∂x/ρ
  - L_track      : predicted steering-flow integral matches IBTrACS Δposition

Master switch `use_phys_loss`; per-component switches `use_continuity`,
`use_geostrophic`, `use_track_consistency`. **Default OFF.**

Channel layout: (B, 5, H, W)
    ch 0 = GRIDSAT IR
    ch 1 = U-wind     (m/s)
    ch 2 = V-wind     (m/s)
    ch 3 = Temperature (K)
    ch 4 = Pressure   (Pa)

For multi-frame the wrapper expects shape (B, K_out, 5, H, W) and applies
each component per lead-time, averaging.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Finite-difference helpers
# ---------------------------------------------------------------------------

def _ddx(field: torch.Tensor) -> torch.Tensor:
    """Central finite difference along the last (x) axis. Same shape as input.
    Edges use one-sided differences."""
    pad = F.pad(field, (1, 1), mode='replicate')
    return (pad[..., 2:] - pad[..., :-2]) * 0.5


def _ddy(field: torch.Tensor) -> torch.Tensor:
    """Central finite difference along the second-to-last (y) axis."""
    pad = F.pad(field, (0, 0, 1, 1), mode='replicate')
    return (pad[..., 2:, :] - pad[..., :-2, :]) * 0.5


def _denormalise(field: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return field * std + mean


# ---------------------------------------------------------------------------
# Mass continuity:  ∂U/∂x + ∂V/∂y ≈ 0
# ---------------------------------------------------------------------------

def continuity_loss(pred_5ch: torch.Tensor,
                    era5_mean: Sequence[float],
                    era5_std: Sequence[float],
                    core_mask_size: int = 16) -> torch.Tensor:
    """L_continuity = mean(div(U_pred, V_pred)^2) outside the masked storm core.

    Args:
        pred_5ch: (B, 5, H, W), z-scored
        era5_mean, era5_std: ERA5 normalisation stats, length-4 (U, V, T, P)
        core_mask_size: side length (px) of the central region to mask out
            (where real meteorological divergence is large)
    """
    U = _denormalise(pred_5ch[:, 1], float(era5_mean[0]), float(era5_std[0]))
    V = _denormalise(pred_5ch[:, 2], float(era5_mean[1]), float(era5_std[1]))

    div = _ddx(U) + _ddy(V)                       # (B, H, W)

    # Build a mask that zeros out the central core_mask_size x core_mask_size box
    H, W = div.shape[-2], div.shape[-1]
    mask = torch.ones_like(div)
    y0 = (H - core_mask_size) // 2
    x0 = (W - core_mask_size) // 2
    mask[..., y0:y0 + core_mask_size, x0:x0 + core_mask_size] = 0.0

    # mean over masked pixels only
    sq = div ** 2
    denom = mask.sum().clamp(min=1.0)
    return (sq * mask).sum() / denom


# ---------------------------------------------------------------------------
# Geostrophic balance:  f·U ≈ -∂P/∂y/ρ,  f·V ≈ ∂P/∂x/ρ
# ---------------------------------------------------------------------------

OMEGA  = 7.2921e-5    # Earth's angular velocity (rad/s)
RHO    = 1.2          # near-surface air density (kg/m^3)
DEG2KM = 111.0        # 1 deg latitude ≈ 111 km
DX_M   = 25_000.0     # rough pixel size in metres at 256² (~0.25 deg @ tropics);
                      # used only to scale finite differences to per-metre.
                      # In practice this only affects the loss *magnitude*, not
                      # the optimum, so it is absorbed by lambda_geostrophic.


def geostrophic_loss(pred_5ch: torch.Tensor,
                     cur_lat: torch.Tensor,
                     era5_mean: Sequence[float],
                     era5_std: Sequence[float],
                     core_mask_size: int = 32) -> torch.Tensor:
    """L_geo = mean( (f·U + dP/dy/ρ)^2 + (f·V - dP/dx/ρ)^2 ) outside core.

    Args:
        pred_5ch: (B, 5, H, W) z-scored predictions
        cur_lat:  (B,) latitudes in *degrees* (for Coriolis parameter)
        era5_mean, era5_std: length-4
        core_mask_size: mask side in pixels for the central core (gradient-wind
            balance dominates there, not geostrophic)
    """
    B, _, H, W = pred_5ch.shape
    U = _denormalise(pred_5ch[:, 1], float(era5_mean[0]), float(era5_std[0]))
    V = _denormalise(pred_5ch[:, 2], float(era5_mean[1]), float(era5_std[1]))
    P = _denormalise(pred_5ch[:, 4], float(era5_mean[3]), float(era5_std[3]))

    f = (2.0 * OMEGA * torch.sin(torch.deg2rad(cur_lat))).view(B, 1, 1)  # (B,1,1)

    dPdx = _ddx(P) / DX_M
    dPdy = _ddy(P) / DX_M

    res_x = f * U + dPdy / RHO
    res_y = f * V - dPdx / RHO

    sq = res_x ** 2 + res_y ** 2                  # (B, H, W)
    mask = torch.ones_like(sq)
    y0 = (H - core_mask_size) // 2
    x0 = (W - core_mask_size) // 2
    mask[..., y0:y0 + core_mask_size, x0:x0 + core_mask_size] = 0.0
    denom = mask.sum().clamp(min=1.0)
    return (sq * mask).sum() / denom


# ---------------------------------------------------------------------------
# Track consistency  (single-frame and multi-frame)
# ---------------------------------------------------------------------------

def track_consistency_loss(pred_5ch_seq: torch.Tensor,
                            cur_lat: torch.Tensor,
                            cur_lon: torch.Tensor,
                            target_lat: torch.Tensor,
                            target_lon: torch.Tensor,
                            era5_mean: Sequence[float],
                            era5_std: Sequence[float],
                            roi: int = 64,
                            dt_hours: float = 3.0) -> torch.Tensor:
    """Penalty: predicted-U/V steering-flow integral should reproduce the actual
    Δposition from IBTrACS.

    Single-frame call: pred_5ch_seq shape (B, 5, H, W) and target_lat/lon (B,).
    Multi-frame call:  pred_5ch_seq shape (B, K_out, 5, H, W),
                       target_lat/lon (B, K_out).

    Lead-time k uses k * dt_hours hours of integration.
    """
    u_mean = float(era5_mean[0]); u_std = float(era5_std[0])
    v_mean = float(era5_mean[1]); v_std = float(era5_std[1])

    if pred_5ch_seq.dim() == 4:                   # single frame
        pred_5ch_seq = pred_5ch_seq.unsqueeze(1)  # (B, 1, 5, H, W)
        target_lat   = target_lat.unsqueeze(1)    # (B, 1)
        target_lon   = target_lon.unsqueeze(1)

    B, K, _, H, W = pred_5ch_seq.shape
    y0 = (H - roi) // 2
    x0 = (W - roi) // 2
    losses = []
    for k in range(K):
        U = _denormalise(pred_5ch_seq[:, k, 1, y0:y0+roi, x0:x0+roi], u_mean, u_std)
        V = _denormalise(pred_5ch_seq[:, k, 2, y0:y0+roi, x0:x0+roi], v_mean, v_std)
        dt_sec = (k + 1) * dt_hours * 3600.0      # lead-time scaling
        # predicted displacement (deg)
        cos_lat = torch.cos(torch.deg2rad(cur_lat)).clamp(min=0.1)
        d_lat_pred = V.mean(dim=(1, 2)) * dt_sec / 111_000.0
        d_lon_pred = U.mean(dim=(1, 2)) * dt_sec / (111_000.0 * cos_lat)
        # ground-truth displacement (deg)
        d_lat_gt = target_lat[:, k] - cur_lat
        d_lon_gt = target_lon[:, k] - cur_lon
        losses.append(((d_lat_pred - d_lat_gt) ** 2
                       + (d_lon_pred - d_lon_gt) ** 2).mean())
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Master driver
# ---------------------------------------------------------------------------

class PhysicsLossDriver:
    """Wraps the three losses behind one switchable interface.

    Usage in a trainer:

        phys = PhysicsLossDriver(config, era5_mean, era5_std)
        ...
        loss = velocity_loss
        if phys.enabled:
            extra = phys.compute(
                pred_5ch=decoded_pred,        # (B, 5, H, W) or (B, K_out, 5, H, W)
                cur_lat=batch['raw_lat'],      # (B,) degrees
                cur_lon=batch['raw_lon'],      # (B,) degrees
                target_lat=batch['raw_target_lat'],  # (B,) or (B, K_out)
                target_lon=batch['raw_target_lon'],
                epoch=epoch,
            )
            loss = loss + extra['total']
    """

    def __init__(self, config: dict,
                 era5_mean: Sequence[float],
                 era5_std:  Sequence[float]):
        self.cfg = config
        self.era5_mean = era5_mean
        self.era5_std  = era5_std

        self.use_phys_loss     = bool(config.get('use_phys_loss', False))
        self.use_continuity    = bool(config.get('use_continuity', False))
        self.use_geostrophic   = bool(config.get('use_geostrophic', False))
        self.use_track         = bool(config.get('use_track_consistency', False))

        self.lambda_cont  = float(config.get('lambda_continuity', 1e-3))
        self.lambda_geo   = float(config.get('lambda_geostrophic', 1e-4))
        self.lambda_track = float(config.get('lambda_track', 1e-2))

        self.warmup_epochs = int(config.get('phys_warmup_epochs', 10))

        # Empirical per-loss scales — set after one warm-up batch via
        # `calibrate_scales`. Until then we use 1.0 (no rescaling).
        self.scale_cont  = 1.0
        self.scale_geo   = 1.0
        self.scale_track = 1.0
        self._calibrated = False

    @property
    def enabled(self) -> bool:
        return self.use_phys_loss and (self.use_continuity
                                        or self.use_geostrophic
                                        or self.use_track)

    def _warmup_factor(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, max(epoch, 0) / float(self.warmup_epochs))

    @torch.no_grad()
    def calibrate_scales(self, sample_loss_values: dict):
        """Once, after the first training batch, call this to set the per-loss
        empirical scales so the lambdas are interpretable across runs."""
        for k, v in sample_loss_values.items():
            v_pos = max(abs(float(v)), 1e-9)
            if   k == 'continuity':  self.scale_cont  = v_pos
            elif k == 'geostrophic': self.scale_geo   = v_pos
            elif k == 'track':       self.scale_track = v_pos
        self._calibrated = True
        print(f"  [PhysicsLoss] scales calibrated: "
              f"continuity={self.scale_cont:.4g}, "
              f"geostrophic={self.scale_geo:.4g}, "
              f"track={self.scale_track:.4g}")

    def compute(self,
                pred_5ch: torch.Tensor,
                cur_lat: torch.Tensor,
                cur_lon: torch.Tensor,
                target_lat: Optional[torch.Tensor] = None,
                target_lon: Optional[torch.Tensor] = None,
                epoch: int = 0) -> dict:
        """Returns dict {'total': loss, 'continuity': ..., 'geostrophic': ..., 'track': ...}.
        Any disabled component is reported as 0.0."""
        out = {'continuity': torch.zeros(1, device=pred_5ch.device),
               'geostrophic': torch.zeros(1, device=pred_5ch.device),
               'track':       torch.zeros(1, device=pred_5ch.device)}
        if not self.enabled:
            out['total'] = sum(out.values())
            return out

        w = self._warmup_factor(epoch)

        # Continuity (multi-frame: average across K_out)
        if self.use_continuity:
            if pred_5ch.dim() == 5:   # (B, K, 5, H, W)
                cont = torch.stack(
                    [continuity_loss(pred_5ch[:, k], self.era5_mean, self.era5_std)
                     for k in range(pred_5ch.shape[1])]).mean()
            else:
                cont = continuity_loss(pred_5ch, self.era5_mean, self.era5_std)
            out['continuity'] = cont / self.scale_cont

        if self.use_geostrophic:
            if pred_5ch.dim() == 5:
                geo = torch.stack(
                    [geostrophic_loss(pred_5ch[:, k], cur_lat, self.era5_mean, self.era5_std)
                     for k in range(pred_5ch.shape[1])]).mean()
            else:
                geo = geostrophic_loss(pred_5ch, cur_lat, self.era5_mean, self.era5_std)
            out['geostrophic'] = geo / self.scale_geo

        if self.use_track and target_lat is not None and target_lon is not None:
            tr = track_consistency_loss(pred_5ch,
                                        cur_lat, cur_lon,
                                        target_lat, target_lon,
                                        self.era5_mean, self.era5_std)
            out['track'] = tr / self.scale_track

        out['total'] = (w * self.lambda_cont  * out['continuity']
                        + w * self.lambda_geo   * out['geostrophic']
                        + w * self.lambda_track * out['track'])
        return out
