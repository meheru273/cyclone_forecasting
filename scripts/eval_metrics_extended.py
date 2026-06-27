"""
Canonical evaluation-metrics module.

Two layers of metrics:

  CORE (used by every trainer):
    - MetricsCalculator  : MAE, MSE, RMSE, PSNR, SSIM, SID
    - FID                : Frechet Inception Distance (InceptionV3 features)
    - InceptionScore     : IS for single-channel image generations

  EXTENDED (used by Phase-2 + Phase-3):
    - LPIPS              : perceptual similarity (VGG, pretrained)
    - CRPS               : Continuous Ranked Probability Score
    - threshold CSI/POD/FAR at IR brightness thresholds
    - RAPSD              : radially-averaged power spectral density + L1 distance
    - MSLP / wind        : tropical-cyclone intensity errors from ERA5
    - track km           : great-circle distance from steering-flow integration

Conventions:
  - inputs are torch tensors on the same device, layout (B, C, H, W)
  - channels: 0 = GRIDSAT, 1 = U-wind, 2 = V-wind, 3 = Temp, 4 = Pressure
  - per-channel mean/std arrays follow the same ordering as the dataset
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3


# ===========================================================================
# CORE METRICS
# ===========================================================================

class MetricsCalculator:
    """Pixel-level image-quality metrics. Stateless (all static methods)."""

    @staticmethod
    def mae(pred, target):
        return torch.mean(torch.abs(pred - target)).item()

    @staticmethod
    def mse(pred, target):
        return F.mse_loss(pred, target).item()

    @staticmethod
    def rmse(pred, target):
        return torch.sqrt(F.mse_loss(pred, target)).item()

    @staticmethod
    def psnr(pred, target, max_val=2.0):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return 100.0
        psnr = 20 * torch.log10(
            torch.tensor(max_val, device=mse.device) / torch.sqrt(mse))
        return psnr.item()

    @staticmethod
    def ssim(pred, target, window_size=11, size_average=True):
        # Scale constants to data range L  (SSIM standard: C1=(K1*L)^2, C2=(K2*L)^2)
        L = (target.max() - target.min()).item()
        L = max(L, 1e-8)
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        sigma = 1.5
        gauss = torch.tensor(
            [np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
             for x in range(window_size)],
            dtype=pred.dtype, device=pred.device)
        gauss = gauss / gauss.sum()

        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)

        channel = pred.size(1)
        window = window.expand(channel, 1, window_size, window_size).contiguous()

        mu1 = F.conv2d(pred,   window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = F.conv2d(pred   * pred,   window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12   = F.conv2d(pred   * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean().item()
        return ssim_map.mean(1).mean(1).mean(1)

    @staticmethod
    def sid(pred, target, eps=1e-10):
        """Spectral Information Divergence — treats channels as spectral bands,
        computes symmetric KL divergence per pixel."""
        p = torch.clamp(pred   - pred.min()   + eps, min=eps)
        q = torch.clamp(target - target.min() + eps, min=eps)
        p = p / p.sum(dim=1, keepdim=True)
        q = q / q.sum(dim=1, keepdim=True)
        kl_pq = (p * torch.log(p / q)).sum(dim=1).mean()
        kl_qp = (q * torch.log(q / p)).sum(dim=1).mean()
        return (kl_pq + kl_qp).item()


class FID(nn.Module):
    """Fréchet Inception Distance — InceptionV3 feature space."""

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()
        inception.eval()
        for param in inception.parameters():
            param.requires_grad = False
        self.inception = inception.to(device)

    def preprocess(self, images):
        images = F.interpolate(images, size=(299, 299),
                               mode='bilinear', align_corners=False)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = (images + 1) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        return images

    @torch.no_grad()
    def get_features(self, images):
        self.inception.eval()
        images = self.preprocess(images)
        features = self.inception(images)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features.cpu().numpy()

    def calculate_fid(self, real_features, fake_features):
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)
        diff = mu_real - mu_fake
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return float(fid)


class InceptionScore:
    """Inception Score — predicate distribution sharpness for IR generations."""

    def __init__(self, device='cuda'):
        self.device = device
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        for param in inception.parameters():
            param.requires_grad = False
        self.inception = inception.to(device)

    def preprocess(self, images):
        images = F.interpolate(images, size=(299, 299),
                               mode='bilinear', align_corners=False)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        return (images + 1) / 2.0

    @torch.no_grad()
    def calculate_inception_score(self, images, splits=10):
        self.inception.eval()
        images = self.preprocess(images)
        preds = self.inception(images)
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = F.softmax(preds, dim=1).cpu().numpy()

        actual_splits = min(splits, preds.shape[0])
        if actual_splits == 0:
            return 1.0, 0.0
        split_scores = []
        for k in range(actual_splits):
            part = preds[k * (preds.shape[0] // actual_splits):
                         (k + 1) * (preds.shape[0] // actual_splits), :]
            if part.shape[0] == 0:
                continue
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
            if scores:
                split_scores.append(np.exp(np.mean(scores)))
        if not split_scores:
            return 1.0, 0.0
        return float(np.mean(split_scores)), float(np.std(split_scores))


# ---------------------------------------------------------------------------
# LPIPS  (lazy-loaded — only constructs the VGG net on first call)
# ---------------------------------------------------------------------------

class _LPIPSWrapper:
    """Wraps the `lpips` package; falls back to MSE on import failure so the
    eval loop still runs (just without the LPIPS column)."""

    def __init__(self):
        self._net = None
        self._unavailable = False

    def _lazy_init(self, device):
        if self._net is not None or self._unavailable:
            return
        try:
            import lpips  # type: ignore
            self._net = lpips.LPIPS(net='vgg').to(device).eval()
            for p in self._net.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"  [LPIPS] unavailable ({e}); reporting as NaN")
            self._unavailable = True

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """pred, target: (B, 1, H, W) or (B, 3, H, W), normalised z-score.
        Returns mean LPIPS over the batch."""
        self._lazy_init(pred.device)
        if self._unavailable:
            return float('nan')
        # LPIPS expects 3-channel RGB-like input in [-1, 1].
        # Channel duplicate, then linear scale into [-1, 1] using the per-batch
        # min/max so the inputs land in the expected range.
        x = pred.detach()
        y = target.detach()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        # robust scaling: clip then rescale
        def to_unit(t):
            t_min = t.amin(dim=(1, 2, 3), keepdim=True)
            t_max = t.amax(dim=(1, 2, 3), keepdim=True)
            denom = (t_max - t_min).clamp(min=1e-6)
            return 2 * (t - t_min) / denom - 1
        x_u = to_unit(x)
        y_u = to_unit(y)
        return float(self._net(x_u, y_u).mean().item())


_LPIPS = _LPIPSWrapper()


def lpips_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    """LPIPS perceptual similarity. Lower is better. Returns scalar mean over batch."""
    return _LPIPS(pred, target)


# ---------------------------------------------------------------------------
# CRPS  (Continuous Ranked Probability Score)
# ---------------------------------------------------------------------------

@torch.no_grad()
def crps_ensemble(samples: torch.Tensor, target: torch.Tensor) -> float:
    """Compute CRPS from an ensemble of N stochastic samples.

    Args:
        samples: (N, B, C, H, W) — N samples per batch item
        target:  (B, C, H, W)

    Returns: scalar mean CRPS averaged over (B, C, H, W).

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|
    where X, X' are independent draws from the ensemble F.
    Estimated with the unbiased fair-CRPS form for finite N.
    """
    assert samples.dim() == 5, f"expected (N, B, C, H, W), got {samples.shape}"
    N = samples.shape[0]
    target_b = target.unsqueeze(0)                # (1, B, C, H, W)
    term1 = (samples - target_b).abs().mean(dim=0)        # (B, C, H, W)
    # E|X - X'|: pairwise absolute differences within the ensemble (fair CRPS)
    # Sort along the sample dim, then use the closed-form identity.
    s_sorted, _ = samples.sort(dim=0)
    # weights for fair CRPS: 2*(i - (N+1)/2) / (N*(N-1)) for 1-indexed i
    weights = (2.0 * (torch.arange(1, N + 1, device=samples.device) - (N + 1) / 2.0)
               / (N * (N - 1)))
    weights = weights.view(N, 1, 1, 1, 1).to(samples.dtype)
    term2 = (s_sorted * weights).sum(dim=0)               # (B, C, H, W)
    return float((term1 - term2).mean().item())


# ---------------------------------------------------------------------------
# Threshold-based skill scores  (CSI, POD, FAR)
# ---------------------------------------------------------------------------

@torch.no_grad()
def threshold_skill_scores(pred: torch.Tensor, target: torch.Tensor,
                           threshold: float,
                           comparison: str = 'less') -> dict:
    """Critical Success Index and friends, both pred and target as binary
    masks defined by a brightness-temperature threshold.

    Args:
        pred, target: (B, 1, H, W) — *raw physical units* (Kelvin for IR)
        threshold:    scalar in the same units
        comparison:   'less' (cold cloud-top, IR < threshold) or 'greater'

    Returns dict with keys 'csi', 'pod', 'far' (each a float in [0, 1]).
    """
    if comparison == 'less':
        p_mask = (pred   < threshold).float()
        t_mask = (target < threshold).float()
    else:
        p_mask = (pred   > threshold).float()
        t_mask = (target > threshold).float()

    hits  = (p_mask * t_mask).sum().item()
    misses = ((1 - p_mask) * t_mask).sum().item()
    false_alarms = (p_mask * (1 - t_mask)).sum().item()

    pod = hits / max(hits + misses,        1e-9)        # probability of detection
    far = false_alarms / max(hits + false_alarms, 1e-9) # false alarm ratio
    csi = hits / max(hits + misses + false_alarms, 1e-9)
    return {'csi': float(csi), 'pod': float(pod), 'far': float(far)}


def denormalise_gridsat(x_norm: torch.Tensor, gs_mean: float, gs_std: float) -> torch.Tensor:
    """Convert z-scored GRIDSAT brightness temperatures back to Kelvin."""
    return x_norm * gs_std + gs_mean


# ---------------------------------------------------------------------------
# RAPSD  (Radially-Averaged Power Spectral Density)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rapsd(image: torch.Tensor) -> torch.Tensor:
    """RAPSD of a 2-D image.

    Args:
        image: (H, W) torch tensor (real-valued, any units)

    Returns: 1-D tensor of length ~min(H, W)//2, the radially-averaged PSD.
    """
    assert image.dim() == 2
    H, W = image.shape
    # 2-D FFT
    F2 = torch.fft.fftshift(torch.fft.fft2(image))
    psd2 = (F2.real ** 2 + F2.imag ** 2)
    # build radial-bin indices
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H, device=image.device) - cy,
        torch.arange(W, device=image.device) - cx,
        indexing='ij',
    )
    r = torch.sqrt((xx.float() ** 2) + (yy.float() ** 2)).round().long()
    r_max = min(H, W) // 2
    # accumulate
    radial = torch.zeros(r_max + 1, device=image.device, dtype=psd2.dtype)
    counts = torch.zeros(r_max + 1, device=image.device, dtype=psd2.dtype)
    flat_r = r.flatten().clamp(max=r_max)
    flat_p = psd2.flatten()
    radial.scatter_add_(0, flat_r, flat_p)
    counts.scatter_add_(0, flat_r, torch.ones_like(flat_p))
    radial = radial / counts.clamp(min=1)
    return radial[1:]   # drop the DC bin


@torch.no_grad()
def rapsd_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """L1 distance between log-RAPSD of prediction and target, averaged over
    batch and channel.

    Args:
        pred, target: (B, C, H, W). Operates per-(B, C); averages.
    """
    B, C, H, W = pred.shape
    diffs = []
    for b in range(B):
        for c in range(C):
            p = rapsd(pred[b, c])
            t = rapsd(target[b, c])
            # log10 with a small epsilon to keep things finite
            lp = torch.log10(p.clamp(min=1e-12))
            lt = torch.log10(t.clamp(min=1e-12))
            diffs.append((lp - lt).abs().mean().item())
    return float(np.mean(diffs)) if diffs else 0.0


# ---------------------------------------------------------------------------
# Tropical-cyclone intensity errors  (MSLP / max wind)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mslp_err_hpa(pred_5ch: torch.Tensor, target_5ch: torch.Tensor,
                 era5_mean: Sequence[float], era5_std: Sequence[float]) -> float:
    """Mean error of minimum sea-level pressure within the central 64x64 ROI,
    in hPa.

    Args:
        pred_5ch, target_5ch: (B, 5, H, W) z-scored — channel 4 is pressure (Pa)
        era5_mean / era5_std: per-channel ERA5 normalisation stats (4 entries)
    """
    p_mean = float(era5_mean[3]); p_std = float(era5_std[3])
    h, w = pred_5ch.shape[2], pred_5ch.shape[3]
    roi = 64
    y0, x0 = (h - roi) // 2, (w - roi) // 2
    P_pred = pred_5ch[:, 4, y0:y0+roi, x0:x0+roi] * p_std + p_mean  # Pa
    P_tgt  = target_5ch[:, 4, y0:y0+roi, x0:x0+roi] * p_std + p_mean
    # Min over spatial dims, convert Pa → hPa (divide by 100), then mean abs err
    min_pred = P_pred.amin(dim=(1, 2)) / 100.0   # (B,)
    min_tgt  = P_tgt.amin(dim=(1, 2)) / 100.0
    return float((min_pred - min_tgt).abs().mean().item())


@torch.no_grad()
def max_wind_err(pred_5ch: torch.Tensor, target_5ch: torch.Tensor,
                 era5_mean: Sequence[float], era5_std: Sequence[float]) -> dict:
    """Mean absolute error of max wind component (U and V separately), in m/s.

    Returns dict with keys 'max_u_err', 'max_v_err', 'max_wind_err'.
    """
    u_mean, u_std = float(era5_mean[0]), float(era5_std[0])
    v_mean, v_std = float(era5_mean[1]), float(era5_std[1])
    h, w = pred_5ch.shape[2], pred_5ch.shape[3]
    roi = 64
    y0, x0 = (h - roi) // 2, (w - roi) // 2
    U_pred = pred_5ch[:, 1, y0:y0+roi, x0:x0+roi] * u_std + u_mean
    V_pred = pred_5ch[:, 2, y0:y0+roi, x0:x0+roi] * v_std + v_mean
    U_tgt  = target_5ch[:, 1, y0:y0+roi, x0:x0+roi] * u_std + u_mean
    V_tgt  = target_5ch[:, 2, y0:y0+roi, x0:x0+roi] * v_std + v_mean

    speed_pred = torch.sqrt(U_pred ** 2 + V_pred ** 2)
    speed_tgt  = torch.sqrt(U_tgt  ** 2 + V_tgt  ** 2)

    return {
        'max_u_err':    float((U_pred.abs().amax(dim=(1, 2))
                               - U_tgt.abs().amax(dim=(1, 2))).abs().mean().item()),
        'max_v_err':    float((V_pred.abs().amax(dim=(1, 2))
                               - V_tgt.abs().amax(dim=(1, 2))).abs().mean().item()),
        'max_wind_err': float((speed_pred.amax(dim=(1, 2))
                               - speed_tgt.amax(dim=(1, 2))).abs().mean().item()),
    }


# ---------------------------------------------------------------------------
# Track great-circle error from steering-flow displacement
# ---------------------------------------------------------------------------

def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in km between two (lat, lon) points in degrees."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def track_error_km_from_displacement(gt_d_lat: float, gt_d_lon: float,
                                     pred_d_lat: float, pred_d_lon: float,
                                     base_lat: float, base_lon: float) -> float:
    """Convert (dlat, dlon) displacements to great-circle km between predicted
    and ground-truth landing points, given a base (current) position."""
    gt_endpoint   = (base_lat + gt_d_lat,   base_lon + gt_d_lon)
    pred_endpoint = (base_lat + pred_d_lat, base_lon + pred_d_lon)
    return great_circle_km(*gt_endpoint, *pred_endpoint)
