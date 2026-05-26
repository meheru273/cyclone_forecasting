# Trajectory Prediction Design Analysis

## Problem Statement

The current `CoordPredictor` MLP predicts next-timestep (lat, lon) from a compressed context embedding. While the context *does* contain encoded weather info, the spatial wind patterns — which are the primary driver of cyclone motion — are collapsed into a 256-dim vector before the MLP ever sees them.

### What `context_emb` actually contains

```python
# ConditionEncoder.forward():
temporal_emb = self.temporal_encoder(timestamps)  # time info
coord_emb = self.coord_proj(coords)               # current position
temporal_emb = temporal_emb + coord_emb            # sum
context_emb = self.context_projection(temporal_emb)  # → 256-dim
```

**Critical observation**: `context_emb` = f(time, current_coords). It does NOT contain the ERA5 spatial features. The ERA5 image features go into `image_features` (the spatial conditioning map), but the MLP only gets `context_emb`. So the MLP is literally predicting displacement from *time + position* alone — no wind information.

---

## Option Analysis

### Option 0: Current — MLP on context_emb (Baseline)

```
context_emb (256) → Linear(128) → Linear(64) → Linear(2) → (lat, lon)
```

| Pros | Cons |
|------|------|
| Simple, fast, no arch change needed | No access to spatial wind patterns |
| Jointly optimized with diffusion | Predicts absolute coords (harder target) |
| | Context is time+position only — no ERA5 |

**Verdict**: Weakest option. The MLP learns a statistical prior of "storms at position X in month Y tend to move to Z" — it cannot adapt to the actual wind field of a specific storm.

---

### Option A: Steering Flow Head (Physics-Guided)

```
ERA5 U/V (B,2,256,256) + current_coords → ROI crop → Conv → MLP → Δlat, Δlon
next_coords = current_coords + Δlat, Δlon
```

| Pros | Cons |
|------|------|
| Physically grounded — TC motion ≈ steering flow | Requires careful ROI implementation |
| Predicts **displacement** (Δ) — much easier target | Only uses wind channels, ignores satellite structure |
| Directly reads ERA5 U/V wind at full resolution | Separate from diffusion — not jointly generated |
| Interpretable — you can visualize the steering vector | Fixed crop size assumption |

**Physics basis**: In operational meteorology, tropical cyclone motion is approximated by the deep-layer mean steering flow (typically 850-200 hPa average). ERA5 channels [0,1] = (U-wind, V-wind) at ~850 hPa. Area-averaging these around the storm center gives the "steering vector" — the expected displacement direction.

**Implementation sketch**:
```python
class SteeringFlowHead(nn.Module):
    def __init__(self, roi_size=32, img_size=256):
        super().__init__()
        self.roi_size = roi_size
        self.img_size = img_size
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64), nn.SiLU(),
            nn.Linear(64, 2)  # Δlat, Δlon
        )

    def forward(self, era5, current_coords, coord_mean, coord_std):
        # Extract U/V wind channels (first 2 of 4 ERA5 channels)
        uv_wind = era5[:, :2]  # (B, 2, H, W)
        # Crop ROI around current storm center
        roi = self._extract_roi(uv_wind, current_coords, coord_mean, coord_std)
        features = self.conv(roi)
        delta = self.mlp(features)
        return current_coords + delta  # predict next position as displacement

    def _extract_roi(self, wind, coords, coord_mean, coord_std):
        # Un-normalize coords to pixel space
        lat_raw = coords[:, 0] * coord_std[0] + coord_mean[0]
        lon_raw = coords[:, 1] * coord_std[1] + coord_mean[1]
        # Map to [0, img_size] — assumes the image is centered on the storm
        # For storm-centered crops, center is approximately (H//2, W//2)
        cx, cy = self.img_size // 2, self.img_size // 2
        half = self.roi_size // 2
        # Simple center crop (since images are already storm-centered)
        return wind[:, :, cy-half:cy+half, cx-half:cx+half]
```

> [!IMPORTANT]
> Since dataset images are **already cropped/centered on the storm**, the center of the image ≈ storm center. The ROI crop is essentially a center-crop of the wind field, which simplifies implementation.

---

### Option B: Gaussian Heatmap as Diffusion Channel

```
(lat, lon) → Gaussian heatmap (1×H×W) → concat with 5ch → 6ch VAE → latent → diffusion jointly denoises all 6 channels
```

| Pros | Cons |
|------|------|
| Fully joint — position is generated alongside weather | We already tried similar (7ch VAE) and abandoned it |
| Cross-channel attention means weather informs position | Heatmap is "easy" for VAE — most pixels are zero |
| Native to the diffusion process | VAE capacity wasted on reconstructing Gaussian blob |
| No separate prediction head | Extracting (lat, lon) from heatmap requires post-processing |
| Position uncertainty is naturally modeled | Increases latent_dim, UNet params |

**Key difference from the abandoned v1**: v1 used **constant spatial maps** (every pixel = same value). A Gaussian heatmap is spatially structured — it has a peak, falloff, and is information-rich. The VAE can meaningfully encode/decode it.

**BUT**: The heatmap channel would dominate VAE reconstruction loss early in training (it's a simple peaked distribution with mostly-zero background), potentially starving the weather channels of gradient signal. You'd need loss weighting per channel.

**Implementation sketch**:
```python
def coords_to_heatmap(lat, lon, H=256, W=256, sigma=8.0, 
                       coord_mean, coord_std, lat_range, lon_range):
    """Convert normalized coords to spatial Gaussian heatmap."""
    # Un-normalize
    lat_raw = lat * coord_std[0] + coord_mean[0]
    lon_raw = lon * coord_std[1] + coord_mean[1]
    # Map to pixel coords (assuming image covers some lat/lon extent)
    cy = (lat_raw - lat_range[0]) / (lat_range[1] - lat_range[0]) * H
    cx = (lon_raw - lon_range[0]) / (lon_range[1] - lon_range[0]) * W
    # Generate Gaussian
    y = torch.arange(H, device=lat.device).float()
    x = torch.arange(W, device=lat.device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    heatmap = torch.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
    return heatmap  # (H, W), values in [0, 1]
```

> [!WARNING]
> **Major complication**: The dataset images are storm-centered crops, NOT geo-referenced grids. There's no fixed lat/lon ↔ pixel mapping. The storm center is *always* near the image center by construction. A heatmap would always peak at the center — carrying almost no information about absolute position. This makes Option B **problematic for this dataset**.

---

### Option C: Hybrid — Steering Flow + Context

Combine the physics-guided wind reading with the existing context embedding:

```
SteeringFlowHead(ERA5 wind, current_pos) → Δ_physics (2,)
MLP(context_emb)                         → Δ_learned (2,)
next_coords = current_coords + α * Δ_physics + β * Δ_learned
```

| Pros | Cons |
|------|------|
| Best of both — physics prior + learned residual | More hyperparameters (α, β or learned weighting) |
| Physics handles typical motion, MLP handles anomalies | Still separate from diffusion generation |
| Robust — even if MLP fails, physics provides reasonable motion | |

**Implementation**: Make α and β learnable:
```python
class HybridCoordPredictor(nn.Module):
    def __init__(self, embed_dim=256, roi_size=32):
        super().__init__()
        self.steering = SteeringFlowHead(roi_size)
        self.learned = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.SiLU(), nn.Linear(64, 2))
        self.gate = nn.Sequential(
            nn.Linear(embed_dim + 2, 1), nn.Sigmoid())  # learn blending weight

    def forward(self, era5, context_emb, current_coords, coord_mean, coord_std):
        delta_phys = self.steering(era5, current_coords, coord_mean, coord_std) - current_coords
        delta_learn = self.learned(context_emb)
        # Learned gate: how much to trust physics vs learned
        gate_input = torch.cat([context_emb, current_coords], dim=-1)
        alpha = self.gate(gate_input)
        delta = alpha * delta_phys + (1 - alpha) * delta_learn
        return current_coords + delta
```

---

## Recommendation

| Criterion | Option 0 (Current) | Option A (Steering) | Option B (Heatmap) | Option C (Hybrid) |
|-----------|:--:|:--:|:--:|:--:|
| Uses spatial wind data | ❌ | ✅ | ✅ (indirectly) | ✅ |
| Physics grounded | ❌ | ✅✅ | ❌ | ✅ |
| Easy to implement | ✅✅ | ✅ | ⚠️ | ✅ |
| Works with storm-centered crops | ✅ | ✅ | ❌ | ✅ |
| Predicts displacement (easier) | ❌ | ✅ | N/A | ✅ |
| Joint with diffusion | ✅ | ❌ | ✅ | ❌ |
| Interpretable | ❌ | ✅✅ | ⚠️ | ✅ |

### **Recommended: Option A (Steering Flow Head)**

**Why**:
1. **Physically correct** — TC motion IS the steering flow. This is what operational forecasters use.
2. **Predicts Δ (displacement)** — much easier regression target than absolute position.
3. **Uses ERA5 wind at full spatial resolution** — doesn't throw away the most relevant information.
4. **Storm-centered crops work perfectly** — center crop of wind field = steering flow around the storm.
5. **Simple to implement** — one small CNN + MLP, no architecture changes to VAE or UNet.

### **Why NOT Option B (Heatmap)**:
The dataset uses storm-centered crops. The storm center is always near pixel (128, 128). A Gaussian heatmap would always peak at roughly the same location, encoding almost no useful position information. You'd need geo-referenced imagery for this to work — which you don't have.

### **Option C (Hybrid) as stretch goal**:
If Option A alone gives >5° MAE, add the learned residual. The gate mechanism lets the model learn when physics is insufficient (e.g., during rapid intensification when β-drift matters).
