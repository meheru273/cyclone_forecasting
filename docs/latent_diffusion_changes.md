# Changes Ported from `diffusion_unet_train.py` → `latent_diffusion.py`

> **Date:** 2026-03-15  
> **Purpose:** Replace the old `ConditionalUNet` with the proven, high-performing architecture from `diffusion_unet_train.py`, while preserving the two-stage VAE pipeline.

---

## 1. EMA (Exponential Moving Average)

**Ported class:** `EMA`

```python
class EMA:
    def __init__(self, model, decay=0.9999): ...
    def update(self, model): ...
    def apply(self, model): ...   # swap in EMA weights
    def restore(self, model): ...  # restore original weights
    def state_dict(self) / load_state_dict(self, sd): ...
```

**Integration in `DiffusionTrainer`:**
- `self.ema = EMA(self.unet, decay=config['ema_decay'])`
- `self.ema.update(self.unet)` called after every optimizer step
- `ema.apply / ema.restore` wraps all evaluation and sample generation
- EMA weights saved as `diffusion_best.pt`

---

## 3. Normalization — Hardcoded Stats

Removed `norm_stats_path` fallback entirely. Stats are now permanently hardcoded in `CycloneDataset.__init__` (precomputed from 5,000-sample dataset scan):

| Variable | Mean | Std |
|---|---|---|
| GRIDSAT | 169.8992 | 122.2612 |
| ERA5[0] u-wind | -1.1655 | 5.3889 |
| ERA5[1] v-wind | 0.3835 | 5.0042 |
| ERA5[2] temp | 298.4892 | 4.2326 |
| ERA5[3] pressure | 100976.29 | 521.15 |

`norm_stats_path` removed from: `CycloneDataset`, `get_dataloaders`, `train_vae`, `train_diffusion`, `main()`, and config.

---

## 4. `ConditionEncoder` — Replaced

**Removed:** `StormEncoding`, `storm_names` parameter from all calls.  
**Added:** Deeper 3-layer context projection, additive spatial fusion.

```python
class ConditionEncoder(nn.Module):
    # gridsat CNN → spatial features (output_channels)
    # ERA5 + timestamp → temporal_emb
    # 3-layer context_projection: temporal_emb → context_emb (embed_dim*2)
    # context_to_spatial: context_emb → additive bias on image_features
    def forward(self, gridsat, era5, timestamps):  # no storm_names
        return image_features, context_emb          # context_emb has dim embed_dim*2
```

---

## 5. UNet Building Blocks — Replaced

### `ResnetBlock` (new)
```python
class ResnetBlock(nn.Module):
    # GroupNorm → SiLU → Conv → + t_proj(t_emb) → GroupNorm → Dropout → Conv → + shortcut
    def __init__(self, in_channels, out_channels, t_emb_dim, dropout=0.1): ...
```

### `SelfAttention2D` (new)
```python
class SelfAttention2D(nn.Module):
    # GroupNorm + nn.MultiheadAttention (residual)
    def __init__(self, channels, num_heads=4): ...
```

### `DownBlock` — Replaced
- Uses `ResnetBlock × 2` + optional `SelfAttention2D`
- Downsamples with `nn.Conv2d(stride=2)` (not `MaxPool`)
- **Gradient checkpointing** enabled during training

### `MidBlock` — Replaced
- `ResnetBlock → SelfAttention2D → ResnetBlock`

### `UpBlock` — Replaced
- Uses `ResnetBlock × 2` + optional `SelfAttention2D`
- Upsamples with `nn.ConvTranspose2d` (learnable, not `F.interpolate`)

---

## 6. `ConditionalUNet` → `DiffusionUNet`

| | Old `ConditionalUNet` | New `DiffusionUNet` |
|---|---|---|
| Input | `noisy_latent, ..., storm_names, t` | `noisy_latent, ..., t` (no storm_names) |
| `channel_mults` | `(1, 2, 4, 8)` — 4 levels | `(1, 2, 4)` — 3 levels |
| Attention | Every encoder/decoder block | Middle encoder + MidBlock only |
| Conditioning | `concat([t_emb, ctx_emb])` → linear | Additive: `t_emb + ctx_emb` |
| Upsampling | `F.interpolate` | `ConvTranspose2d` |
| Dropout | None | `dropout=0.1` in ResnetBlock |
| context_emb dim | `embed_dim` | `embed_dim * 2` (deeper projection) |

**Key init:**
```python
self.t_proj       = Linear(t_emb_dim, emb_dim) → SiLU → Linear
self.context_proj = Linear(cond_emb_dim, emb_dim) × 3 layers
comb = t_emb + ctx_emb  # additive fusion
```

---

## 7. `DiffusionTrainer` Updates

- `ConditionalUNet` → `DiffusionUNet`
- `torch.cuda.amp.GradScaler()` → `torch.amp.GradScaler('cuda')`
- `torch.cuda.amp.autocast` → `torch.amp.autocast('cuda', ...)`
- `storm_names` removed from: `train_epoch`, `validate`, `sample`, `save_samples`
- `save_checkpoint`: best checkpoint saves **EMA weights**
- `_load_diffusion_checkpoint`: restores EMA state dict

---

## 8. `evaluate_model` & `save_comparison_images`

- `storm_names` removed from all condition encoder / UNet forward calls
- `evaluate_model`: **RNG state save/restore** (fixes training randomness contamination)
  ```python
  rng_state = torch.random.get_rng_state()
  torch.manual_seed(42)
  # ... evaluation ...
  torch.random.set_rng_state(rng_state)
  ```

---

## 9. Config Changes (`main()`)

```python
'channel_mults':   (1, 2, 4),     # was (1, 2, 4, 8)
't_emb_dim':       128,
'dropout':         0.1,            # new
'ema_decay':       0.9999,         # new
'kl_weight':       1e-2,           # was 1e-3
'kl_anneal_start': 0,              # new — KL warm-up
'kl_anneal_end':   10,             # new
'fid_every':       5,              # new — periodic FID
'eval_samples':    200,            # was 50
```

---

## 10. VAE Trainer Improvements (separate from diffusion_unet_train.py)

### KL Annealing
```python
def _get_kl_weight(self, epoch):
    # linear ramp: 0 → kl_weight_target over [kl_anneal_start, kl_anneal_end]
```
Applied in `train_epoch` and `compute_loss(kl_weight=...)`.

### Periodic FID
FID now computed every `fid_every` epochs (default 5) instead of only at the final epoch. `kl_weight` column added to metrics CSV.
