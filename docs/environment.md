# SDM Environment & Dependencies

## Runtime Environment

| Property | Value |
|----------|-------|
| Primary Platform | Kaggle Notebooks (GPU P100 / T4) |
| Timeout Limit | ~9 hours per session |
| Local Dev | Windows, `d:\Documents\satellite\` |
| Python | 3.10+ |
| Framework | PyTorch 2.x (CUDA) |

## Key Dependencies

```
torch >= 2.0
torchvision
numpy
pandas
scipy
matplotlib
tqdm
```

## Dataset Paths

| Resource | Kaggle Path |
|----------|-------------|
| Cyclone Dataset | `/kaggle/input/datasets/meheruzannat/cyclone-dataset` |
| Coordinate Lookup | `/kaggle/input/datasets/meheruzannat/coordinate-lookup/coordinate_lookup.json` |
| IBTrACS CSV (cache) | `./ibtracs_all.csv` (downloaded at runtime by `build_coordinate_lookup.py`) |

## Output Paths

| Output | Path |
|--------|------|
| VAE Checkpoints | `./checkpoints/mc_vae/` |
| Diffusion Checkpoints | `./checkpoints/mc_diffusion/` |
| Figures & Metrics | `./multichannel-diffusion/` |

## Checkpoint Resume Paths

After Kaggle timeout, save checkpoints as Kaggle Models, then set in config:

```python
'vae_checkpoint_path': '/kaggle/input/models/.../mc_vae_best.pt',
'diffusion_checkpoint_path': '/kaggle/input/models/.../mc_diff_best.pt',
```

## Hardware Notes

- **AMP disabled** (`use_amp: False`) — mixed precision caused NaN issues in earlier experiments
- **Gradient accumulation = 4** — effective batch size = 4 × 4 = 16
- `torch.cuda.empty_cache()` called every 50 batches to prevent OOM
