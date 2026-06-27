"""
assembly.py — single entry point for the full thesis pipeline.

This is the **canonical config + orchestrator**. Other scripts (`vae.py`,
`multi_channel_diffusion.py`, `multi_frame_flow.py`, `latent_cache.py`,
`analysis/*`) are library modules; their `main()` functions, if present,
defer to this module's CONFIG.

Pipeline stages (each independently toggleable in STAGES below; each detects
already-completed work and skips):

    1. train_vae               → checkpoints/mc_vae/mc_vae_best.pt
    2. precompute_cache        → cache/latents_mc_vae/<vae_fingerprint>/
    3. train_single_frame_rf   → checkpoints/rectified_flow/...  (OPTIONAL,
                                  default OFF — multi-frame is the headline)
    4. train_multi_frame_rf    → checkpoints/mf_*/...            (HEADLINE)
    5. kfold                   → outputs/kfold/...               (OPTIONAL,
                                  default OFF per user request — k-fold is not
                                  a usual stability metric for generative
                                  models and is expensive)
    6. aggregate               → paper_results/main_results_table.csv

All output paths are root-relative (one level above `scripts/`):

    <repo_root>/checkpoints/...
    <repo_root>/cache/...
    <repo_root>/outputs/...
    <repo_root>/paper_results/...

Usage:
    python scripts/assembly.py                      # run all enabled stages
    python scripts/assembly.py --stage train_vae    # run a single stage
    python scripts/assembly.py --use-kfold          # enable k-fold for this run
    python scripts/assembly.py --skip train_vae train_single_frame_rf
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch

# ============================================================================
# Performance: autotune cuDNN convolution algorithms for fixed input shapes.
# Safe because every trainer in this repo has fixed spatial dimensions per
# stage. Gains: ~5–10% on the conv-heavy UNets.
# ============================================================================

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # TF32 is already on by default in recent PyTorch on Ampere+; make it explicit.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# Paths — all relative to the repo root (one above this file's directory)
# ============================================================================

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPTS_DIR.parent

CHECKPOINT_DIR = REPO_ROOT / 'checkpoints'
OUTPUT_DIR     = REPO_ROOT / 'outputs'
CACHE_DIR      = REPO_ROOT / 'cache'
PAPER_RESULTS  = REPO_ROOT / 'paper_results'

for d in (CHECKPOINT_DIR, OUTPUT_DIR, CACHE_DIR, PAPER_RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Dataset location — drive/PC-independent.
#
# By default the SETCD dataset is expected at  <repo_root>/cyclone dataset  so
# the whole tree moves between machines just by copying the project folder
# (E:\thesis_2007039 → D:\thesis_2007039 needs no edits). Override either:
#   * env var  THESIS_DATA_ROOT  → folder that *contains* "cyclone dataset", or
#   * env var  THESIS_DATASET_DIR → the dataset folder itself, or
#   * CLI flag --root (highest priority; handled in main()).
# ----------------------------------------------------------------------------

if os.environ.get('THESIS_DATASET_DIR'):
    DATASET_DIR = Path(os.environ['THESIS_DATASET_DIR'])
else:
    DATA_ROOT   = Path(os.environ.get('THESIS_DATA_ROOT', REPO_ROOT))
    DATASET_DIR = DATA_ROOT / 'cyclone dataset'


# ============================================================================
# Stage toggles — switch a stage off here OR pass --skip on the CLI
# ============================================================================

STAGES = {
    'train_vae':              True,
    'precompute_cache':       True,
    'train_single_frame_rf':  False,   # the Phase-1 reference; headline is multi-frame
    'train_multi_frame_rf':   True,
    'train_cascaded_diffusion': True,  # Phase-3 R4 baseline (Nath et al. cascaded DDPM)
    'kfold':                  False,   # k-fold is expensive and partially redundant for generative models
    'aggregate':              True,
}


# ============================================================================
# THE single CONFIG dict — shared by every stage. Override here, not in the
# per-script main() functions.
# ============================================================================

CONFIG = {
    # ---- Global ----
    'device':       'cuda' if torch.cuda.is_available() else 'cpu',
    'use_amp':      True,
    'amp_dtype':    'bfloat16',
    # Default grad-accum (single-frame trainers use this).
    # Multi-frame overrides via 'p2_gradient_accumulation_steps' below.
    'gradient_accumulation_steps': 4,    # single-frame: batch=4 × accum=4 → effective=16
    'weight_decay': 1e-5,

    # ---- Data (year-based split, validated by count_storms_per_year.py) ----
    'root_dir':           str(DATASET_DIR),
    'coord_lookup_path':  str(DATASET_DIR / 'coordinate_lookup.json'),
    'train_years':        ['2012','2013','2014','2015','2016','2017','2018','2019','2020'],
    'val_years':          ['2021'],
    'test_years':         ['2022'],
    'img_size':           256,
    'num_workers':        8,    # i7-14700K has 28 logical cores; 8 keeps the GPU fed without thrashing

    # ---- VAE (Stage 1) ----
    'latent_dim':         4,
    'vae_base_channels':  64,
    'vae_epochs':         50,
    'vae_lr':             1e-4,
    'kl_weight':          1e-4,
    'kl_anneal_start':    0,
    'kl_anneal_end':      20,
    'fid_every':          5,

    # ---- Backbone scaffolding (shared across all generative models) ----
    'base_channels':      128,
    'channel_mults':      (1, 2, 3, 4),
    'num_heads':          8,
    'embed_dim':          128,
    't_emb_dim':          128,
    'dropout':            0.1,
    'ema_decay':          0.9999,

    # ---- Phase 1: single-frame Rectified Flow ----
    'p1_epochs':              100,
    'p1_reflow_iterations':   1,           # 0 = 1-RF (no reflow); 1 = 2-RF (one reflow pass)
    'p1_reflow_epochs':       40,
    'p1_reflow_sample_steps': 20,
    'p1_learning_rate':       1e-4,
    'p1_min_lr':              5e-5,
    'p1_warmup_epochs':       5,
    'p1_batch_size':          4,
    'p1_flow_sampler':        'heun',
    'p1_flow_sample_steps':   50,
    'p1_eval_every':          5,
    'p1_eval_samples':        200,
    'p1_sampler_step_sweep':  [1, 4, 16, 50],

    # ---- Phase 2: multi-frame Rectified Flow (HEADLINE) ----
    'p2_variant':             'v2',      # 'v1' = channel-stacked 2-D UNet;
                                          # 'v2' = factorised spatio-temporal
                                          # (per-frame UNet + temporal attention,
                                          # Ablation B). V2 is heavier — use a
                                          # smaller p2_batch_size with grad-accum.
    'p2_k_in':                3,
    'p2_k_out':               3,
    'p2_epochs':              100,       # was 100 — first run plateaued at ep~50, needs more room
    'p2_reflow_iterations':   0,         # 1-RF (no reflow) — compare V2 vs V1 at same flow depth
    'p2_reflow_epochs':       40,
    'p2_reflow_sample_steps': 20,
    'p2_learning_rate':       1e-4,      # was 1e-4 — more aggressive; cosine has 250 ep to decay
    'p2_min_lr':              5e-5,
    'p2_warmup_epochs':       8,        # was 5 — longer ramp at higher peak LR
    # Resume controls:
    'p2_auto_resume':           True,     # K3-K3 1RF no-phys started but didn't finish
                                          # (best.pt exists, per_leadtime.csv missing) — resume
    'p2_fresh_lr_schedule':     True,    # when resuming, ignore the saved scheduler state
                                          # and start a fresh cosine from current LR over remaining epochs
    'p2_batch_size':                  4,   # V2 temporal attention needs more VRAM
    'p2_gradient_accumulation_steps': 4,   # → effective 16, identical training dynamics
    'p2_eval_every':          5,
    'p2_eval_samples':        200,
    'p2_eval_sampler_steps':  4,
    'p2_sampler_step_sweep':  [1, 4, 16, 50],
    'p2_crps_n_samples':      10,

    # ---- Physics losses (Phase 2 Ablation G — OFF: clean K3-K3 1RF no-phys control) ----
    'use_phys_loss':          False,
    'use_continuity':         False,
    'use_geostrophic':        False,
    'use_track_consistency':  False,
    'lambda_continuity':      1e-3,
    'lambda_geostrophic':     1e-4,
    'lambda_track':           1e-2,
    'phys_warmup_epochs':     10,

    # ---- Phase 3: Nath et al. cascaded-diffusion baseline (R4) ----
    # Pixel-space 3-stage DDPM cascade (64²→128²→256²), classifier-free guidance.
    # Multi-frame + 5-channel to match the headline flow rows for a fair Table-1
    # comparison. Trained by scripts/cascaded_diffusion.py.
    'cascade_run_tag':        'cascaded_diffusion_R4',
    'cascade_num_timesteps':  250,        # Nath: 250
    'cascade_prediction_type': 'v',       # 'v' (Salimans & Ho) — eps-pred let the
                                          # base stage ignore conditioning → noise
    'cascade_cond_scale':     3.0,        # Nath: classifier-free guidance scale
    'cascade_epochs':         [60, 50, 40],   # per stage: base64, sr128, sr256
    'cascade_batch_sizes':    [32, 16, 6],    # per stage (higher res → smaller)
    'cascade_eval_samples':   200,
    'cascade_eval_sampler_steps': 50,

    # ---- Phase 3: k-fold (OFF by default per user) ----
    'use_kfold':              False,
    'kfold_k':                5,
    'kfold_epochs':           80,
    'kfold_seed':             0,

    # ---- Output paths (root-relative, auto-populated below) ----
    'checkpoint_dir':         str(CHECKPOINT_DIR),
    'output_dir':             str(OUTPUT_DIR),
    'cache_dir':              str(CACHE_DIR),
    'paper_results_dir':      str(PAPER_RESULTS),
    'mc_vae_checkpoint':      str(CHECKPOINT_DIR / 'mc_vae' / 'mc_vae_best.pt'),
    'latent_cache_dir':       None,        # set after precompute step

    # ---- Phase 2 run tag (changes per ablation) ----
    'run_tag':                'mf_kin3_kout3_1rf_v2_no_phys',
}


# ============================================================================
# Stage 1: train MC-VAE
# ============================================================================

def build_vae_dataloaders(cfg: dict):
    """Returns single-frame (t, t+1) dataloaders for VAE training.
    Fails fast if any split is empty — bogus 'best' checkpoints have bitten us."""
    from multi_channel_diffusion import get_dataloaders
    loaders = get_dataloaders(
        root_dir=cfg['root_dir'],
        train_years=cfg['train_years'],
        val_years=cfg['val_years'],
        test_years=cfg['test_years'],
        batch_size=cfg.get('p1_batch_size', 4),
        num_workers=cfg['num_workers'],
        img_size=cfg['img_size'],
        coord_lookup_path=cfg.get('coord_lookup_path'),
    )
    train_loader, val_loader, test_loader = loaders
    for name, loader, years in [('train', train_loader, cfg['train_years']),
                                 ('val',   val_loader,   cfg['val_years']),
                                 ('test',  test_loader,  cfg['test_years'])]:
        if len(loader.dataset) == 0:
            raise RuntimeError(
                f"\n  {name}-split is EMPTY ({years} contains no (t, t+1) pairs under "
                f"\n    root_dir={cfg['root_dir']!r}."
                f"\n  Run scripts/count_storms_per_year.py against your local dataset to see "
                f"\n  which years are actually present, then either:"
                f"\n    (a) copy the missing years into your local dataset, or"
                f"\n    (b) edit CONFIG[{name}_years] in scripts/assembly.py to a year you have.")
    return loaders


def stage_train_vae(cfg: dict) -> bool:
    """Train MC-VAE. Skipped if checkpoint already exists."""
    vae_ckpt = Path(cfg['mc_vae_checkpoint'])
    if vae_ckpt.exists():
        print(f"[stage:train_vae] SKIP — checkpoint already at {vae_ckpt}")
        return True

    print(f"\n{'=' * 70}\n[stage:train_vae] training MC-VAE\n{'=' * 70}")
    from vae import MultiChannelVAETrainer

    train_loader, val_loader, _ = build_vae_dataloaders(cfg)
    trainer = MultiChannelVAETrainer(cfg)
    trainer.train(train_loader, val_loader)

    if not vae_ckpt.exists():
        raise RuntimeError(f"VAE training reported done but {vae_ckpt} is missing.")
    print(f"[stage:train_vae] OK — checkpoint at {vae_ckpt}")
    return True


# ============================================================================
# Stage 2: precompute latent cache
# ============================================================================

def stage_precompute_cache(cfg: dict) -> bool:
    """Encode every (storm, timestamp) once with the frozen VAE."""
    from latent_cache import vae_fingerprint

    vae_ckpt = Path(cfg['mc_vae_checkpoint'])
    if not vae_ckpt.exists():
        raise FileNotFoundError(
            f"[stage:precompute_cache] need VAE at {vae_ckpt} — run stage:train_vae first.")

    fp = vae_fingerprint(str(vae_ckpt))
    cache_root = Path(cfg['cache_dir']) / 'latents_mc_vae' / fp
    cfg['latent_cache_dir'] = str(cache_root)

    # Cheap completion check: if the cache root has at least 80 % of the
    # expected (storm, timestamp) pairs already, treat as done.
    if cache_root.exists():
        existing_files = sum(1 for _ in cache_root.rglob('*.npy'))
        if existing_files > 0:
            print(f"[stage:precompute_cache] SKIP — {existing_files:,} cached "
                  f"latents already present at {cache_root}")
            return True

    print(f"\n{'=' * 70}\n[stage:precompute_cache] populating {cache_root}\n{'=' * 70}")
    all_years = list(cfg['train_years']) + list(cfg['val_years']) + list(cfg['test_years'])
    cmd = [
        sys.executable, str(SCRIPTS_DIR / 'latent_cache.py'),
        '--vae-checkpoint', str(vae_ckpt),
        '--cache-dir',      str(Path(cfg['cache_dir']) / 'latents_mc_vae'),
        '--root',           cfg['root_dir'],
        '--years',          *all_years,
        '--img-size',       str(cfg['img_size']),
        '--latent-dim',     str(cfg['latent_dim']),
        '--vae-base-channels', str(cfg['vae_base_channels']),
    ]
    print('  $', ' '.join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"latent_cache.py exited with code {result.returncode}")
    print(f"[stage:precompute_cache] OK — cache at {cache_root}")
    return True


# ============================================================================
# Stage 3: single-frame Rectified Flow (Phase 1 reference)
# ============================================================================

def _p1_config(cfg: dict) -> dict:
    """Map the master CONFIG into the single-frame RF trainer's expected keys."""
    sub = dict(cfg)
    sub.update({
        # The trainer's existing key names
        'diffusion_epochs':       cfg['p1_epochs'],
        'reflow_iterations':      cfg['p1_reflow_iterations'],
        'reflow_epochs':          cfg['p1_reflow_epochs'],
        'reflow_sample_steps':    cfg['p1_reflow_sample_steps'],
        'learning_rate':          cfg['p1_learning_rate'],
        'min_lr':                 cfg['p1_min_lr'],
        'warmup_epochs':          cfg['p1_warmup_epochs'],
        'batch_size':             cfg['p1_batch_size'],
        'eval_every':             cfg['p1_eval_every'],
        'eval_samples':           cfg['p1_eval_samples'],
        'sampler_step_sweep':     cfg['p1_sampler_step_sweep'],
    })
    return sub


def stage_train_single_frame_rf(cfg: dict) -> bool:
    """Phase-1 P1-RF2 (initial flow + one reflow → 2-RF)."""
    out = Path(cfg['checkpoint_dir']) / 'rectified_flow' / 'rf_reflow_1_best.pt'
    if out.exists():
        print(f"[stage:train_single_frame_rf] SKIP — {out} already exists")
        return True

    print(f"\n{'=' * 70}\n[stage:train_single_frame_rf] single-frame 2-RF\n{'=' * 70}")
    from multi_channel_diffusion import get_dataloaders
    from rectified_flow_score_matching import RectifiedFlowTrainer

    sub = _p1_config(cfg)
    train_loader, val_loader, _ = get_dataloaders(
        root_dir=sub['root_dir'],
        train_years=sub['train_years'],
        val_years=sub['val_years'],
        test_years=sub['test_years'],
        batch_size=sub['batch_size'],
        num_workers=sub['num_workers'],
        img_size=sub['img_size'],
        coord_lookup_path=sub.get('coord_lookup_path'),
    )
    trainer = RectifiedFlowTrainer(sub, cfg['mc_vae_checkpoint'])
    trainer.train(train_loader, val_loader)
    print(f"[stage:train_single_frame_rf] OK")
    return True


# ============================================================================
# Stage 4: multi-frame Rectified Flow (HEADLINE)
# ============================================================================

def _p2_config(cfg: dict) -> dict:
    """Map the master CONFIG into the multi-frame RF trainer's expected keys.

    Note: multi-frame overrides 'gradient_accumulation_steps' so single-frame
    trainers keep their grad-accum=4 default while MF runs at accum=1.
    """
    sub = dict(cfg)
    sub.update({
        'k_in':                   cfg['p2_k_in'],
        'k_out':                  cfg['p2_k_out'],
        'diffusion_epochs':       cfg['p2_epochs'],
        'reflow_iterations':      cfg['p2_reflow_iterations'],
        'reflow_epochs':          cfg['p2_reflow_epochs'],
        'reflow_sample_steps':    cfg['p2_reflow_sample_steps'],
        'learning_rate':          cfg['p2_learning_rate'],
        'min_lr':                 cfg['p2_min_lr'],
        'warmup_epochs':          cfg['p2_warmup_epochs'],
        'batch_size':             cfg['p2_batch_size'],
        'gradient_accumulation_steps': cfg.get(
            'p2_gradient_accumulation_steps',
            cfg.get('gradient_accumulation_steps', 1)),
        'eval_every':             cfg['p2_eval_every'],
        'eval_samples':           cfg['p2_eval_samples'],
        'eval_sampler_steps':     cfg['p2_eval_sampler_steps'],
        'sampler_step_sweep':     cfg['p2_sampler_step_sweep'],
        'crps_n_samples':         cfg['p2_crps_n_samples'],
        'fresh_lr_schedule':      cfg.get('p2_fresh_lr_schedule', False),
        'variant':                cfg.get('p2_variant', 'v1'),
    })
    return sub


def stage_train_multi_frame_rf(cfg: dict) -> bool:
    """Phase-2 headline: K_in=3, K_out=3, 2-RF.

    Skip-logic:
      - With reflow_iterations > 0: SKIP if mf_reflow_<N>_best.pt exists.
      - With reflow_iterations = 0: SKIP if mf_initial_best.pt exists AND
        the saved checkpoint already reached p2_epochs.

    Auto-resume:
      - If p2_auto_resume is True and mf_initial_latest.pt exists, the trainer
        is constructed with resume_path pointing at it.
    """
    ckpt_dir = Path(cfg['checkpoint_dir']) / cfg['run_tag']
    if cfg['p2_reflow_iterations'] > 0:
        final_ckpt = ckpt_dir / f'mf_reflow_{cfg["p2_reflow_iterations"]}_best.pt'
    else:
        final_ckpt = ckpt_dir / 'mf_initial_best.pt'

    # Smarter skip: only skip if the saved checkpoint reached the configured epoch.
    resume_path: Optional[Path] = None
    if cfg.get('p2_auto_resume', False):
        cand = ckpt_dir / 'mf_initial_latest.pt'
        if cand.exists():
            try:
                import torch
                meta = torch.load(cand, map_location='cpu', weights_only=False)
                saved_epoch = int(meta.get('epoch', 0))
                if saved_epoch >= int(cfg['p2_epochs']):
                    print(f"[stage:train_multi_frame_rf] SKIP — checkpoint at epoch "
                          f"{saved_epoch} already reached p2_epochs={cfg['p2_epochs']}")
                    return True
                resume_path = cand
                print(f"[stage:train_multi_frame_rf] resuming from epoch {saved_epoch} "
                      f"→ target epoch {cfg['p2_epochs']}")
            except Exception as e:
                print(f"  [resume] couldn't read {cand}: {e}; starting fresh")
                resume_path = None

    if resume_path is None and final_ckpt.exists():
        print(f"[stage:train_multi_frame_rf] SKIP — {final_ckpt} exists (auto-resume off)")
        return True

    print(f"\n{'=' * 70}\n[stage:train_multi_frame_rf] multi-frame 2-RF "
          f"({cfg['p2_k_in']}→{cfg['p2_k_out']}, run_tag={cfg['run_tag']})\n{'=' * 70}")
    from multi_frame_dataset import get_multi_frame_dataloaders
    from multi_frame_flow import MultiFrameRFTrainer

    sub = _p2_config(cfg)
    train_loader, val_loader, _ = get_multi_frame_dataloaders(
        root_dir=sub['root_dir'],
        train_years=sub['train_years'],
        val_years=sub['val_years'],
        test_years=sub['test_years'],
        k_in=sub['k_in'], k_out=sub['k_out'],
        batch_size=sub['batch_size'],
        num_workers=sub['num_workers'],
        img_size=sub['img_size'],
        coord_lookup_path=sub.get('coord_lookup_path'),
        latent_cache_dir=cfg.get('latent_cache_dir'),
    )
    trainer = MultiFrameRFTrainer(
        sub, cfg['mc_vae_checkpoint'],
        resume_path=str(resume_path) if resume_path else None,
    )
    trainer.train(train_loader, val_loader)
    print(f"[stage:train_multi_frame_rf] OK")
    return True


# ============================================================================
# Stage 4b: Nath et al. cascaded-diffusion baseline (Phase-3 R4)
# ============================================================================

def stage_train_cascaded_diffusion(cfg: dict) -> bool:
    """Phase-3 external baseline: pixel-space cascaded DDPM (64²→128²→256²).

    Skips if the final per-leadtime CSV already exists. Per-stage checkpoints
    are skipped individually inside the trainer when present, so an interrupted
    run resumes at the first un-trained stage.
    """
    run_tag = cfg['cascade_run_tag']
    out_csv = Path(cfg['output_dir']) / run_tag / 'metrics' / 'per_leadtime.csv'
    if out_csv.exists():
        print(f"[stage:train_cascaded_diffusion] SKIP — {out_csv} exists")
        return True

    print(f"\n{'=' * 70}\n[stage:train_cascaded_diffusion] Nath-style cascaded DDPM "
          f"({cfg['p2_k_in']}→{cfg['p2_k_out']}, run_tag={run_tag})\n{'=' * 70}")
    from cascaded_diffusion import CascadedDiffusionTrainer

    sub = {
        'device':          cfg['device'],
        'use_amp':         cfg.get('use_amp', True),
        'amp_dtype':       cfg.get('amp_dtype', 'bfloat16'),
        'weight_decay':    cfg.get('weight_decay', 1e-5),
        'ema_decay':       cfg.get('ema_decay', 0.9999),
        't_emb_dim':       cfg.get('t_emb_dim', 128),
        'num_heads':       cfg.get('num_heads', 8),
        'k_in':            cfg['p2_k_in'],
        'k_out':           cfg['p2_k_out'],
        'run_tag':         run_tag,
        'checkpoint_dir':  cfg['checkpoint_dir'],
        'output_dir':      cfg['output_dir'],
        'num_timesteps':   cfg['cascade_num_timesteps'],
        'prediction_type': cfg.get('cascade_prediction_type', 'v'),
        'cond_scale':      cfg['cascade_cond_scale'],
        'cascade_epochs':  cfg['cascade_epochs'],
        'cascade_batch_sizes': cfg['cascade_batch_sizes'],
        'learning_rate':   cfg['p2_learning_rate'],
        'min_lr':          cfg['p2_min_lr'],
        'warmup_epochs':   cfg['p2_warmup_epochs'],
        'eval_samples':    cfg['cascade_eval_samples'],
        'eval_sampler_steps': cfg['cascade_eval_sampler_steps'],
        'crps_n_samples':  cfg['p2_crps_n_samples'],
    }
    trainer = CascadedDiffusionTrainer(sub)
    trainer.train(
        root_dir=cfg['root_dir'],
        train_years=cfg['train_years'],
        val_years=cfg['val_years'],
        test_years=cfg['test_years'],
        num_workers=cfg['num_workers'],
        coord_lookup_path=cfg.get('coord_lookup_path'),
    )
    print(f"[stage:train_cascaded_diffusion] OK")
    return True


# ============================================================================
# Stage 5: k-fold (OPTIONAL)
# ============================================================================

def stage_kfold(cfg: dict) -> bool:
    if not cfg.get('use_kfold', False):
        print("[stage:kfold] SKIP — use_kfold=False (k-fold is not a usual "
              "stability check for generative models). Pass --use-kfold to enable.")
        return True

    kfold_dir = Path(cfg['output_dir']) / 'kfold'
    summary   = kfold_dir / 'kfold_summary.csv'
    if summary.exists():
        print(f"[stage:kfold] SKIP — {summary} already exists")
        return True

    print(f"\n{'=' * 70}\n[stage:kfold] basin-stratified storm-level "
          f"{cfg['kfold_k']}-fold\n{'=' * 70}")
    cmd = [
        sys.executable, str(SCRIPTS_DIR / 'analysis' / 'kfold_runner.py'),
        '--root',           cfg['root_dir'],
        '--mc-vae',         cfg['mc_vae_checkpoint'],
        '--train-years',    *cfg['train_years'],
        '--k',              str(cfg['kfold_k']),
        '--diffusion-epochs', str(cfg['kfold_epochs']),
        '--reflow-iterations', str(cfg['p2_reflow_iterations']),
        '--reflow-epochs',    str(cfg['p2_reflow_epochs']),
        '--batch-size',     str(cfg['p2_batch_size']),
        '--num-workers',    str(cfg['num_workers']),
        '--img-size',       str(cfg['img_size']),
        '--k-in',           str(cfg['p2_k_in']),
        '--k-out',          str(cfg['p2_k_out']),
        '--base-out',       str(kfold_dir),
        '--seed',           str(cfg['kfold_seed']),
    ]
    if cfg.get('coord_lookup_path'):
        cmd.extend(['--coord-lookup-path', cfg['coord_lookup_path']])
    if cfg.get('latent_cache_dir'):
        cmd.extend(['--latent-cache-dir', cfg['latent_cache_dir']])
    print('  $', ' '.join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"kfold_runner.py exited with code {result.returncode}")
    print(f"[stage:kfold] OK")
    return True


# ============================================================================
# Stage 6: aggregate → main_results_table.csv
# ============================================================================

def stage_aggregate(cfg: dict) -> bool:
    """Builds the paper's Table-1 from per-run CSVs.

    The registry maps run IDs (paper row labels) to their per-leadtime CSVs.
    Only includes rows whose CSV exists on disk; missing rows are silently
    skipped (e.g. if Nath et al. baseline hasn't been added yet).
    """
    out_csv = Path(cfg['paper_results_dir']) / 'main_results_table.csv'
    if out_csv.exists():
        print(f"[stage:aggregate] WARN — {out_csv} already exists; overwriting.")

    registry = [
        {
            'run_id':            'R6_mf_2rf_no_phys',
            'description':       'Ours — multi-frame 2-RF (no physics)',
            'per_leadtime_csv':  str(Path(cfg['output_dir']) / cfg['run_tag']
                                      / 'metrics' / 'per_leadtime.csv'),
            'sampler_steps_csv': str(Path(cfg['output_dir']) / cfg['run_tag']
                                      / 'metrics' / 'sampler_steps.csv'),
        },
        {
            'run_id':            'R4_nath_cascaded_diffusion',
            'description':       'Nath et al. — cascaded DDPM (our setting)',
            'per_leadtime_csv':  str(Path(cfg['output_dir']) / cfg['cascade_run_tag']
                                      / 'metrics' / 'per_leadtime.csv'),
            'sampler_steps_csv': str(Path(cfg['output_dir']) / cfg['cascade_run_tag']
                                      / 'metrics' / 'sampler_steps.csv'),
        },
    ]
    # Honour the contract above: only keep rows whose per-leadtime CSV is on
    # disk. If none exist yet (e.g. nothing has finished a final eval), skip the
    # stage cleanly instead of crashing the whole pipeline.
    present = [r for r in registry if Path(r['per_leadtime_csv']).exists()]
    if not present:
        print("[stage:aggregate] SKIP — no per_leadtime.csv found yet for any "
              "registered run:")
        for r in registry:
            print(f"    [missing] {r['run_id']}: {r['per_leadtime_csv']}")
        return True
    missing = [r for r in registry if r not in present]
    for r in missing:
        print(f"    [skip row] {r['run_id']}: {r['per_leadtime_csv']} not found")

    registry_path = Path(cfg['paper_results_dir']) / 'registry.json'
    Path(cfg['paper_results_dir']).mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(present, f, indent=2)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / 'analysis' / 'aggregate_results.py'),
        '--registry', str(registry_path),
        '--out',      cfg['paper_results_dir'],
    ]
    print('  $', ' '.join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"aggregate_results.py exited with code {result.returncode}")

    # Auto-generate the paper figures whose data already exists on disk.
    # Figures needing per-sample npz dumps (RAPSD, ECDF, failure cases, etc.)
    # are deferred to a separate post-training dump step.
    figs_dir = Path(cfg['paper_results_dir']) / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)
    figures_script = SCRIPTS_DIR / 'analysis' / 'figures.py'

    figure_jobs = [
        # Fig 3 — lead-time × metric curves (from aggregated main results)
        ('lead_time',
         Path(cfg['paper_results_dir']) / 'main_results_table.csv',
         figs_dir / 'fig3_lead_time_metrics.png'),
        # Fig 5 — sampler-budget × quality (from sampler_steps_combined.csv if exists)
        ('sampler',
         Path(cfg['paper_results_dir']) / 'sampler_steps_combined.csv',
         figs_dir / 'fig5_sampler_budget.png'),
    ]
    n_ok, n_skip = 0, 0
    for kind, in_path, out_path in figure_jobs:
        if not Path(in_path).exists():
            print(f"  [skip] {out_path.name}: input {in_path} not found")
            n_skip += 1
            continue
        sub = subprocess.run(
            [sys.executable, str(figures_script),
             '--kind', kind, '--in', str(in_path), '--out', str(out_path)],
            cwd=str(REPO_ROOT))
        if sub.returncode == 0:
            print(f"  [fig] {out_path}")
            n_ok += 1
        else:
            print(f"  [warn] figures.py failed for kind={kind}; continuing")
            n_skip += 1

    print(f"[stage:aggregate] OK → {out_csv}  (+{n_ok} figs, {n_skip} skipped)")
    return True


# ============================================================================
# Stage dispatcher
# ============================================================================

STAGE_FNS = {
    'train_vae':              stage_train_vae,
    'precompute_cache':       stage_precompute_cache,
    'train_single_frame_rf':  stage_train_single_frame_rf,
    'train_multi_frame_rf':   stage_train_multi_frame_rf,
    'train_cascaded_diffusion': stage_train_cascaded_diffusion,
    'kfold':                  stage_kfold,
    'aggregate':              stage_aggregate,
}


def run_pipeline(cfg: dict, stages_to_run):
    print(f"\nProject root: {REPO_ROOT}")
    print(f"Dataset root: {cfg['root_dir']}"
          f"{'  [MISSING!]' if not Path(cfg['root_dir']).exists() else ''}")
    print(f"Device:       {cfg['device']}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU:          {gpu_name} ({vram_total_gb:.1f} GB VRAM)")
        print(f"AMP:          {cfg.get('amp_dtype', 'fp16')} via torch.amp.autocast")
        print(f"cudnn benchmark: {torch.backends.cudnn.benchmark}")
        print(f"TF32 matmul:     {torch.backends.cuda.matmul.allow_tf32}")
    print(f"num_workers:  {cfg['num_workers']}  (CPU cores available)")
    print(f"Stages to run: {stages_to_run}")

    for name in stages_to_run:
        fn = STAGE_FNS.get(name)
        if fn is None:
            print(f"  [skip] unknown stage '{name}'")
            continue
        fn(cfg)

    print(f"\n{'=' * 70}\nPipeline complete.\n{'=' * 70}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=list(STAGE_FNS.keys()),
                        help='Run a single stage (else runs all enabled).')
    parser.add_argument('--skip', nargs='+', choices=list(STAGE_FNS.keys()),
                        default=[], help='Stages to skip this run.')
    parser.add_argument('--use-kfold', action='store_true',
                        help='Force k-fold ON for this run.')
    parser.add_argument('--run-tag', default=None,
                        help='Override CONFIG["run_tag"] (e.g. for an ablation row).')
    parser.add_argument('--enable-phys-loss', action='store_true',
                        help='Force use_phys_loss=True (turns on all 3 components).')
    parser.add_argument('--variant', choices=['v1', 'v2'], default=None,
                        help='Multi-frame backbone: v1 (channel-stack) or v2 '
                             '(factorised spatio-temporal). Appends _v2 to run_tag.')
    parser.add_argument('--root', default=None,
                        help='Override CONFIG["root_dir"] (SETCD root).')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = dict(CONFIG)

    # CLI overrides
    if args.use_kfold:
        cfg['use_kfold'] = True
    if args.run_tag:
        cfg['run_tag'] = args.run_tag
    if args.enable_phys_loss:
        cfg['use_phys_loss']        = True
        cfg['use_continuity']       = True
        cfg['use_geostrophic']      = True
        cfg['use_track_consistency'] = True
        # bump the run_tag if the user didn't override it manually
        if not args.run_tag:
            cfg['run_tag'] = cfg['run_tag'].replace('no_phys', 'with_phys')
    if args.variant:
        cfg['p2_variant'] = args.variant
        # Tag the run so V2 checkpoints/outputs never collide with V1.
        if not args.run_tag and args.variant == 'v2' and '_v2' not in cfg['run_tag']:
            cfg['run_tag'] = cfg['run_tag'] + '_v2'
    if args.root:
        cfg['root_dir'] = args.root

    if args.stage:
        stages_to_run = [args.stage]
    else:
        stages_to_run = [s for s, on in STAGES.items()
                         if on and s not in set(args.skip)]
        # also include kfold if explicitly enabled via flag
        if cfg.get('use_kfold') and 'kfold' not in stages_to_run \
           and 'kfold' not in set(args.skip):
            stages_to_run.append('kfold')

    run_pipeline(cfg, stages_to_run)


if __name__ == '__main__':
    main()
