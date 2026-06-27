# Phase 2 + Phase 3 — Deferred Items

> Tracks what was **intentionally not implemented** in the Phase-2 / Phase-3
> code drop, and what it would take to build each. None of these block the
> headline result (multi-frame 2-RF, with or without physics, vs the Phase-1
> rolled-out baseline). All are reachable from the existing scaffolding.

---

## D1. V2 backbone — factorised 2-D spatial + 1-D temporal attention

**Status.** Skipped. Only V1 (channel-stacked 2-D UNet) is implemented in
`multi_frame_flow.py` as `MultiFrameFlowUNet`.

**Why deferred.** V1 is the "lightweight headline" row. V2 only matters if
V1's PSNR per lead-time degrades quickly with K_out > 3 or if temporal
coherence (FVD) is reviewer-flagged. Both can be measured first.

**What it takes.** ~150–200 lines in a new `MultiFrameFlowUNetV2` class:

- Add a `TemporalAttention1D` block: `nn.MultiheadAttention(embed_dim=ch,
  num_heads=num_heads, batch_first=True)` operating on the T-token sequence
  at each (h, w) location. Sinusoidal frame-index PE injected as bias.
- Reshape latents `(B, T, ch, h, w) ↔ (B*h*w, T, ch)` around the attention.
- Insert one such block after each `MidBlock` (one is enough — temporal
  signal is short).
- Stack input + output frames into the same `T = K_in + K_out` axis and use
  causal masking so output frames attend back to input frames but not to
  each other (matches the 2501.16003 layout).
- New `--variant v2` flag in `multi_frame_flow.py:main()`; route to the new
  UNet class. Trainer doesn't change beyond that.

**Compute envelope.** Activation memory ≈ T× V1; on the 16 GB RTX 4090 we
expect batch=1 + grad-accum=4 + grad-checkpointing on every level for
K_in=K_out=3. Wall-clock per epoch ≈ 2× V1.

**Required for paper claim.** Only if Phase-2 §2.6 Ablation B (V1 vs V2)
shows V2 wins by ≥ 1 dB PSNR or ≥ 5 % FVD — otherwise V1 is the headline
and V2 is a 1-row supplementary entry.

---

## D2. Conditioning fusion variants (C-Concat, C-CrossAttn)

**Status.** Only C-Add (additive, the current default in `ConditionEncoder`)
is wired through Phase-2. C-Concat and C-CrossAttn are described in
`docs/phase2.md` §2.5 but not implemented.

**Why deferred.** Single-axis ablation; adding it before V2 / physics is
turned on would inflate the run count. Most useful **after** the headline
row is locked.

**What it takes.**

- C-Concat: change the spatial fusion in `ConditionEncoder.forward` from
  `image_features + context_spatial` to
  `cat([image_features, context_spatial], dim=1)` followed by a 1×1 conv
  that reduces back to `output_channels`. Same for scalar fusion:
  `proj(cat([t_emb, ctx_emb]))` → `emb_dim` via a `Linear(2*emb_dim, emb_dim)`.
- C-CrossAttn: keep additive scalar fusion; in the UNet mid-block, replace
  the existing `SelfAttention2D` with a `CrossAttention2D` whose K/V come
  from the broadcast `context_spatial` tokens. One extra `nn.MultiheadAttention`.

**Trigger to add.** When Ablation C (`docs/phase2.md` §2.5) becomes blocking
for the paper. Adds 2 training runs.

---

## D3. FVD (Fréchet Video Distance) — multi-frame only

**Status.** Not implemented. `eval_metrics_extended.py` has FID-free
metrics only.

**Why deferred.** Adds an inflated dependency (typically a pretrained
I3D-Kinetics feature extractor). Quality information overlaps heavily with
per-lead PSNR/SSIM + RAPSD.

**What it takes.**

- `pip install fvd-pytorch` or the official `pytorch-fid`-style FVD pkg.
- New `fvd_score(samples, targets)` in `eval_metrics_extended.py` —
  accept per-frame `(B, T, C, H, W)` and return scalar.
- Add it to `evaluate_per_leadtime` once, *not* per lead — FVD is a
  whole-video statistic.

**Trigger to add.** Needed if a reviewer questions temporal coherence
beyond what per-lead curves can show.

---

## D4. Nath et al. (`arXiv:2310.01690`) baseline reproduction (Phase-3 §3.3)

**Status.** Not implemented. The paper's cascaded 3-stage DDPM
(64² → 128² → 256²) is a separate codebase.

**Why deferred.** Two reasons:

1. It is a non-trivial separate-paper re-implementation; the published
   code, if available, should be used in preference to a from-scratch
   re-implementation.
2. It does not interact with the multi-frame headline; it can run after
   Phase-2 results land.

**What it takes.**

- **Option A (preferred):** locate the released checkpoint or training
   script from the Nath et al. paper. Adapt their config to our
   `MultiFrameCycloneDataset` (year-based split) and run their training
   loop end-to-end on our setting.
- **Option B (fallback re-implementation):**
  - Cascade stage 1: pixel-space DDPM at 64² IR-only, 1000 ε-pred steps.
  - Cascade stage 2: super-resolution DDPM 64² → 128², conditioned on
    stage-1 output via channel-concat.
  - Cascade stage 3: super-resolution DDPM 128² → 256², ditto.
  - For our setting (R4), extend stage-3 output channels from 1 (IR) to
    5 (IR + ERA5) and train jointly.
- Per-storm CSV with the same columns as
  `outputs/<mf_run>/metrics/per_leadtime.csv` for direct comparison via
  `analysis/significance_tests.py`.

**Estimated effort.** ~1 week including reproduction sanity check (R3
in `docs/phase3.md` Table 1).

---

## D5. Cross-basin OOD runner (Phase-3 §3.5)

**Status.** The basin tagging helper (`basin_from_sid` in
`scripts/analysis/kfold_runner.py`) is implemented; a dedicated cross-basin
runner is not.

**Why deferred.** Implementation is a 50-line wrapper around the k-fold
runner; not worth a separate file until the user confirms basin coverage
in the 2012–2020 training years is sufficient (the basin tally is the
prerequisite — see open question #2 in `phase0_research_plan.md`).

**What it takes.** Copy `kfold_runner.py` to `cross_basin_runner.py`,
replace `stratified_kfold_storms` with a "hold out one basin" iterator.
Train once per basin to hold out. Stop after the first held-out basin
yields a stable estimate of degradation; running every basin is overkill.

---

## D6. Latent cache integration into the multi-frame trainer

**Status.** `LatentCache` class and the precompute script are written
(`scripts/latent_cache.py`). The trainer (`MultiFrameRFTrainer`) accepts
`latent_cache_dir` in its dataset config and the dataset returns
`target_latents` when caching is enabled — but the trainer's
`_encode_target_block` still calls `vae.encode` rather than checking the
batch for `target_latents`.

**Why deferred.** Two competing approaches and I want the user to pick
before committing:

- **A.** Strict cache lookup in the trainer: replace the
  `_encode_target_block` body with `return batch['target_latents'] *
  self.latent_scale_factor` when the key is present; fall back to the
  live encode otherwise. Cleanest, but requires every dataset sample
  to *always* hit the cache.
- **B.** Hybrid: trainer keeps `_encode_target_block` as-is, but the
  dataset short-circuits the disk-load step when the cache is hit,
  setting `target_latents` to the cached `mu` and the trainer multiplies
  by `latent_scale_factor` once.

**What it takes.** Either:

```python
# Option A — in MultiFrameRFTrainer._encode_target_block
@torch.no_grad()
def _encode_target_block(self, target_5ch_block, batch=None):
    if batch is not None and 'target_latents' in batch:
        z = batch['target_latents'].to(self.device)  # (B, K_out, latent_dim, h, w)
        B, K, ld, h, w = z.shape
        return (z.reshape(B, K * ld, h, w) * self.latent_scale_factor)
    # ... existing VAE encode fallback
```

Then `_train_step` and `validate` need to pass `batch=batch` into
`_encode_target_block`. ~10 lines.

**Trigger.** Whenever the first multi-frame run's wall-clock-per-epoch
is unacceptable. Until then, the trainer encodes live — costs ~1 extra
hour per epoch but works without the precompute step.

---

## D7. Bootstrap CI on FID / FVD pooled statistics

**Status.** `analysis/significance_tests.py` provides bootstrap CIs on
per-sample / per-storm metrics. It does *not* implement the "resample
the 200-sample eval set 1 000 times to bootstrap FID" pattern noted in
`docs/phase3.md` §3.6.4 — because FID isn't computed by the multi-frame
trainer (FVD is deferred — D3).

**Trigger.** Add when FID/FVD becomes a Table-1 column. ~20 lines:
sample eval-set indices with replacement, recompute the pooled feature
statistics on each resample, report the empirical CI of the resulting
scalar.

---

## D8. Per-seed averaging for headline rows (Phase-3 §3.8)

**Status.** Not automated. `phase3.md` says headline R6/R7 should be
mean-over-3-seeds.

**What it takes.** Three back-to-back runs of `multi_frame_flow.py` with
different `--seed` (currently hardcoded). Add a `--seed` CLI arg to the
trainer's main and seed `torch.manual_seed`, `numpy.random.seed`,
`random.seed`. Then aggregate the 3 per-run CSVs and report mean ± std.

The aggregation tool (`analysis/aggregate_results.py`) already handles
multiple runs via the registry JSON; the user just adds three entries
with the same row label.

---

## D9. Live IBTrACS coord lookup for predicted frames

**Status.** `MultiFrameCycloneDataset` only supplies `current_coords` for
input frames and `target_coords` for ground-truth future frames. At
*inference* time we don't have ground-truth target coords (they're what
we're trying to predict, indirectly). The headline track-km error metric
uses steering-flow integration from the current frame — this is fine.

But for a paper figure that **overlays** predicted vs truth tracks
(`docs/phase3.md` Fig 7), we need both per-frame predicted positions
(steering-integral cumulative from t) and the actual IBTrACS truth at
t+1, t+2, t+3.

**What it takes.** ~20 lines in a separate `dump_tracks.py`: walk val
samples, compute per-storm steering-integral trajectory from the
predicted U/V, emit a CSV with columns `storm, lead, gt_lat, gt_lon,
pred_lat, pred_lon`. `figures.py:track_overlay` already consumes that
format.

---

## Summary table

| ID | Item | Blocking the headline? | Effort | Trigger |
|----|------|---|---|---|
| D1 | V2 temporal attention | No | ~1 day | Ablation B if V1 underperforms |
| D2 | C-Concat / C-CrossAttn fusion | No | ~0.5 day each | Ablation C in the paper |
| D3 | FVD                 | No | ~0.5 day | Reviewer asks for temporal coherence |
| D4 | Nath et al. baseline | **Yes** for Table 1 R3/R4 | ~1 week | Phase 3 final table |
| D5 | Cross-basin runner   | No | ~0.5 day | After basin tally confirms coverage |
| D6 | Latent cache trainer integration | No (correctness); Yes (compute) | < 1 hour | First multi-frame epoch is too slow |
| D7 | FID/FVD bootstrap | No | ~1 hour | If FID/FVD makes it to Table 1 |
| D8 | 3-seed averaging | No (single seed publishable) | ~1 day wall-clock | Final paper polish |
| D9 | Track CSV dump | No (Fig 7 only) | ~1 hour | Generating Fig 7 |

D4 is the only deferred item that gates the Phase-3 main results table.
Everything else is post-headline polish.
