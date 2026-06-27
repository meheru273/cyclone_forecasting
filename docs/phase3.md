# Phase 3 — Paper-Ready Study: Nath et al. Baseline, k-Fold, Cross-Basin OOD, Significance

> **Goal.** Convert the Phase-1 / Phase-2 model + ablations into a defensible empirical study. One external baseline (Nath et al. cascaded diffusion), the k-fold robustness check, a cross-basin OOD test using SETCD metadata, and bootstrap + Diebold–Mariano significance tests.
>
> **Pre-requisite.** Phase 2 done; one winning multi-frame 2-RF configuration identified from the ablation matrix; `phase2_report.md` committed.

---

## 3.1  Scope (what is *not* here)

- No internal LDM/DDPM ablation rows (dropped in Phase 0 §0.4). Diffusion comparison lives in §3.3 below as the Nath et al. external baseline.
- No 2501.16003 reproduction. It is acknowledged in related work but not reproduced. (User decision; the Nath et al. baseline is sufficient for the diffusion-vs-flow story because 2501.16003 also benchmarks against Nath et al., so our Nath number is directly comparable to theirs.)

---

## 3.2  The main results table (Table 1 of the paper)

Every row uses the year-based split (train 2012–2020 / val 2021 / test 2022), same MC-VAE, same backbone scaffolding (`base=128, mults=(1,2,3,4)`).

| # | Row | Description | Source |
|---|---|---|---|
| **R1** | Persistence | Predict next frame = last input frame. Trivial baseline; sanity check. | New |
| **R2** | Optical-flow extrapolation | Farneback / RAFT applied to ERA5 U/V, extrapolated 3 / 6 / 9 h. Trivial-but-physical baseline. | New |
| **R3** | Nath et al. (`arXiv:2310.01690`) cascaded DDPM — reproduced | Their published numbers on their own 2019–2023 SETCD split. | §3.3.1 |
| **R4** | Nath et al. — re-trained on our setting | Their architecture, our split (2012–2020 / 2021 / 2022), same metric suite as our models. | §3.3.2 |
| **R5** | Ours — single-frame 2-RF (Phase 1 P1-RF2), rolled out for t+1, t+2, t+3 | The S1 falsification baseline. | Phase 2 |
| **R6** | Ours — multi-frame 2-RF (Phase-2 ablation winner), **without** physics loss | Headline row. | Phase 2 |
| **R7** | Ours — multi-frame 2-RF + physics losses (best G row) | Headline + physics. | Phase 2 |

Bold the better of R6 and R7 as the published headline (almost certainly R7 if the physics losses help; if not, R6).

Column set per row: `PSNR | SSIM | LPIPS | CRPS | RAPSD-L1 | track-km | MSLP-err | params | inf-ms | train-h`. All reported as mean across 3 seeds for R6 and R7 (the headline rows); single-run for R1–R5.

---

## 3.3  The Nath et al. external baseline (§3.3.1 and §3.3.2 — both runs are needed)

### 3.3.1  Reproduction sanity check

- Obtain Nath et al.'s released code (linked from arXiv `2310.01690`). If unavailable, re-implement the cascaded 3-stage DDPM (forecast 64² → super-res 128² → super-res 256²) from the paper.
- Train on their declared split (51 storms, Jan 2019 – Mar 2023, six basins) — their data shard or our equivalent.
- Reproduce their published SSIM > 0.5, PSNR > 20 dB at 36 h roll-out.
- Acceptable tolerance: ± 5 % on each metric. Report any gap.

This row stays at the same hyperparameters / training schedule as the original. If we can't reproduce within tolerance, we say so in the paper and use our re-trained number (§3.3.2) as the comparator.

### 3.3.2  Apples-to-apples comparison on our setting

- Same Nath architecture (cascaded pixel-space DDPM, 3 stages).
- Our split (2012–2020 / 2021 / 2022), our 256² resolution at the final cascade stage, *5 channels jointly* (IR + ERA5) at the final stage rather than IR-only. This is a small extension to their model — implementable by setting the final-stage output channels to 5 — and is documented in the paper.
- Same evaluation suite as our models (§3.5).
- This is the row that goes head-to-head with R6 / R7 in Table 1.

---

## 3.4  Storm-level k-fold cross-validation (within train years)

Random sample-level k-fold leaks consecutive frames of the same storm across folds. Split at the **storm level**.

### 3.4.1  Protocol

- Take all storms from train years 2012–2020 (call this `S_train`).
- Stratify by SETCD-encoded basin (parse the SID prefix; the helper `parse_storm_coordinates` at `multi_channel_diffusion.py:48-62` already extracts hemisphere + lat/lon, which maps to a basin via IBTrACS basin polygons).
- **k = 5** folds; each fold's test slice is a stratified 20 % of `S_train`.
- Retrain only the **headline row R6** (multi-frame 2-RF, no physics) for each fold, from the same VAE checkpoint, 80 epochs (saves compute vs the headline's 100; verify 80-vs-100 delta is < 0.3 dB before committing).
- 2021 and 2022 are *not* used for the k-fold (they remain the val/test holdouts).

### 3.4.2  Reporting

- `kfold_results.csv` with per-fold metrics.
- Mean ± std table.
- Box plot for the headline metrics (PSNR, SSIM, CRPS, track-km).

If the across-fold std is high relative to the R6 vs R4 gap, the headline claim is not robust and the paper must acknowledge it.

### 3.4.3  Cost

5 folds × ~12 h on RTX 4090 ≈ 60 h. Bearable.

---

## 3.5  Cross-basin OOD test (using SETCD metadata, not ibtracs_all.csv)

### 3.5.1  Protocol

- Build the basin tally from SETCD storm IDs (post-hoc on the year-counting script's output).
- Pick the *most populous non-headline basin* in 2012–2020 (likely WPAC) and hold out **all** of its storms from training.
- Retrain headline R6 once.
- Evaluate on the held-out basin's storms (using their native test-time positions). Compare degradation vs in-basin test.

### 3.5.2  Reporting

`cross_basin_results.csv`: per-basin metrics for both the all-basin headline and the leave-one-basin-out variant. Discuss whether the model has learned basin-specific cloud-morphology shortcuts vs more universal dynamics.

A degradation of < 1.5 dB PSNR + < 5 % SSIM is a defensible claim of "basin-generalising"; more than that is a finding worth honestly reporting.

### 3.5.3  Cost

1 retrain × ~12 h = 12 h.

---

## 3.6  Evaluation metric stack (canonical for Phase 3)

Reproduced from `phase2.md` §2.8 for completeness. All metrics computed per channel + overall, and per lead time for multi-frame rows.

| Category | Metrics | New? |
|---|---|---|
| Pixel | MAE, MSE, PSNR, SSIM, LPIPS | LPIPS new |
| Spectral | RAPSD-L1 between log-RAPSD curves | New |
| Probabilistic | CRPS (N=10 samples per input), Sliced-W1 | New |
| Threshold (weather) | CSI / POD / FAR at IR < 220 K and < 200 K | New |
| TC track | Steering-flow displacement (deg), great-circle (km) per lead | km derived is new |
| TC intensity | MSLP error (hPa), max |U|, |V| error (m/s) | New |
| Multi-frame | FVD | New |
| Compute | Param count, train h/epoch, inf ms/sample @ 1/4/16/50 ODE steps | New columns |

I am explicitly **not** using Inception Score (meaningless for satellite imagery — no class labels) and **not** treating FID as a per-sample statistic (it isn't — bootstrap on FID by resampling the 200-sample eval set 1 000 times).

---

## 3.7  Plots (paper figures)

| Figure | What it shows | Source |
|---|---|---|
| Fig 1 | Architecture: VAE + multi-frame UNet + RF path, conditioning fusion close-up | Hand-drawn / SVG |
| Fig 2 | Qualitative samples — 3 storms × (Input/Target/Predicted) × (IR + 4 ERA5) × (t+1, t+2, t+3) | Phase-2 `save_samples` |
| Fig 3 | Lead-time × metric (PSNR, SSIM, CRPS, track-km) for R3/R4/R5/R6/R7 | Phase-2/3 `per_leadtime.csv` |
| Fig 4 | RAPSD log-log per channel — R4 vs R6 vs R7 (do flow models preserve high-frequency content?) | New |
| Fig 5 | Sample-budget × quality scatter (ODE steps on x, PSNR on y) — R6/R7 only | Phase-1/2 sweep CSVs |
| Fig 6 | k-fold box plot (PSNR, SSIM, CRPS, track-km) | §3.4 |
| Fig 7 | Track-overlay basemap: 6 representative storms (best / median / worst PSNR) — predicted vs truth tracks on a map (cartopy) | New |
| Fig 8 | Reliability diagram (predicted CDF vs empirical CDF for IR pixel intensity) | New |
| Fig 9 | Spatial error heatmap, val-averaged | New |
| Fig 10 | Per-channel ECDF: predicted vs target value distribution | New |
| Fig 11 | Failure-case gallery (6 worst predictions, with diagnostics) | New |
| Fig 12 (suppl.) | Cross-basin degradation table + per-basin metric bars | §3.5 |
| Fig 13 (suppl.) | Physics-loss component ablation (G2–G5) with divergence histogram before/after | Phase-2 |

---

## 3.8  Statistical significance

Per-sample metrics on consecutive frames of the same storm are not independent. Use cluster-aware resampling.

### 3.8.1  Paired bootstrap CIs

For each metric `M` and each pair `(model_A, model_B)`:

1. For each storm in the test set, compute `M_A(storm) − M_B(storm)` (within-storm difference, averaged over all that storm's val frames).
2. Resample storms with replacement, 10 000 times.
3. Resampled mean of differences; 95 % CI = [2.5 %, 97.5 %].

If CI excludes 0, the difference is significant at α = 0.05 (cluster-corrected).

Apply to: R6 vs R4 (our headline vs Nath); R7 vs R6 (does physics actually help?); R6 vs R5 (does multi-frame beat single-frame rollout?); each *k*-fold mean vs the pooled mean.

### 3.8.2  Diebold–Mariano

For each lead time and each metric:

```
d_t = M_A(t) − M_B(t)
DM  = mean(d_t) / sqrt(Var_HAC(d_t) / N)
```

Newey-West variance with lag = `k − 1` for the 3-hour autocorrelation. Report DM statistic and p-value.

### 3.8.3  Wilcoxon sanity

For SSIM (clipped to ~1) and great-circle km (heavy right tail), additionally report a Wilcoxon signed-rank p-value. If bootstrap and Wilcoxon agree, cite bootstrap; if not, report both and discuss.

### 3.8.4  Don'ts

- Do not report bare standard errors over individual generated frames as if they were independent.
- Do not run t-tests on FID — bootstrap on the eval-sample set instead.
- Do not cherry-pick seeds. R6 and R7 are mean over 3 seeds.

---

## 3.9  Compute budget

Year-based split + drop of internal LDM/DDPM ablation = a smaller study than before.

| Run set | # runs | RTX 4090 hours |
|---|---|---|
| R1, R2 (persistence, optical flow) | 0 train | < 1 (eval only) |
| R3 — Nath et al. reproduction | 1 | ~24 (theirs is 64²/128²/256² cascaded, light) |
| R4 — Nath et al. retrained on our setting | 1 | ~30 |
| R5 — Phase-1 rolled out | 0 | already trained in Phase 1 |
| R6 — multi-frame 2-RF best, 3 seeds | 3 | ~3 × 14 = 42 |
| R7 — multi-frame 2-RF + physics, 3 seeds | 3 | ~3 × 14 = 42 |
| k-fold of R6 (80 ep) | 5 | ~5 × 10 = 50 |
| Cross-basin retrain of R6 | 1 | ~12 |
| Stats / plots | n/a | local CPU |
| **Total** | | **~200 RTX 4090 hours** ≈ 8.5 days at 24/7 |

Comfortable on a single GPU run continuously.

---

## 3.10  Cuts under compute pressure (in order of "cut first")

1. R7 down to 1 seed instead of 3 (−28 h).
2. Cross-basin OOD (−12 h).
3. R3 (Nath reproduction). Keep only R4 (Nath on our setting). Saves ~24 h; the paper says "we re-train Nath et al. on our split; their original numbers are reported as in their paper" — acceptable.

What we **do not cut** under any pressure:

- R4 (Nath baseline on our setting).
- The k-fold for the headline row.
- The bootstrap CI on R6 vs R4.

---

## 3.11  Acceptance criteria

1. `ablation_table.csv` (or `main_results_table.csv`) with R1–R7, all metric columns, programmatically generated from per-run CSVs.
2. `nath_repro.csv` and `nath_oursetting.csv` for R3 and R4.
3. `kfold_results.csv` and Fig 6.
4. `cross_basin_results.csv` and Fig 12.
5. `significance_tests.json` with bootstrap CI + DM stat + Wilcoxon for each headline pair × each metric × each lead time.
6. All 13 figures generated from a single `analysis/figures.ipynb`.
7. `paper_results.md` drops directly into the paper's Section 4 with no hand-editing.

When 1–7 hold, the empirical work is done.
