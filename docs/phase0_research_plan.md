# Phase 0 — Research Plan & Narrative

> **Working title:** *Cyclone Forecasting with Rectified Flow Matching: A Joint Latent Generative Model for GRIDSAT-IR and ERA5 Reanalysis*
>
> **Status:** Design draft (revision 2 — post user feedback on 2026-05-26). No code changes here — this document drives Phases 1–3.

---

## 0.1  Stand-back view (what is the thesis actually claiming?)

The user proposed three novelty pillars. After a literature scan (May 2026) and discussion, the final contribution set is below.

| User-proposed novelty | Assessment | Final form |
|---|---|---|
| **N1.** Flow / rectified-flow for cyclone forecasting | **Outdated as written.** FlowCast (ICLR 2026, CFM for precipitation), FlowCast-ODE, 3DTCR, RectiCast, MFC-RFNet are all 2025 work in this space. *Refined:* first application of rectified flow specifically to **joint IR + reanalysis** TC forecasting in latent space. | ✅ Kept, refined |
| **N2.** Single model generates both GRIDSAT IR and ERA5 fields | **Holds.** Prior work generates either IR satellite imagery (Nath et al. 2310.01690, 2501.16003) or ERA5 fields (Phys-Diff, FuXi family) but not both jointly. | ✅ Keep |
| **N3.** Longer forecasting horizon | At 256² and 5-channel joint, the 9 h horizon is unmatched. (2501.16003 reaches 50 h but at 64² IR-only.) | ✅ Keep, scoped to the joint-channel setting |
| **N4.** Physics-based forecasting | *Refined:* add three explicit, flag-toggled, finite-difference physics losses (mass continuity, geostrophic balance, track consistency) and ablate them. Off by default; turned on after the base model is stable. | ✅ Keep |

### 0.1.1  Final contribution set (paper abstract)

1. **C1 — Joint generative forecasting.** A single latent rectified-flow model that produces both 1-channel GRIDSAT-IR imagery and 4-channel ERA5 reanalysis fields (U-wind, V-wind, T, P) at 256 × 256 for a 9-hour cyclone forecast at 3-hour cadence.
2. **C2 — Multi-frame conditioning + multi-frame prediction.** (t-2, t-1, t) → (t+1, t+2, t+3) with V1 channel-stack and V2 factorised-attention backbones ablated against a single-frame rolled-out baseline. Internal ablations: context window, temporal mixing, conditioning fusion.
3. **C3 — Extended horizon at high resolution + joint channels.** The 256² 5-channel 9 h horizon is the unique cell in the existing literature grid.
4. **C4 — Physics-soft losses (optional, flag-controlled).** Continuity, geostrophic, track-consistency. Each ablated independently. Reported as an *improvement on top of* the base model, not the headline.
5. **C5 — Statistical rigour.** Year-based split (2012–2020 / 2021 / 2022), storm-level k-fold within train years, cross-basin OOD (SETCD metadata, not IBTrACS), paired bootstrap CIs, Diebold–Mariano forecast-skill tests.

**Headline external comparison:** Nath et al. cascaded diffusion (`arXiv:2310.01690`, the original SETCD paper). Same dataset family, different architectural family (pixel-space cascaded DDPM vs latent rectified flow).

---

## 0.2  Disagreements I am still flagging with the user (read carefully)

Items 1, 2, 3, 4 from revision 1 were resolved. The remaining flag:

1. **`ibtracs_all.csv` is irrelevant to SETCD.** Confirmed by the user. All Phase-3 basin-stratification and cross-basin OOD work uses **SETCD's own per-storm metadata** (storm IDs encode genesis basin in IBTrACS-SID format: char 8 = N/S hemisphere; positions 9–11 = latitude; positions 12–14 = longitude; map → basin via standard IBTrACS basin polygon). The `parse_storm_coordinates` helper already in `multi_channel_diffusion.py:48-62` does most of this.

2. **Cross-basin OOD requires SETCD to be multi-basin in our shard.** The user confirmed SETCD is multi-basin globally; we need to verify it within the 2012–2020 years we use. Output of `count_storms_per_year.py` plus a follow-up basin-tally step is the prerequisite.

3. **2021 alone for validation might be thin.** Some seasons are quiet. If `count_storms_per_year.py` reports < ~20 storms in 2021, we widen to `VAL_YEARS = ['2020', '2021']` and slide train down to 2012–2019. Decision deferred to the script's output.

---

## 0.3  Frozen experimental scaffolding (do not change between phases)

| Item | Value | Reason |
|---|---|---|
| Dataset | SETCD (GRIDSAT-B1 + ERA5 + IBTrACS), multi-basin | Already curated; SETCD is the standard |
| Split | **Year-based** — train 2012–2020 / val 2021 / test 2022 (subject to confirmation by `count_storms_per_year.py`) | No more random storm selection; deterministic, reviewer-friendly |
| Spatial resolution | 256 × 256 | Reviewers will not accept 64 × 64 as the headline |
| Native cadence | 3 hours | GRIDSAT-B1 native; ERA5 sub-sampled to match |
| Latent VAE | Frozen `MultiChannelVAE(in=5, latent=4, base=64)` with `latent_scale_factor` recomputed from data | One VAE for *all* generative ablations |
| Backbone | `DiffusionUNet`, **`base_channels=128`**, **`channel_mults=(1,2,3,4)`**, dropout 0.1, attention at the middle encoder + MidBlock | Updated from the old (64, 1-2-4) — fixed underfitting |
| EMA decay | 0.9999 | |
| Optimiser | AdamW, lr 1e-4 → 5e-5 cosine, weight decay 1e-5, 5-epoch warmup | |
| AMP | BF16 on Ampere+ (RTX 4090) | Avoid FP16 grad-scaler quirks |
| Conditioning | Image + temporal + per-frame IBTrACS lat/lon via `coord_proj`, additive fusion (C-Add — to be challenged in Ablation C) | Matches MC-Diff; required for all flow scripts after Phase 1 |
| Training objective | **2-Rectified Flow** (default, revisit if S3 ablation prefers OT-CFM or 1-RF) | Phase-2 §2.6 ablation S3 |
| Sampler | ODE Euler with 4 steps for evaluation; 1 / 16 / 50 also evaluated at the end | Speed-quality budget |
| Epochs | 100 (single-frame); 80–100 (multi-frame; A100 — no, RTX 4090) | Compute is unlimited but cosine LR needs a finite horizon |
| Eval samples / epoch | 200 | Stable estimates without huge cost |
| Hardware | **RTX 4090 (16 GB dedicated VRAM, 128 GB system RAM), 24/7, no time limit** | Single-GPU, BF16, possibly grad-checkpointing for multi-frame V2 |
| Physics loss | `use_phys_loss=False` by default. Turned on only in Phase-2 Ablation G. | User decision |

---

## 0.4  Phase map

| Phase | Doc | Goal | Output artefacts |
|---|---|---|---|
| 0 | `phase0_research_plan.md` (this) | High-level narrative, novelty audit, frozen scaffolding | This doc |
| 1 | `phase1.md` | Bring the flow scripts to feature parity with MC-Diff; train a single-frame 2-RF reference; lock in the year-based split | Updated `conditional_flow_matching.py`, `rectified_flow_score_matching.py`; trained checkpoints; matched eval CSVs in `results-new/{cfm,rf1,rf2}/metrics/` |
| 2 | `phase2.md` | Multi-frame model + 4 ablations (A context, B temporal mixing, C conditioning fusion, G physics) + S1–S5 cheap add-ons | `MultiFrameCycloneDataset`; trained multi-frame checkpoints; ablation matrix CSV; `phase2_report.md` |
| 3 | `phase3.md` | Convert into a paper: Nath et al. external baseline reproduced *and* re-trained on our setting, storm-level k-fold, cross-basin OOD, statistical tests, all figures | `ablation_table.csv`, `kfold_results.csv`, `cross_basin_results.csv`, `nath_repro.csv`, `nath_oursetting.csv`, `significance_tests.json`, all figures, `paper_results.md` |

Internal LDM/DDPM ablation rows have been **removed** from Phase 3. Diffusion is represented by the Nath et al. external baseline only.

---

## 0.5  Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MC-VAE bottleneck (FID ≈ 175 floor in `result_summary.md`) caps every downstream model | High | High | Phase-1 optional VAE touch-up (`phase1.md` §1.6). Report decoded-vs-VAE-reconstructed-target metrics so the VAE floor doesn't mask flow ablation deltas. |
| 2021 has too few storms for validation | Medium | Medium | Decided by `count_storms_per_year.py` output. Fall-back: widen `VAL_YEARS` to `['2020','2021']`. |
| Multi-frame doesn't beat rolled-out single-frame (`phase2.md` S1) | Medium | High | Falsification test, exactly the question A and B ask. If it loses, narrow the paper to single-frame joint forecasting — still novel. |
| Physics losses destabilise | Low (turned on late, small λ, warmup) | Low | `phase2.md` §2.9: warmup ramp, masked core, ablate independently. |
| Cross-basin OOD impossible because SETCD basin coverage is uneven in 2012–2020 | Low (SETCD is global) | Medium | Confirm with basin tally; worst case downgrade to held-out-year and keep storm-level k-fold. |
| Compute estimates are wrong (BF16 might not save as much as hoped) | Medium | Low | 24/7 runtime is the safety margin; we can absorb 2–3× overrun. |
| Nath et al. reproduction fails (theirs is pixel cascade; their code may not match what we read in the paper) | Medium | Medium | Reproduce their reported numbers on their own SETCD subset first as a sanity check (Phase-3 §3.3). |

---

## 0.6  Open questions — all resolved (2026-05-26)

| # | Question | Answer |
|---|---|---|
| 1 | Compute envelope | RTX 4090, 16 GB VRAM, 128 GB RAM, 24/7. (Likely mobile 4090; desktop is 24 GB. Either way, design budget = 16 GB.) |
| 2 | Cross-basin coverage | SETCD is multi-basin by design (user confirmed). Tally after `count_storms_per_year.py`. |
| 3 | Phase-2 horizon | Option A: 3 in / 3 out at 3-hour cadence. Locked. |
| 4 | Physics-loss appetite | Keep C4, but behind a flag, off by default, ablated separately. |
| 5 | Primary external baseline | Nath et al. `arXiv:2310.01690`. |
| 6 | Internal LDM/DDPM ablation | Dropped. MC-Diff results in `result_summary.md` remain as historical reference, not as a published row. |
| 7 | Year-counting script | Written: `scripts/count_storms_per_year.py`. Run before Phase 1 begins. |

---

## 0.7  Headline reading order for a reader of all four docs

1. `phase0_research_plan.md` (this) — what we are building, why, what we are *not* doing.
2. `phase2.md` — the headline experiment, multi-frame ablation matrix.
3. `phase1.md` — the prerequisite scaffolding work.
4. `phase3.md` — paper-ready results, k-fold, statistics, baseline reproductions.
