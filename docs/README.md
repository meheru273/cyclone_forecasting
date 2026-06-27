# Documentation Index

Design notes, phase plans, and result summaries for the thesis.

## Design plans

| Doc | Scope |
|---|---|
| [phase0_research_plan.md](phase0_research_plan.md) | Problem framing, dataset, scaffolding decisions |
| [phase1.md](phase1.md) | Bring flow scripts to parity with the diffusion baseline; year-based split; single-frame 1-RF reference |
| [phase2.md](phase2.md) | Multi-frame rectified flow (K_in=3, K_out=3) — V1 channel-stack vs V2 temporal attention; ablations A/B/C/G |
| [phase3.md](phase3.md) | External baseline (Nath et al.), test-storm rollout, sampler-budget sweep, paper figures |
| [phase23_deferred.md](phase23_deferred.md) | Items moved out of scope (reflow, k-fold, LDM ablation) |

## Reference

| Doc | Scope |
|---|---|
| [project_overview.md](project_overview.md) | Architecture overview, dataset schema, normalization stats |
| [environment.md](environment.md) | Runtime environment, dependencies, paths |
| [system.md](system.md) | System / training notes |
| [trajectory_design_analysis.md](trajectory_design_analysis.md) | Coordinate-conditioning design rationale |
| [latent_diffusion_changes.md](latent_diffusion_changes.md) | Change log for the diffusion baseline |
| [result_summary.md](result_summary.md) | Experiment results (pixel UNet vs LDM vs multi-channel diffusion) |
| [memory.md](memory.md) | Working notes |
