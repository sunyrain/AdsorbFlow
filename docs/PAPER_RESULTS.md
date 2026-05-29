# Paper Results

This document mirrors the result tables reported in the AdsorbFlow manuscript
and points to the tracked artifacts needed to audit them.

## Core Claim

AdsorbFlow accelerates adsorbate placement by replacing long diffusion
sampling with short deterministic ODE integration over adsorbate translation
and rotation. Candidate placements are relaxed with a machine-learned force
field, filtered for structural anomalies, ranked by relaxed energy, and then
verified with DFT where required.

AdsorbFlow is an upstream adsorption-geometry generator. It does not replace
microkinetic modeling, reactor simulation, or process-system optimization.

## DFT-Verified OC20-Dense Results

### In-Distribution Validation

44 systems.

| Method | Backbone | Steps | SR@1 | SR@2 | SR@5 | SR@10 | Anom.@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| AdsorbML | rule-based | - | 9.1 | 20.5 | 34.1 | 47.7 | 6.8 |
| AdsorbDiff | EquiformerV2 | about 100 | 31.8 | 34.1 | 36.3 | 41.0 | 13.6 |
| AdsorbFlow | EquiformerV2 | 5 | 34.1 | 45.5 | 54.5 | 61.4 | 6.8 |
| AdsorbFlow | PaiNN | 5 | 27.3 | 34.1 | 45.5 | 47.7 | 13.6 |

### Out-of-Distribution Validation

50 systems.

| Method | Backbone | Steps | SR@1 | SR@2 | SR@5 | SR@10 | Anom.@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| AdsorbFlow | EquiformerV2 | 5 | 28.0 | 46.0 | 54.0 | 58.0 | 6.0 |
| AdsorbFlow | PaiNN | 5 | 32.0 | 42.0 | 44.0 | 46.0 | 6.0 |

## Hyperparameter Grid

Grid-search results are stored as JSONL in
`paper_artifacts/grid_search_results/`.

| Backbone | Selected guidance scale | Selected ODE steps | MLFF SR@10 | DFT SR@10 |
|---|---:|---:|---:|---:|
| EquiformerV2 | 7 | 5 | 72.7 | 61.4 |
| PaiNN | 5 | 5 | 63.6 | 47.7 |

## Case-Study Artifacts

Tracked case-study folders:

- `examples/CO2RR/`
- `examples/NO3RR/`
- `examples/OER/`
- `examples/NRR/`

Tracked figure folders:

- `figures/ablation/`
- `figures/casestudies/`
- `figures/no3rr/`

## Audit Notes

- Large training datasets, checkpoints, trajectories, VASP outputs, and raw
  cluster logs are excluded from Git.
- Reproduction requires local dataset/checkpoint placement as described in
  `DATA.md` and `REPRODUCIBILITY.md`.
- DFT verification requires a licensed VASP installation and local
  pseudopotentials.
