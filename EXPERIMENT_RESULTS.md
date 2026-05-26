# Experiment Results

This file summarizes the public result tables used to validate AdsorbFlow.
Detailed machine-readable grid-search outputs are provided in
`paper_artifacts/grid_search_results/`, and case-study CSV summaries are
provided under `examples/`.

## Evaluation Protocol

AdsorbFlow follows the OC20-Dense placement protocol. For each system and each
SR@k level, the workflow:

1. Generates k adsorbate placements.
2. Relaxes each candidate with GemNet-OC.
3. Removes anomalous relaxations, including desorption, dissociation, slab
   changes, or intercalation.
4. Selects the lowest relaxed MLFF-energy candidate among the remaining
   structures.
5. Verifies the selected candidate by VASP DFT single-point calculation.

A prediction is counted as successful when the verified adsorption energy is no
more than 0.1 eV above the best-known DFT reference minimum.

## OC20-Dense In-Distribution Validation

44 validation systems, MLFF ranking followed by DFT verification.

| Method | Backbone | Steps | SR@1 | SR@2 | SR@5 | SR@10 | Anom.@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| AdsorbML | rule-based | - | 9.1 | 20.5 | 34.1 | 47.7 | 6.8 |
| AdsorbDiff | EquiformerV2 | about 100 | 31.8 | 34.1 | 36.3 | 41.0 | 13.6 |
| AdsorbFlow | EquiformerV2 | 5 | 34.1 | 45.5 | 54.5 | 61.4 | 6.8 |
| AdsorbFlow | PaiNN | 5 | 27.3 | 34.1 | 45.5 | 47.7 | 13.6 |

## OC20-Dense Out-of-Distribution Validation

50 validation systems, disjoint from the in-distribution evaluation set.

| Method | Backbone | Steps | SR@1 | SR@2 | SR@5 | SR@10 | Anom.@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| AdsorbFlow | EquiformerV2 | 5 | 28.0 | 46.0 | 54.0 | 58.0 | 6.0 |
| AdsorbFlow | PaiNN | 5 | 32.0 | 42.0 | 44.0 | 46.0 | 6.0 |

## MLFF-Level Hyperparameter Selection

The main paper settings were selected using GemNet-OC relaxed energies before
running DFT verification.

| Backbone | Guidance scale | ODE steps | MLFF SR@10 | DFT SR@10 |
|---|---:|---:|---:|---:|
| EquiformerV2 | 7 | 5 | 72.7 | 61.4 |
| PaiNN | 5 | 5 | 63.6 | 47.7 |

Tracked JSONL files:

- `paper_artifacts/grid_search_results/runB_best_nsites10.jsonl`
- `paper_artifacts/grid_search_results/v2_best_nsites10_val_nonrelaxed_update_nsites10.jsonl`

## Case Studies

Lightweight CSV outputs for downstream reaction screening are stored under:

- `examples/CO2RR/`
- `examples/NO3RR/`
- `examples/OER/`
- `examples/NRR/`

Publication figures generated from these outputs are stored under `figures/`.
