# Repository Structure

```text
adsorbdiff/
  datasets/                 LMDB dataset wrappers and metadata utilities
  models/                   PaiNN, EquiformerV2, GemNet-OC, embeddings
  placement/                adsorbate/slab utilities and anomaly detection
  relaxation/               MLFF relaxation and flow/diffusion samplers
  trainers/                 flow-matching and denoising trainers

configs/
  flow/                     AdsorbFlow training configurations
  denoising/                inherited AdsorbDiff denoising baselines
  relaxation/gemnet_oc/     GemNet-OC relaxation template and small wrapper asset

scripts/
  case_studies/             CO2RR, NO3RR, OER, and NRR case-study runners
  cluster_vasp/             DFT input generation and SR analysis
  create_lmdbs/             OC20-Dense preprocessing utilities
  evaluation/               grid-search, relaxation, and batch evaluation helpers
  training_utils/           LMDB splitting and metadata utilities
  viz/                      figure-generation scripts

examples/
  CO2RR/ NO3RR/ OER/ NRR/   lightweight case-study CSV outputs

figures/
  ablation/ casestudies/ no3rr/
                             publication figures generated from tracked outputs

docs/
  COMMANDS.md DATA.md MODEL_CARD.md PAPER_RESULTS.md REPRODUCIBILITY.md
                             public documentation and result audit notes

paper_artifacts/
  grid_search_results/      compact JSONL summaries for MLFF grid searches
```

Large checkpoints, LMDB datasets, VASP outputs, trajectories, and external
Open-Catalyst-Dataset checkouts are not tracked in Git. See `DATA.md`.
