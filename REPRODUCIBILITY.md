# Reproducibility Guide

This guide describes the end-to-end workflow used to reproduce the public
AdsorbFlow results.

## 1. Prepare The Environment

Install the package following `README.md`. A CUDA GPU is recommended for
training and large-scale inference. CPU-only installation is useful for reading
data and running small utility scripts, but not for full reproduction.

## 2. Prepare Data

Download OC20-Dense data and mapping files from the original sources. Place or
generate local datasets following `DATA.md`.

Required local inputs for full reproduction:

- Training LMDB.
- Validation and evaluation LMDBs.
- OC20-Dense tags and reference-energy mappings.
- AdsorbFlow generator checkpoints or compute resources for retraining.
- GemNet-OC relaxer checkpoint.

## 3. Train Or Load AdsorbFlow

Use `configs/flow/eqv2_conditional_flow.yml` for the main EqV2 flow baseline
and `configs/flow/painn_conditional_flow.yml` for the PaiNN baseline.

The v2 EqV2 configuration with Fourier energy encoding and endpoint-weighted
flow training is available at:

```text
configs/flow/eqv2_fourier_cosine_v2.yml
```

Command templates are provided in `EXPERIMENT_COMMANDS.md`.

## 4. Generate And Relax Candidates

Run the grid-search script to generate candidates, relax them with GemNet-OC,
and evaluate MLFF-level success and anomaly rates:

```bash
python -u scripts/grid_search_cfg_flow.py \
  --cfg-scales 0 1 3 5 7 10 \
  --num-steps 5 10 30 \
  --flow-checkpoint checkpoints/{adsorbflow_checkpoint}.pt \
  --relax-checkpoint configs/relaxation/gemnet_oc/gemnet-oc.pt \
  --model-type eqv2 \
  --nsites 10 \
  --gpus 4 \
  --skip-existing
```

Tracked MLFF grid summaries are stored in
`paper_artifacts/grid_search_results/`.

## 5. Verify With DFT

For paper-level SR@k, use the scripts in `scripts/cluster_vasp/` to prepare
VASP single-point input folders, run VASP externally, and analyze completed
outputs.

The repository does not distribute VASP, pseudopotentials, or cluster launch
scripts. Configure these locally.

## 6. Regenerate Figures

Case-study CSVs are tracked under `examples/`. Plotting scripts are under
`scripts/viz/`, and generated publication figures are stored under `figures/`.

Typical entry point:

```bash
python scripts/viz/plot_casestudies.py
```

## Known Non-Determinism

Training and generation may vary with GPU type, PyTorch/CUDA versions,
distributed-training order, random seeds, and MLFF relaxation details. Report
the exact checkpoint, guidance scale, ODE step count, anomaly filter, and DFT
selection protocol when comparing results.
