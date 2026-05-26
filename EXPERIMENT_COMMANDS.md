# Reproducible Command Recipes

This file records public, path-agnostic command templates for reproducing the
main AdsorbFlow workflow. Replace placeholder paths such as
`/path/to/AdsorbFlow` and `checkpoints/<adsorbflow_checkpoint>.pt` with local
paths on your workstation or cluster.

For a higher-level workflow description, see `REPRODUCIBILITY.md`.

## Environment

```bash
conda create -n adsorbflow python=3.10
conda activate adsorbflow

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

git clone https://github.com/sunyrain/AdsorbFlow.git
cd AdsorbFlow
pip install -r requirements.txt
pip install -e .
```

## Train AdsorbFlow EqV2

```bash
python -u -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=1234 \
  main.py --mode train \
  --config-yml configs/flow/eqv2_conditional_flow.yml \
  --distributed \
  --identifier adsorbflow_eqv2_2d \
  --optim.p_cfg=0.20 \
  --optim.flow.allow_z=False \
  --optim.flow.tr_sigma_z_scale=0
```

The updated EqV2 v2 configuration used for energy-conditioned ablations is:

```bash
python -u -m torch.distributed.launch \
  --nproc_per_node=3 --master_port=1234 \
  main.py --mode train \
  --config-yml configs/flow/eqv2_fourier_cosine_v2.yml \
  --distributed \
  --identifier adsorbflow_eqv2_fourier_v2
```

## Train AdsorbFlow PaiNN

```bash
python -u -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=1235 \
  main.py --mode train \
  --config-yml configs/flow/painn_conditional_flow.yml \
  --distributed \
  --identifier adsorbflow_painn_2d \
  --optim.p_cfg=0.20 \
  --optim.flow.allow_z=False \
  --optim.flow.tr_sigma_z_scale=0
```

## MLFF-Level Grid Search

```bash
python -u scripts/grid_search_cfg_flow.py \
  --cfg-scales 0 1 3 5 7 10 \
  --num-steps 5 10 30 \
  --flow-checkpoint checkpoints/<adsorbflow_checkpoint>.pt \
  --relax-checkpoint configs/relaxation/gemnet_oc/gemnet-oc.pt \
  --model-type eqv2 \
  --nsites 10 \
  --gpus 4 \
  --master-port 1237 \
  --skip-existing
```

Use `--model-type painn` for the PaiNN backbone.

## Paper-Style DFT Verification

Prepare VASP input folders for selected SR@k levels:

```bash
python scripts/cluster_vasp/prepare_multilevel_vasp_inputs.py \
  --cfg-dir grid_search_runs/<run_name>/val_nonrelaxed_update/nsites_10/cfg7_steps5 \
  --tag-path oc20_dense_mappings/oc20dense_tags.pkl \
  --levels 1 2 5 10 \
  --max-sites 10
```

Compute paper-style SR@k after VASP single-point calculations complete:

```bash
python scripts/cluster_vasp/paper_faithful_sr.py \
  --cfg-dir grid_search_runs/<run_name>/val_nonrelaxed_update/nsites_10/cfg7_steps5 \
  --ref-energy-path oc20_dense_mappings/oc20dense_ref_energies.pkl
```

VASP itself, pseudopotentials, scheduler files, and cluster-specific launch
commands are not distributed in this repository.
