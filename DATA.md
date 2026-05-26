# Data And Checkpoints

AdsorbFlow relies on OC20-Dense structures, processed LMDB datasets, generated
trajectory folders, ML checkpoints, and VASP outputs. Only lightweight source
files, CSV summaries, figures, and JSONL result summaries are tracked in Git.

## Tracked In This Repository

| Path | Contents |
|---|---|
| `assets/fig1_pse_workflow.png` | Main workflow figure used in the manuscript and README |
| `examples/CO2RR/`, `examples/NO3RR/`, `examples/OER/`, `examples/NRR/` | Lightweight case-study CSV summaries and small helper files |
| `figures/` | Publication figures generated from case-study and ablation outputs |
| `paper_artifacts/grid_search_results/` | MLFF-level grid-search JSONL summaries |
| `configs/` | Training, sampling, and relaxation configuration templates |
| `scripts/` | Data preprocessing, sampling, evaluation, VASP preparation, and plotting utilities |

## Excluded From Git

The following files are intentionally excluded because they are large,
license-restricted, generated, or machine-specific:

- OC20-Dense raw files and mapping archives.
- Processed LMDB datasets.
- AdsorbFlow neural-network checkpoints.
- Full GemNet-OC checkpoints when distributed outside this repository.
- Generated trajectories, relaxation folders, grid-search runs, and VASP input
  or output folders.
- Cluster job logs and scheduler files.

## Expected Local Layout

Most scripts assume the following repository-relative layout after data are
downloaded or generated:

| Asset | Suggested path |
|---|---|
| OC20-Dense mapping files | `oc20_dense_mappings/` |
| Conditional training LMDB | `train_allE/` or `train_split/` |
| Validation LMDB | `val_split/` |
| ID evaluation LMDB | `val_nonrelaxed_update/` |
| OOD evaluation LMDB | `valood50_R1I0.1/` |
| AdsorbFlow checkpoints | `checkpoints/` |
| Generated MLFF grid searches | `grid_search_runs/`, `grid_search_runs_ood/` |

Update `src`, `traj_dir`, and checkpoint fields in the YAML configs if your
local data layout differs.

## Dataset Sources

Use the original Open Catalyst and OC20-Dense distribution channels for raw
data. This repository does not redistribute OC20-Dense raw data or VASP
pseudopotentials.

The preprocessing scripts in `scripts/create_lmdbs/` convert raw OC20-Dense
structures and trajectories into LMDB datasets expected by AdsorbFlow.

## Checkpoints

Place AdsorbFlow generator checkpoints under `checkpoints/`. The README command
templates refer to them as:

```text
checkpoints/{adsorbflow_checkpoint}.pt
```

For MLFF relaxation, configure a GemNet-OC checkpoint in
`configs/relaxation/gemnet_oc/gemnet_relax.yml`. VASP verification requires a
separate licensed VASP installation and pseudopotential path.

## Reproducibility Artifacts

The tracked JSONL files under `paper_artifacts/grid_search_results/` contain
compact summaries of the MLFF-level hyperparameter sweeps. Absolute paths from
the original compute environment have been removed or converted to
repository-relative placeholders.
