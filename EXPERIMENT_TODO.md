# Public Release Checklist

This checklist tracks repository-facing work only. Internal cluster notes,
credentials, private paths, and job logs are intentionally excluded from the
public repository.

## Completed

- AdsorbFlow README rewritten for public release.
- Main workflow figure added under `assets/`.
- ArXiv citation added to `README.md`.
- Lightweight case-study CSV summaries added under `examples/`.
- Publication figures added under `figures/`.
- MLFF grid-search JSONL summaries added under `paper_artifacts/`.
- Public command templates collected in `EXPERIMENT_COMMANDS.md`.
- Private datasets, checkpoints, VASP outputs, trajectories, and logs excluded
  from Git tracking.

## Before Reproducing Results

- Download or obtain OC20-Dense data and mapping files.
- Place AdsorbFlow checkpoints under `checkpoints/`.
- Place the GemNet-OC relaxation checkpoint at the path configured in
  `configs/relaxation/gemnet_oc/gemnet_relax.yml`.
- Configure VASP, pseudopotentials, and any cluster scheduler scripts locally.
- Update YAML dataset paths if your local directory layout differs from the
  defaults documented in `DATA.md`.

## Optional Improvements

- Add released checkpoint download links when a permanent archive is available.
- Add a DOI once the code/data archive is deposited.
- Add small smoke tests using synthetic structures that do not require OC20
  data or VASP.
