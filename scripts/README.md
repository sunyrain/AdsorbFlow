# Scripts

The public scripts are grouped by workflow stage.

| Folder | Purpose |
|---|---|
| `create_lmdbs/` | Convert OC20-Dense structures and trajectories into LMDB datasets |
| `evaluation/` | Run flow sampling, MLFF relaxation, grid search, and SR/anomaly evaluation |
| `case_studies/` | Reproduce CO2RR, NO3RR, OER, and NRR screening examples |
| `cluster_vasp/` | Prepare and analyze paper-style VASP single-point verification |
| `training_utils/` | Dataset splitting and metadata utilities |
| `viz/` | Plotting and trajectory-export utilities |

Development-only diagnostics and historical experiment symlinks are not part of
the public release tree. Use `docs/COMMANDS.md` for the supported command
templates.
