# Security Policy

Do not commit credentials, API keys, private machine paths, cluster login
details, VASP pseudopotentials, proprietary binaries, or unpublished raw data.

If you discover a leaked secret or private dataset path, remove it from the
current branch immediately and rotate the affected credential outside Git.

Large generated assets such as checkpoints, LMDB datasets, VASP folders,
trajectories, and scheduler logs should remain outside Git and be distributed
through an appropriate data archive when release permissions are clear.
