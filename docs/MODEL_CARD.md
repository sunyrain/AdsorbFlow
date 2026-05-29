# Model Card

## Model Summary

AdsorbFlow is an energy-conditioned flow-matching model for adsorbate placement
on catalytic surfaces. It generates adsorbate poses through short ODE
integration over adsorbate translation and rotation, then relies on MLFF
relaxation and optional DFT verification for ranking and validation.

## Intended Use

AdsorbFlow is intended for computational heterogeneous-catalysis workflows that
need plausible low-energy adsorption geometries before calculating adsorption
energies, reaction descriptors, microkinetic inputs, or catalyst-screening
features.

It is best used as:

- A candidate adsorption-geometry generator.
- A replacement for slow stochastic placement sampling in screening workflows.
- An upstream module before MLFF relaxation and DFT verification.

## Not Intended For

AdsorbFlow is not a direct substitute for:

- DFT relaxation or single-point verification where high-fidelity energies are
  required.
- Microkinetic solvers, reactor models, or process-system optimization tools.
- Experimental catalyst validation.

## Inputs

The model expects adsorbate-slab structures represented with atom positions,
atomic numbers, periodic cell information, tags, and optional conditioning
energy fields prepared in an OCP/OC20-Dense-style LMDB.

## Outputs

The sampler produces candidate adsorbate placements. Downstream scripts relax
these candidates with GemNet-OC or compatible MLFF models, flag anomalous
relaxations, and rank candidates by relaxed energy.

## Evaluation

The reported evaluation uses OC20-Dense ID and OOD systems. For each system and
each SR@k level, candidates are MLFF-relaxed, filtered for anomalies, ranked by
relaxed energy, and selected candidates are verified by VASP DFT single-point
calculations.

## Limitations

- Performance depends on the training distribution and the quality of the MLFF
  relaxer used for ranking.
- Adsorbate placement is only one part of a full reaction or process-design
  workflow.
- DFT verification remains necessary for final high-fidelity adsorption-energy
  claims.
- Scripts that prepare or analyze VASP calculations require local configuration
  of VASP, pseudopotentials, and scheduler/runtime paths.
- Open Catalyst data and code dependencies are external resources; this
  repository does not redistribute OC20-Dense raw data, VASP pseudopotentials,
  or full relaxation checkpoints.

## Ethical And Scientific Use

AdsorbFlow is a research codebase. Reported catalyst-screening conclusions
should include dataset scope, MLFF relaxation settings, anomaly filtering
criteria, and DFT verification protocol.
