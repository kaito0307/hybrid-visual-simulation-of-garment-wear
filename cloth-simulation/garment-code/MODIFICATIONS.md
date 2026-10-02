# Modifications to GarmentCode

This directory is a modified copy of [GarmentCode](https://github.com/maria-korosteleva/GarmentCode) by Maria Korosteleva and contributors (see `ReadMe.md`). It is distributed under the original MIT license in `LICENSE`. If you use it, please cite the GarmentCode papers listed in `ReadMe.md`.

It is based on upstream commit `0229a8d` (2025-06-19, "Merge pull request #49 from akmorrow13/install"; pygarment 2.0.2). The files from that commit are unchanged, except for the changes by Peizhuo Li listed below.

## Changes

**New files**
- `generate_geometry.py`: command-line tool that turns a design YAML into a draped simulation mesh (`*_sim.obj`). It uses the `f_smpl_average_A40` body and the Warp draping simulator.
- `assets/design_params/pants.yaml`: the pants design used in the release.

**Modified files**
- `assets/garment_programs/pants.py`:
  - Optional knee expansion (`knee_length`, `knee_expansion`, `knee_position`) and knee darts (`knee_dart_{inside,outside}_{width,depth,position}`). All of them default to off.
  - The bottom and inner leg edges are always built as tangent-controlled curves.
- `pygarment/meshgen/garment.py`: the simulation mesh is written in meters, not centimeters.
- `pygarment/meshgen/boxmeshgen.py`, `pygarment/meshgen/render/texture_utils.py`:
  - New `length_preservation` UV option, so UVs are in real length units instead of being normalized. ARCSim uses the UVs as the material rest shape.
  - Fix for the libigl 2.6 `facet_components` API.
- `pygarment/meshgen/simulation.py`: `run_sim` returns whether draping failed; image rendering after draping is disabled.
