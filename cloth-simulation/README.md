# Cloth Simulation

This folder contains the cloth simulation of "Hybrid Visual Simulation of Garment Wear" (SCF '26). It simulates pants on an animated human body and measures the frictional work between body and cloth at every cloth vertex. The pipeline is:

1. A parametric pants design (`templates/pants.yaml`) goes through **GarmentCode**, which drapes it on an SMPL body and produces a simulation mesh. The fabric can optionally be rotated relative to the garment by rotating the mesh UVs.
2. A **modified ARCSim** simulates the garment on a moving body (jumping or running). Its collision handling records the normal contact force and the tangential relative velocity at each cloth vertex, and the frictional work is integrated over time and written out every frame.
3. `compute_wear.py` extracts the per-vertex wear map of each run (`wear_map.npy`) and summarizes it as wear metrics.

For texture synthesis, the wear maps are quantized into 10 wear levels and rendered as 2000 × 4000 px images in the garment's UV space. The images used in the paper are available in [`../texture-synthesis/data/colormaps_each_fabrics/`](../texture-synthesis/data/colormaps_each_fabrics).

## Setup

All commands below are run from `cloth-simulation/`.

1. **Build ARCSim.** See `arcsim/MODIFICATIONS.md`.
2. **GarmentCode environment:**
   ```bash
   conda env create -f envs/pygarment.yml
   cp garment-code/system.template.json garment-code/system.json
   ```
   Draping uses GarmentCode's Warp-based simulator. Install [NvidiaWarp-GarmentCode](https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode) into the `pygarment` environment. It needs a CUDA GPU.
3. **Driver environment:** `pip install -r requirements.txt` (Python ≥ 3.10).
4. **Body motions.** They are derived from AMASS and SMPL, so they are not included. Generate `jump-hires.npy`, `running.npy` and `A-pose-hires.obj` into `arcsim/human-data/motion/` by following `body_process/README.md`.
5. **Paths.** The defaults point into this folder. Override them with environment variables if needed:

   | Variable | Default | Meaning |
   |---|---|---|
   | `ARCSIM` | `./arcsim` | ARCSim root; materials and motions are resolved under it |
   | `ARCSIM_BIN` | `$ARCSIM/build/arcsim` | ARCSim executable |
   | `GARMENTCODE` | `./garment-code` | GarmentCode root |
   | `GARMENTCODE_PYTHON` | `conda run -n pygarment python` | Command that runs Python in the GarmentCode environment |

## Usage

```bash
# Simulate all combinations: 13 materials x {jump-hires, running} x {0, 90} deg material rotation
python run_sweep.py --prefix runs/ours --config configs/materials_ours.json

# Wear map of every run -> runs/ours/<run>/wear_map.npy, and wear metrics -> runs/ours/wear.csv
python compute_wear.py --prefix runs/ours

# Optional: render the first and last frames (quickpass) or a video (render). Needs an X display.
python render.py --prefix runs/ours --mode quickpass
```

### Sweep configuration

A sweep config maps parameter names to lists of values, and runs are the Cartesian product of all lists. Numeric ranges can also be written as `["range", start, stop, n_steps, default]`. The recognized keys are:
- **material:** `material`, the name of a JSON file in `$ARCSIM/materials/`.
- **motion:** `motion_file`, the name of a `.npy` file in `$ARCSIM/human-data/motion/`.
- **meshing:** `material_rotation` in degrees and `resolution`, the GarmentCode mesh resolution; results used 0.8.
- **simulation:** `frame_steps`, `end_time` and `slow_motion`.
- **pants design:** `length`, `width`, `flare` and `rise`.

### Materials

`arcsim/materials/ours-*.json` are the ARCSim material parameters we estimated for the fabrics in [`../wear-image-dataset/`](../wear-image-dataset). Each file is named after the fabric's fiber and weave; for example, `ours-cotton-plain` is broadcloth and `ours-polyester-plain` is taffeta.

The `jumping` motion in the paper corresponds to the `jump-hires` motion here (AMASS CMU 01_01), and `running` to AMASS CMU 128_02; see `body_process/README.md`.

### Wear map and metrics

The simulator writes one `sim_results/NNNNN.npz` per frame, 25 frames per second. Its `damage` array holds, per cloth vertex, the lumped rest area and the accumulated body–cloth frictional work. The friction power per step is |normal contact force| × |tangential slip speed|.

`compute_wear.py` subtracts the work accumulated up to frame 26, which is the end of the 1 s transition from the draping pose into the motion. The resulting per-vertex frictional work is the **wear map**. It is saved as `<run>/wear_map.npy`, one value per vertex of `<run>/sim_results/cloth0.obj`; the mesh's UVs map it to the garment's sewing pattern. `compute_wear.py` also reports:
- **total:** summed work
- **mean:** mean work per unit area
- **max:** maximum work per unit area

The full column layout of `damage` is documented in `compute_wear.py`.

## License

- Our own code in this folder is released under the MIT license (see `LICENSE`). The exceptions are `arcsim/` and `garment-code/`, which keep their original licenses as described below. `body_process/smpl.py` is adapted from [CalciferZh/SMPL](https://github.com/CalciferZh/SMPL) and keeps its original MIT notice.
- `arcsim/` is a modified version of ARCSim and remains under the ARCSim license (`arcsim/LICENSE`): educational, research and not-for-profit use only. Publications that use it must cite Narain et al. 2012 and 2013. The vendored libraries in `arcsim/lib/` keep their own licenses: ALGLIB is GPL-2.0+, libigl is MPL-2.0, cnpy is MIT.
- `garment-code/` is a modified copy of GarmentCode by Maria Korosteleva and remains under its MIT license (`garment-code/LICENSE`); see `garment-code/MODIFICATIONS.md`.
- AMASS and SMPL data are not distributed; their own licenses apply.

## Acknowledgements

- [ARCSim](http://graphics.berkeley.edu/resources/ARCSim/) by Rahul Narain, Armin Samii, Tobias Pfaff and James O'Brien.
- Wajov's [CMake port of ARCSim 0.3.1](https://github.com/Wajov/arcsim-0.3.1). It replaced the Makefile with CMake and TAUCS/LAPACK with Eigen, and our modified simulator builds on it.
- [GarmentCode](https://github.com/maria-korosteleva/GarmentCode) by Maria Korosteleva and contributors, which generates and drapes the sewing patterns. If you use this code, please also cite the GarmentCode papers listed in `garment-code/ReadMe.md`. Our changes are listed in `garment-code/MODIFICATIONS.md`.
- [CalciferZh/SMPL](https://github.com/CalciferZh/SMPL) for the PyTorch SMPL layer.
