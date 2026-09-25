# Hybrid Visual Simulation of Garment Wear (SCF '26)

Kaito Kikuchi, Peizhuo Li, Rin Ishiguro, I-Chao Shen, Olga Sorkine-Hornung, Koya Narumi, Takeo Igarashi

*Proceedings of the 11th ACM Symposium on Computational Fabrication (SCF '26), Tokyo, Japan*

[[Paper (DOI)](https://doi.org/10.1145/3828611.3845642)]

This repository contains the official dataset and source code for "Hybrid Visual Simulation of Garment Wear".

## Overview

Our framework simulates the visual appearance of garment wear by combining physics-based cloth simulation with data-driven texture synthesis. It consists of three components:

1. **Worn fabric dataset**: We built a custom device that abrades fabric samples with sandpaper under controlled conditions and photographs the surface as it wears. Using this device, we captured 12 fabric types at 10 wear levels each.
2. **Cloth simulation**: We run an FEM cloth simulation (based on ARCSim) of a garment on a moving human body. During the simulation, we accumulate the frictional work between body and garment at each vertex, and quantize the result into a 10-level *wear map*.
3. **Texture synthesis**: We synthesize a worn garment texture using a neural cellular automaton (NCA) with a SIREN decoder. The NCA uses the captured images as appearance exemplars and the simulated wear map as a conditioning signal.

## Repository Structure

```
.
├── wear-image-dataset/   # Captured images of worn fabrics (12 fabrics × 10 wear levels)
├── cloth-simulation/     # Cloth simulation that estimates the wear map (coming soon)
└── texture-synthesis/    # NCA-based texture synthesis conditioned on the wear map
```

Each folder is self-contained, with its own dependencies and instructions. The cloth simulation and the texture synthesis need different environments, so please see the README in each folder for setup and usage.

## Wear Image Dataset

`wear-image-dataset/` contains images of 12 fabric types at 10 wear levels each, captured with our custom abrasion device. See [`wear-image-dataset/README.md`](wear-image-dataset/README.md) for the fabric types, capture conditions, and wear levels.

## Cloth Simulation

**Coming soon.** The cloth simulation code will be released in `cloth-simulation/`. It is a modified version of [ARCSim](http://graphics.berkeley.edu/resources/ARCSim/). We modified its collision handling stage to record the normal contact force and the tangential relative velocity at each cloth vertex. From these, we integrate the frictional work over time to produce a wear map.

In the meantime, the wear maps used in the paper are available in [`texture-synthesis/data/colormaps_each_fabrics/`](texture-synthesis/data/colormaps_each_fabrics).

## Texture Synthesis

`texture-synthesis/` contains our NCA-based texture synthesis code. It is built on the NCA framework of [Neural Cellular Automata: From Cells to Pixels](https://github.com/TheDevilWillBeBee/Cells2Pixels) by Pajouheshgar et al. **Our only modification is conditioning the NCA on the wear map.** We concatenate the wear map to the NCA state channels, and train a single model on all 10 wear levels at once. For the original NCA implementation, please refer to the [Cells2Pixels repository](https://github.com/TheDevilWillBeBee/Cells2Pixels). See `texture-synthesis/README.md` for details.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See [`LICENSE`](LICENSE) for details.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{kikuchi2026hybrid,
  title     = {Hybrid Visual Simulation of Garment Wear},
  author    = {Kikuchi, Kaito and Li, Peizhuo and Ishiguro, Rin and Shen, I-Chao and Sorkine-Hornung, Olga and Narumi, Koya and Igarashi, Takeo},
  booktitle = {Proceedings of the 11th ACM Symposium on Computational Fabrication},
  series    = {SCF '26},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3828611.3845642}
}
```

If you use the texture synthesis code, please also cite the original NCA work:

```bibtex
@article{pajouheshgar2026cells2pixels,
  title   = {Neural Cellular Automata: From Cells to Pixels},
  author  = {Pajouheshgar, Ehsan and Xu, Yitao and Abbasi, Ali and Mordvintsev, Alexander and Jakob, Wenzel and S{\"u}sstrunk, Sabine},
  journal = {arXiv preprint arXiv:2506.22899},
  year    = {2026}
}
```

## Acknowledgments

This work was partially supported by JST Adopting Sustainable Partnerships for Innovative Research Ecosystem (ASPIRE) Grant Number JPMJAP2401 and JST FOREST Program Grant Number JPMJFR232V.

Our texture synthesis is based on the publicly available code of [Neural Cellular Automata: From Cells to Pixels](https://github.com/TheDevilWillBeBee/Cells2Pixels). We thank Ehsan Pajouheshgar for the support with the code.
