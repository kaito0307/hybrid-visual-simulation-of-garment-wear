# Modifications to ARCSim 0.3.1

This is a modified version of ARCSim 0.3.1 by Rahul Narain, Armin Samii, Tobias Pfaff and James O'Brien (UC Berkeley). It is distributed under the original ARCSim license in `LICENSE`, which allows educational, research and not-for-profit use only. Code under `src/` keeps its original notices, for example the UNC SELF-CCD notice in `bvh.cpp`/`bvh.hpp`.

It builds on Wajov's CMake port of ARCSim 0.3.1 (https://github.com/Wajov/arcsim-0.3.1). That port replaced TAUCS/LAPACK with Eigen and vendored ALGLIB; see `README.md`.

## Changes by Peizhuo Li

**Frictional wear measurement**
- `constraint.cpp` (`IneqCon::friction`) sorts each contact into body–cloth or cloth–cloth. Per vertex it accumulates:
  - frictional power `|w_i| * f_n * v_t`
  - normal load
- `wear.cpp`/`wear.hpp` integrate the power into work every step, and carry wear data through remeshing.
- `iomeshseq.cpp` writes one `NNNNN.npz` per frame. Each file holds `positions`, a per-vertex `damage` array (see `../compute_wear.py` for the column layout) and per-face `stretching` energy.

**Animated human body obstacle**
- `obstacle.cpp`: `MeshSeqObstacle` reads per-vertex positions from a `.npy` motion (`obstacles[].motion_file`) and interpolates them by `frame_time`.
- Blending with the previous frame is disabled, so the body follows the motion exactly.

**Pants waistband handle**
- `conf.cpp` adds `{"type": "pants_handle"}`. It pins the highest boundary loop of the cloth (the waistband) to barycentric points on the body.
- Supporting handle types live in `handle.cpp`.

**Performance and solvers**
- `embree.cpp`: Embree 4 broad phase for collision, proximity and separation.
- `eigen.cpp`: CHOLMOD supernodal Cholesky with a reused symbolic factorization.
- `main.cpp`: TBB thread control via `TBB_NUM_THREADS`; Embree reads `EMBREE_NUM_THREADS`.
- `main.cpp` prints a one-line ARCSim license notice at startup, as the license requires.
- `magic.hpp`: `add_jitter` defaults to `true`. The seed is fixed, so results are deterministic for a given build.

**Configuration and I/O**
- `$ENV` variables are expanded in config paths, e.g. `$ARCSIM/materials/...`.
- `replay <dir> quickpass|render` renders from the saved `.npz` frames. The GUI uses GLFW instead of GLUT and has extra wear display modes.

**Materials**
- `materials/ours-*.json`: estimated fabric parameters in the standard ARCSim material format.

**Experimental, not used by the wear pipeline**
- `run_equilibrium.*`, `display_equilibrium.*` (quasi-static solver)
- `qp.*`
- `materialplot.*`

## Building

Requirements:
- CMake ≥ 3.20 and a C++20 compiler
- Embree 4 release package, which also provides TBB
- SuiteSparse (CHOLMOD) and Eigen3
- Boost (system, filesystem, thread), jsoncpp, libpng
- GLFW + OpenGL, only needed for `replay`/GUI; pass `-DNO_OPENGL=ON` to leave them out

The vendored libigl downloads its own copy of Eigen at configure time, so configuring needs network access.

```bash
export EMBREE_ROOT=/path/to/embree-4.x.x.x86_64.linux
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
