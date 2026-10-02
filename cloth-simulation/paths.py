"""Locations of the external tools. Override with environment variables."""
import os
import os.path as osp
import shlex

ROOT = osp.dirname(osp.abspath(__file__))

# Modified ArcSim checkout. Material JSONs and motion files are resolved as $ARCSIM/materials, $ARCSIM/human-data.
ARCSIM = osp.abspath(os.environ.get('ARCSIM', osp.join(ROOT, 'arcsim')))
ARCSIM_BIN = os.environ.get('ARCSIM_BIN', osp.join(ARCSIM, 'build', 'arcsim'))

# GarmentCode fork used to turn a design YAML into a draped simulation mesh.
GARMENTCODE = osp.abspath(os.environ.get('GARMENTCODE', osp.join(ROOT, 'garment-code')))
# Command prefix that runs Python inside the GarmentCode environment.
GARMENTCODE_PYTHON = shlex.split(os.environ.get('GARMENTCODE_PYTHON', 'conda run -n pygarment python'))

# Motion files are sampled at 120 fps.
MOTION_FPS = 120


def arcsim_env(num_threads=16):
    env = os.environ.copy()
    env['ARCSIM'] = ARCSIM
    for key in ('TBB_NUM_THREADS', 'EMBREE_NUM_THREADS', 'OMP_NUM_THREADS'):
        env.setdefault(key, str(num_threads))
    return env
