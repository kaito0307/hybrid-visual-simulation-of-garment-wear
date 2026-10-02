# Body motion preprocessing

The ArcSim obstacle motions (`$ARCSIM/human-data/motion/*.npy` and `A-pose-hires.obj`) come from AMASS and SMPL, whose licenses don't allow redistribution. This folder has the script to regenerate them from your own downloads.

## Required downloads

1. **SMPL body model** (https://smpl.is.tue.mpg.de). Place `SMPL_FEMALE.npz` at `body_process/smpl_models/smpl/SMPL_FEMALE.npz`.
2. **AMASS, CMU subset, SMPL-X G `stageii` format** (https://amass.is.tue.mpg.de). The motions used are:

   | Motion name  | AMASS file                   |
   |--------------|------------------------------|
   | `jump-hires` | `CMU/01_01_stageii.npz`      |
   | `running`    | `CMU/128_02_stageii.npz`     |
   | `walking`    | `CMU/07_02_stageii.npz`      |

`initial_pose.npz` is the SMPL pose GarmentCode uses to drape the pants. The first second of every motion interpolates from this pose into the first mocap frame.

## Generate

```bash
cd body_process
python amass_parser.py --amass ./AMASS/CMU/01_01_stageii.npz --name jump-hires --out_dir ./arcsim_export
python amass_parser.py --amass ./AMASS/CMU/128_02_stageii.npz --name running --out_dir ./arcsim_export

mkdir -p $ARCSIM/human-data/motion
cp arcsim_export/jump-hires.npy arcsim_export/running.npy $ARCSIM/human-data/motion/
cp arcsim_export/mesh.obj $ARCSIM/human-data/motion/A-pose-hires.obj
```

Each `<name>.npy` holds per-frame vertex positions of the once-subdivided SMPL mesh (27554 vertices), sampled at 120 fps. Every motion starts from the same pose, so their `mesh.obj` files are identical.
