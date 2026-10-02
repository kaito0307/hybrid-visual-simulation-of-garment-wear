"""Frictional-wear metrics for every run of a sweep.

The modified ArcSim writes a per-vertex `damage` array (N_vertices x 11) into each frame's NNNNN.npz:
    damage[:, 0]  lumped rest area of the vertex (1/3 of each adjacent face's material-space area)
    damage[:, 1]  body-cloth frictional work accumulated since the start of the simulation
    damage[:, 2]  body-cloth frictional power of the last step
    damage[:, 3:5]  the same two quantities for cloth-cloth contact
    damage[:, 5:8], damage[:, 8:11]  body / cloth normal-load vectors
Frictional power per step is |normal contact force| * |tangential slip speed| (see arcsim/src/constraint.cpp,
IneqCon::friction). The friction coefficient only gates it on or off and does not scale it.

Work accumulated before `--baseline-frame` is subtracted. By default that is frame 26, which is after the
~1 s transition from the draping pose into the motion. The metrics are:
    total  sum of frictional work over all vertices
    mean   mean work density (work / area)
    max    max work density

The per-vertex work itself (the wear map) is saved to <run>/wear_map.npy, one float per vertex of
sim_results/cloth0.obj.
"""
import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd


def wear_map(run_dir, baseline_frame=26):
    """Per-vertex body-cloth frictional work of the last frame, minus the work accumulated up to
    `baseline_frame`. Vertices are in the order of sim_results/cloth0.obj.
    :return: (work, rest_area, n_frames), or None if the run has no usable results
    """
    sim_results = osp.join(run_dir, 'sim_results')
    frames = sorted(f for f in os.listdir(sim_results) if f.endswith('.npz')) if osp.isdir(sim_results) else []
    if not frames:
        return None

    damage = np.load(osp.join(sim_results, frames[-1]))['damage']
    w = damage[:, 1].copy()
    if baseline_frame is not None and baseline_frame >= 0:
        if len(frames) <= baseline_frame:
            return None
        w -= np.load(osp.join(sim_results, frames[baseline_frame]))['damage'][:, 1]
    return w, damage[:, 0], len(frames)


def wear_metrics(run_dir, baseline_frame=26):
    res = wear_map(run_dir, baseline_frame)
    if res is None:
        return None
    w, area, n_frames = res
    w_density = w / area
    return {'total': np.sum(w), 'mean': np.mean(w_density), 'max': np.max(w_density), 'n_frames': n_frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prefix', type=str, required=True, help='Sweep directory (as passed to run_sweep.py)')
    parser.add_argument('--baseline-frame', type=int, default=26,
                        help='Subtract work accumulated up to this frame; -1 to disable')
    parser.add_argument('--output', type=str, default=None, help='CSV path (default: <prefix>/wear.csv)')
    parser.add_argument('--no-maps', action='store_true',
                        help='Do not save the per-vertex wear map of each run to <run>/wear_map.npy')
    args = parser.parse_args()

    rows = []
    for run in sorted(os.listdir(args.prefix)):
        if not osp.isdir(osp.join(args.prefix, run, 'sim_results')):
            continue
        run_dir = osp.join(args.prefix, run)
        res = wear_map(run_dir, args.baseline_frame)
        if res is not None and not args.no_maps:
            np.save(osp.join(run_dir, 'wear_map.npy'), res[0])
        metrics = wear_metrics(run_dir, args.baseline_frame)
        if metrics is None:
            print(f'No usable results for {run}')
            metrics = {'total': np.nan, 'mean': np.nan, 'max': np.nan, 'n_frames': 0}
        rows.append({'run': run, **metrics})

    table = pd.DataFrame(rows)
    output = args.output or osp.join(args.prefix, 'wear.csv')
    table.to_csv(output, index=False)
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(table)
    print(f'Saved to {output}')


if __name__ == '__main__':
    main()
