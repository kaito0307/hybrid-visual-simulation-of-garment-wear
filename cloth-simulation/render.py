"""Render finished runs with ArcSim's replay mode (needs an X display, see $DISPLAY)."""
import argparse
import os
import os.path as osp
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import paths


def render_once(run_dir, mode='quickpass'):
    """
    :param mode:
        - 'quickpass': render the first frame (draping) and the final frame (wear) as PNGs in sim_results/
        - 'render': render every frame and encode <run_dir>/mov2/movie.mp4 with ffmpeg
    :return: list of PNG names (quickpass) or the movie path (render), or None
    """
    sim_results = osp.join(run_dir, 'sim_results')
    if not osp.isdir(sim_results):
        return None

    env = paths.arcsim_env()
    env.setdefault('DISPLAY', ':0')
    subprocess.run([paths.ARCSIM_BIN, 'replay', './', mode], cwd=sim_results, env=env)

    if mode == 'render':
        render_path = osp.join(sim_results, 'render')
        if not osp.isdir(render_path):
            return None
        movie_path = osp.join(run_dir, 'mov2')
        os.makedirs(movie_path, exist_ok=True)
        movie_file = osp.join(movie_path, 'movie.mp4')
        subprocess.run(['ffmpeg', '-y', '-framerate', '25', '-i', osp.join(render_path, '%05d.png'),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', movie_file])
        shutil.rmtree(render_path)
        return movie_file

    pngs = [f for f in os.listdir(sim_results) if f.endswith('.png')]
    pngs.sort(key=lambda x: int(x.split('.')[0]))
    return pngs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=str, required=True, help='Sweep directory (as passed to run_sweep.py)')
    parser.add_argument('--mode', type=str, default='quickpass', choices=['quickpass', 'render'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    prefix = args.prefix
    runs = sorted(d for d in os.listdir(prefix) if osp.isdir(osp.join(prefix, d, 'sim_results')))
    print('Runs found:', runs)

    # Collect all outputs in one folder so they are easy to compare.
    out_dir = osp.join(prefix, 'mov2' if args.mode == 'render' else 'png')
    os.makedirs(out_dir, exist_ok=True)

    def process(run):
        return run, render_once(osp.join(prefix, run), args.mode)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for run, result in executor.map(process, runs):
            if not result:
                continue
            if args.mode == 'render':
                shutil.copy(result, osp.join(out_dir, run + '.mp4'))
            else:
                for png in result:
                    shutil.copy(osp.join(prefix, run, 'sim_results', png),
                                osp.join(out_dir, png.split('.')[0] + f'_{run}.png'))


if __name__ == '__main__':
    main()
