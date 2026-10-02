"""Sweep over materials / motions / material rotations (and optionally pants design parameters),
simulating each combination with ArcSim.

Each run lives in <prefix>/<material>-<motion>-<rotation>/ and contains the design, the GarmentCode
outputs, tmp_config.json and sim_results/ (one NNNNN.npz per frame). See compute_wear.py for the metrics.
"""
import argparse
import json
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import yaml

import paths
from generate_mesh import generate_mesh_from_design
from run_arcsim import prepare_arcsim_inputs, simulate

DESIGN_TEMPLATE = osp.join(paths.ROOT, 'templates', 'pants.yaml')
ARCSIM_TEMPLATE = osp.join(paths.ROOT, 'templates', 'human-hires.json')

DESIGN_KEYS = ['length', 'width', 'flare', 'rise']
MESHING_KEYS = ['resolution', 'material_rotation']
ARCSIM_KEYS = ['frame_steps', 'material', 'motion_file', 'slow_motion', 'end_time']


def assemble_params(params):
    """Split a flat parameter dict into design / meshing / arcsim groups."""
    known_params = DESIGN_KEYS + MESHING_KEYS + ARCSIM_KEYS
    for k in params:
        if k not in known_params:
            raise ValueError(f'Unknown parameter: {k}')

    design_params = {k: params[k] for k in DESIGN_KEYS if k in params}
    meshing_params = {k: params[k] for k in MESHING_KEYS if k in params}
    arcsim_params = {k: params[k] for k in ARCSIM_KEYS if k in params}
    return design_params, meshing_params, arcsim_params


def build_arcsim_config(params, working_dir):
    """Generate the draped mesh and assemble the ArcSim config. Returns the config, or None on failure."""
    design_params, meshing_params, arcsim_params = assemble_params(params)
    os.makedirs(working_dir, exist_ok=True)

    with open(DESIGN_TEMPLATE, 'r') as f:
        design_template = yaml.safe_load(f)
    for key in design_params:
        design_template['design']['pants'][key]['v'] = design_params[key]

    design_path = osp.join(working_dir, 'design.yaml')
    with open(design_path, 'w') as f:
        yaml.dump({'design': design_template['design']}, f)

    if not generate_mesh_from_design(design_path, working_dir, meshing_params):
        print(f'Failed to generate mesh for design {design_path}.')
        return None

    with open(ARCSIM_TEMPLATE, 'r') as f:
        config = json.load(f)

    if 'material' in arcsim_params:
        config['cloths'][0]['materials'][0]['data'] = f'$ARCSIM/materials/{arcsim_params.pop("material")}.json'

    if 'motion_file' in arcsim_params:
        motion_token = f'$ARCSIM/human-data/motion/{arcsim_params.pop("motion_file")}.npy'
        config['obstacles'][0]['motion_file'] = motion_token
        n_frames = np.load(motion_token.replace('$ARCSIM', paths.ARCSIM), mmap_mode='r').shape[0]
        if n_frames / paths.MOTION_FPS < config['end_time']:
            config['end_time'] = int(n_frames / paths.MOTION_FPS * 10) / 10
            print(f"Adjusting end_time to {config['end_time']}s based on motion file.")

    if 'slow_motion' in arcsim_params:
        factor = arcsim_params.pop('slow_motion')
        config['end_time'] *= factor
        config['obstacles'][0]['frame_time'] *= factor

    config['extra'] = {'material_rotation': meshing_params.get('material_rotation', 0.0)}
    config.update(arcsim_params)
    return config


def run_one(params, working_dir):
    config = build_arcsim_config(params, working_dir)
    if config is None:
        return False
    if not simulate(working_dir, config):
        print(f'Failed to run simulation in {working_dir}.')
        return False
    return True


def prepare_one(params, working_dir):
    config = build_arcsim_config(params, working_dir)
    if config is None:
        return False
    prepare_arcsim_inputs(working_dir, config)
    return True


def frange(start, stop, num_step):
    step = (stop - start) / (num_step - 1)
    return (start + i * step for i in range(num_step))


def expand_params(values_list):
    """Each config entry is either an explicit list of values or
    ["range", start, stop, n_steps, default(, "int")]; the default is prepended if it isn't the first value."""
    res = {}
    for key, values in values_list.items():
        if values[0] == 'range':
            res[key] = list(frange(values[1], values[2], values[3]))
            if res[key][0] != values[4]:
                res[key] = [values[4]] + res[key]
            if len(values) > 5 and values[5] == 'int':
                res[key] = [int(v) for v in res[key]]
        else:
            res[key] = values
    return res


def generate_grid(values_list):
    """Cartesian product of all values."""
    all_params = [{}]
    for key, values in values_list.items():
        all_params = [dict(p, **{key: v}) for p in all_params for v in values]
    return all_params


def generate_sequential(values_list):
    """Defaults (first value of each key), then vary one key at a time."""
    default_params = {k: v[0] for k, v in values_list.items()}
    all_params = [default_params]
    for key, values in values_list.items():
        for v in values[1:]:
            new_params = deepcopy(default_params)
            new_params[key] = v
            all_params.append(new_params)
    return all_params


def run_name(i, params):
    parts = [params.get('material', f'run_{i:03d}'), params.get('motion_file'), params.get('material_rotation')]
    return '-'.join(str(p) for p in parts if p is not None)


def write_slurm_scripts(prefix, prepared_dirs, args):
    """Write run_arcsim_job.py (one sim) and submit_jobs.py (sbatch per run) into the prefix directory."""
    abs_prefix = osp.abspath(prefix)
    cpus = args.slurm_cpus
    rel_dirs = [osp.relpath(osp.abspath(d), abs_prefix) for d in prepared_dirs]
    cluster_bin = args.cluster_arcsim_bin or osp.join(args.cluster_arcsim, 'build', 'arcsim')

    job_script = osp.join(abs_prefix, 'run_arcsim_job.py')
    job_lines = [
        '#!/usr/bin/env python3',
        'import subprocess, os, sys',
        '',
        'working_dir = sys.argv[1]',
        'sim_results = os.path.join(working_dir, "sim_results")',
        '',
        'env = os.environ.copy()',
        f'env["ARCSIM"] = {args.cluster_arcsim!r}',
        f'env["TBB_NUM_THREADS"] = "{cpus}"',
        f'env["EMBREE_NUM_THREADS"] = "{cpus}"',
        f'env["OMP_NUM_THREADS"] = "{cpus}"',
        '',
        'result = subprocess.run(',
        f'    [{cluster_bin!r}, "simulateoffline", "../tmp_config.json", "./"],',
        '    cwd=sim_results, env=env,',
        ')',
        'sys.exit(result.returncode)',
    ]
    with open(job_script, 'w') as f:
        f.write('\n'.join(job_lines) + '\n')
    os.chmod(job_script, 0o755)

    sbatch_opts = ['--ntasks=1', f'--cpus-per-task={cpus}', f'--time={args.slurm_time}']
    if args.slurm_partition:
        sbatch_opts.append(f'--partition={args.slurm_partition}')
    if args.slurm_account:
        sbatch_opts.append(f'--account={args.slurm_account}')

    submit_script = osp.join(abs_prefix, 'submit_jobs.py')
    sub_lines = [
        '#!/usr/bin/env python3',
        'import subprocess, os',
        '',
        'script_dir = os.path.dirname(os.path.abspath(__file__))',
        'log_dir = os.path.join(script_dir, "logs")',
        'os.makedirs(log_dir, exist_ok=True)',
        'job_script = os.path.join(script_dir, "run_arcsim_job.py")',
        '',
        f'rel_dirs = {rel_dirs!r}',
        f'sbatch_opts = {sbatch_opts!r}',
        '',
        'for rel_dir in rel_dirs:',
        '    working_dir = os.path.join(script_dir, rel_dir)',
        '    name = os.path.basename(working_dir)',
        '    log = os.path.join(log_dir, f"{name}_%j.log")',
        '    cmd = ["sbatch"] + sbatch_opts + [f"--output={log}", "--wrap", f"python3 {job_script} {working_dir}"]',
        '    subprocess.run(cmd, check=True)',
        '    print(f"Submitted: {name}")',
    ]
    with open(submit_script, 'w') as f:
        f.write('\n'.join(sub_lines) + '\n')
    os.chmod(submit_script, 0o755)

    print(f'\nSlurm scripts for {len(prepared_dirs)} jobs written to {abs_prefix}.')
    print(f'To submit: python3 {submit_script}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prefix', type=str, required=True, help='Output directory for the sweep')
    parser.add_argument('--config', type=str, default='./configs/materials_ours.json', help='Sweep configuration')
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'sequential'],
                        help='grid: Cartesian product; sequential: vary one key at a time from the defaults')
    parser.add_argument('--workers', type=int, default=4, help='Number of runs processed in parallel')
    parser.add_argument('--slurm', action='store_true',
                        help='Generate meshes and ArcSim inputs locally, then write SLURM scripts instead of simulating')
    parser.add_argument('--cluster-arcsim', type=str, default='', dest='cluster_arcsim',
                        help='ArcSim root on the cluster (required with --slurm / --skip-prepare)')
    parser.add_argument('--cluster-arcsim-bin', type=str, default='', dest='cluster_arcsim_bin',
                        help='ArcSim binary on the cluster (default: <cluster-arcsim>/build/arcsim)')
    parser.add_argument('--slurm-partition', type=str, default='', dest='slurm_partition')
    parser.add_argument('--slurm-account', type=str, default='', dest='slurm_account')
    parser.add_argument('--slurm-time', type=str, default='02:00:00', dest='slurm_time')
    parser.add_argument('--slurm-cpus', type=int, default=16, dest='slurm_cpus')
    parser.add_argument('--skip-prepare', action='store_true', dest='skip_prepare',
                        help='Only regenerate SLURM scripts for already-prepared runs in --prefix')
    args = parser.parse_args()

    if (args.slurm or args.skip_prepare) and not args.cluster_arcsim:
        parser.error('--cluster-arcsim is required with --slurm or --skip-prepare')

    prefix = args.prefix

    if args.skip_prepare:
        prepared_dirs = [
            osp.join(prefix, d) for d in sorted(os.listdir(prefix))
            if osp.exists(osp.join(prefix, d, 'tmp_config.json'))
            and osp.exists(osp.join(prefix, d, 'sim_results', 'cloth0.obj'))
        ]
        print(f'Found {len(prepared_dirs)} prepared dirs in {prefix}.')
        write_slurm_scripts(prefix, prepared_dirs, args)
        return

    with open(args.config, 'r') as f:
        sweep_config = json.load(f)

    os.makedirs(prefix, exist_ok=True)
    with open(osp.join(prefix, 'massive_conf.json'), 'w') as f:
        json.dump(sweep_config, f, indent=4)

    values = expand_params(sweep_config)
    all_params = generate_sequential(values) if args.mode == 'sequential' else generate_grid(values)
    print(f'{len(all_params)} runs')

    job = prepare_one if args.slurm else run_one
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, params in enumerate(all_params):
            save_path = osp.join(prefix, run_name(i, params))
            futures[save_path] = executor.submit(job, params, save_path)
        results = {path: future.result() for path, future in futures.items()}

    failed = [p for p, ok in results.items() if not ok]
    if failed:
        print(f'{len(failed)} runs failed:', *failed, sep='\n  ')

    if args.slurm:
        prepared = [p for p, ok in results.items() if ok]
        print(f'\nPrepared {len(prepared)}/{len(all_params)} jobs.')
        write_slurm_scripts(prefix, prepared, args)


if __name__ == '__main__':
    main()
