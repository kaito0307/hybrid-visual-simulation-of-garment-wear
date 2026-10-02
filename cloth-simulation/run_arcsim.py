"""Run the modified ArcSim on a prepared working directory."""
import json
import os
import os.path as osp
import shutil
import subprocess
import threading
import time

import paths
from generate_mesh import find_sim_mesh


class OutputMonitor(threading.Thread):
    def __init__(self, process):
        super().__init__()
        self.process = process
        self.last_output_time = time.time()
        self.daemon = True

    def run(self):
        for line in self.process.stdout:
            print('[OUTPUT]', line.strip())
            self.last_output_time = time.time()


def run_with_output_timeout(cmd, cwd, grace_period=60):
    """Run `cmd`, killing it if it prints nothing for `grace_period` seconds (ArcSim can stall on bad inputs)."""
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=paths.arcsim_env())

    succ = True
    monitor = OutputMonitor(process)
    monitor.start()

    while True:
        time.sleep(1)
        if process.poll() is not None:
            break
        if time.time() - monitor.last_output_time > grace_period:
            print('No output for too long. Killing process...')
            process.kill()
            succ = False
            break

    process.wait()
    return succ


def prepare_arcsim_inputs(working_dir, config):
    """Copy the draped mesh to sim_results/cloth0.obj and write tmp_config.json."""
    sim_results = osp.join(working_dir, 'sim_results')
    os.makedirs(sim_results, exist_ok=True)
    shutil.copy(find_sim_mesh(working_dir), osp.join(sim_results, 'cloth0.obj'))

    config['cloths'][0]['mesh'] = './cloth0.obj'
    with open(osp.join(working_dir, 'tmp_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    return sim_results


def simulate(working_dir, config, grace_period=120):
    sim_results = prepare_arcsim_inputs(working_dir, config)
    cmd = [paths.ARCSIM_BIN, 'simulateoffline', '../tmp_config.json', './']
    return run_with_output_timeout(cmd, cwd=sim_results, grace_period=grace_period)
