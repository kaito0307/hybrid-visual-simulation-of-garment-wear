"""Design YAML -> draped simulation mesh (*_sim.obj) via GarmentCode, with optional material rotation."""
import os
import os.path as osp
import shutil
import subprocess

import numpy as np

import paths


def find_sim_mesh(directory):
    meshes = [f for f in os.listdir(directory) if f.endswith('_sim.obj')]
    if len(meshes) != 1:
        raise RuntimeError(f'Expected exactly one *_sim.obj in {directory}, found {meshes}')
    return osp.join(directory, meshes[0])


def rotate_obj_uvs(input_path, output_path, rotation_angle_rad):
    """Rotate the `vt` coordinates of an OBJ. ArcSim uses the UVs as the material frame,
    so this rotates the fabric's warp/weft directions relative to the garment."""
    R = np.array([
        [np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
        [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)]
    ])

    with open(input_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('vt '):
            parts = line.strip().split()
            u, v = float(parts[1]), float(parts[2])
            rotated_uv = R @ np.array([u, v])
            new_lines.append(f'vt {rotated_uv[0]:.6f} {rotated_uv[1]:.6f}\n')
        else:
            new_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(new_lines)


def generate_mesh_from_design(design_file, save_path, meshing_params):
    """
    :param meshing_params: dict with optional keys
        - resolution: GarmentCode mesh resolution scale (default 2.0; the paper setup uses 0.8)
        - material_rotation: rotation of the material frame in degrees (default 0)
    :return: True on success
    """
    resolution = meshing_params.get('resolution', 2.0)
    material_rotation = float(meshing_params.get('material_rotation', 0.0)) * np.pi / 180.0

    design_file = osp.abspath(design_file)
    save_path = osp.abspath(save_path)
    name = osp.basename(design_file)

    cmd = paths.GARMENTCODE_PYTHON + ['generate_geometry.py', f'--design={design_file}',
                                      f'--resolution={resolution}', f'--save_path={save_path}', f'--name={name}']
    if subprocess.run(cmd, cwd=paths.GARMENTCODE).returncode != 0:
        print(f'Failed to generate mesh from design {design_file}.')
        return False

    if material_rotation != 0.0:
        draped_mesh = find_sim_mesh(save_path)
        shutil.copy(draped_mesh, osp.join(save_path, 'original.obj'))
        rotate_obj_uvs(draped_mesh, draped_mesh, material_rotation)

    return True
