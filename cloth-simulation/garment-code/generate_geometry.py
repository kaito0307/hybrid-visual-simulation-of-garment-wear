import os
import os.path as osp
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from assets.garment_programs.meta_garment import MetaGarment
from pygarment.meshgen.boxmeshgen import BoxMesh
from pygarment.meshgen.simulation import run_sim
import pygarment.data_config as data_config
from pygarment.meshgen.sim_config import PathCofig
from pygarment.data_config import Properties

from assets.bodies.body_params import BodyParameters
import yaml


def get_command_args():
    """command line arguments to control the run"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pattern_spec', '-p', 
        help='pattern specification JSON file. File name should end with "_specification.json"', 
        type=str, 
        default='./assets/Patterns/shirt_mean_specification.json')
    parser.add_argument(
        '--sim_config', '-s', 
        help='Path to simulation config', 
        type=str, 
        default='./assets/Sim_props/default_sim_props.yaml')

    args = parser.parse_args()
    print('Commandline arguments: ', args)

    return args


def prepare_body():
    bodies_measurements = {
        # Our model
        'neutral': './assets/bodies/mean_all.yaml',
        'mean_female': './assets/bodies/mean_female.yaml',
        'mean_male': './assets/bodies/mean_male.yaml',

        # SMPL
        'f_smpl': './assets/bodies/f_smpl_average_A40.yaml',
        'm_smpl': './assets/bodies/m_smpl_average_A40.yaml'
    }
    body_to_use = 'f_smpl'  # CHANGE HERE to use different set of body measurements

    body = BodyParameters(bodies_measurements[body_to_use])
    return body


def prepare_design():
    design_files = {
        # 't-shirt': './assets/design_params/t-shirt.yaml',
        'pants': './assets/design_params/pants.yaml',
        # Add paths HERE to load other parameters
    }

    return yaml.safe_load(design_files['pants'])['design']


def generate_stepped_design_sequences():
    template = './assets/design_params/pants.yaml'
    with open(template, 'r') as f:
        template = yaml.safe_load(f)['design']

    to_be_varied = ['length', 'width', 'flare', 'rise']
    sub_key = 'pants'
    full_template = template
    template = full_template[sub_key]

    step_n = 5

    names = []
    res = []
    for v in to_be_varied:
        d1 = template[v]['v'] - template[v]['range'][0]
        d2 = template[v]['range'][1] - template[v]['v']
        name_prefix = v
        if d1 < d2:
            delta = d2 / step_n
            name_prefix += '_plus'
        else:
            delta = -d1 / step_n
            name_prefix += '_minus'

        for s in range(step_n):
            name = name_prefix + '_' + str(s+1)
            design = deepcopy(full_template)
            design[sub_key][v]['v'] = template[v]['v'] + delta * (s + 1)
            res.append(design)
            names.append(name)

    return res, names


def drape_for_path(path, output_path, geometry_resolution_scale=2):
    sim_config = './assets/Sim_props/default_sim_props.yaml'
    props = data_config.Properties(sim_config)
    props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={})
    props.set_section_stats('render', render_time={})

    spec_path = Path(path)
    garment_name, _, _ = spec_path.stem.rpartition('_')  # assuming ending in '_specification'

    sys_props = data_config.Properties('./system.json')

    if len(output_path.split('/')) < 2:
        output_path = './' + output_path

    paths = PathCofig(
        in_element_path=spec_path.parent,
        out_path='/'.join(output_path.split('/')[:-1]),  # remove last part of path
        out_name=output_path.split('/')[-1],  # last part of path
        in_name=garment_name,
        # body_name='mean_all',    # 'f_smpl_average_A40'
        body_name='f_smpl_average_A40',
        smpl_body=True,  # NOTE: depends on chosen body model
        add_timestamp=False
    )
    os.system(f'cp {paths.in_body_obj} {paths.out_el / f'{garment_name}_body.obj'}')

    # Generate and save garment box mesh (if not existent)
    props['sim']['config']['resolution_scale'] = geometry_resolution_scale

    print(f"Generate box mesh of {garment_name} with resolution {props['sim']['config']['resolution_scale']}...")
    print('\nGarment load: ', paths.in_g_spec)

    garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
    garment_box_mesh.load()

    props['render']['config']['uv_texture']['length_preservation'] = True
    garment_box_mesh.serialize(
        paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])

    props.serialize(paths.element_sim_props)

    fail = run_sim(
        garment_box_mesh.name,
        props,
        paths,
        save_v_norms=False,
        store_usd=False,  # NOTE: False for fast simulation!
        optimize_storage=False,  # props['sim']['config']['optimize_storage'],
        verbose=False
    )

    props.serialize(paths.element_sim_props)

    return fail


def generate_from_design(design_file, name, save_path, resolution):
    with open(design_file, 'r') as f:
        design = yaml.safe_load(f)['design']

    body = prepare_body()
    garment = MetaGarment(name, body, design)
    pattern = garment.assembly()

    new_path = Path(save_path)
    os.makedirs(str(new_path), exist_ok=True)

    folder = pattern.serialize(
        new_path,
        tag='',
        to_subfolder=False,
        with_3d=False, with_text=False, view_ids=False,
        with_printable=True
    )

    body.save(folder)

    return drape_for_path(osp.join(folder, pattern.name + '_specification.json'), save_path, resolution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=str, help='Path to design file to generate garment from')
    parser.add_argument('--resolution', type=float, default=2, help='Geometry resolution scale for garment generation. The higher the value, the more simplified the garment mesh will be.')
    parser.add_argument('--save_path', type=str, help='Path to save generated garment')
    parser.add_argument('--name', type=str, help='Name used as prefix for every generated file.')

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    shutil.copy(args.design, osp.join(args.save_path, 'design_params.yaml'))
    fail = generate_from_design(args.design, args.name, args.save_path, args.resolution)
    if fail:
        print("Generation failed.")
        exit(1)
    else:
        exit(0)
