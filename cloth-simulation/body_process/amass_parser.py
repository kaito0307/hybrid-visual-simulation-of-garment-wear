import os

import igl
import numpy as np
import torch
import pickle
import trimesh
import os.path as osp
from smpl import SMPL_Layer

from align_garment import get_garment_skinning_weights, deform_garment, collision_resolve


def load_my_smpl_model():
    """
    Load a custom SMPL model.

    Returns:
        smplx.SMPLX: Loaded SMPLX model.
    """
    return SMPL_Layer(gender='female')


def write_vert_pos_pickle(filename, vert_pos, faces):
    if isinstance(vert_pos, torch.Tensor):
        vert_pos = vert_pos.cpu().numpy()
    mesh_sequence_pickle = [{'vertices': vert_pos[i], 'faces': faces} for i in
                            range(vert_pos.shape[0])]
    with open(filename, 'wb') as f:
        pickle.dump(mesh_sequence_pickle, f)


def get_smpl_pose(amass_seq):
    poses = amass_seq['pose_body']
    betas = amass_seq['betas'][None]
    root_orient = amass_seq['root_orient']
    transl = amass_seq['trans']
    pose_padding = np.zeros((poses.shape[0], 6), dtype=np.float32)
    poses = np.concatenate((root_orient, poses, pose_padding), axis=-1)

    poses, transl = coord_transform(poses, transl)

    return poses, betas, transl


def coord_transform(poses, transl):
    T = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    transl = transl @ T.T

    rot_vec = poses[:, :3]
    # Use scipy's rotation matrix conversion
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(rot_vec)
    rot_mat = rot.as_matrix()
    rot_mat = T @ rot_mat
    rot_vec = R.from_matrix(rot_mat).as_rotvec()
    poses[:, :3] = rot_vec

    return poses, transl


def get_amass_mesh_smpl_and_garment(path, v_g, morph_time=1.):
    amass_seq = np.load(path, allow_pickle=True)

    poses, betas, transl = get_smpl_pose(amass_seq)
    betas[:] = 0    # To match the initial draping results

    num_frames = poses.shape[0]

    initial_pose = np.load(osp.join(osp.dirname(osp.abspath(__file__)), 'initial_pose.npz'))
    morph_frames = int(morph_time * amass_seq['mocap_frame_rate'])

    num_frames += morph_frames

    lins = np.linspace(0, 1, morph_frames)
    start_pose = initial_pose['pose']
    end_pose = poses[0]
    morph_poses = start_pose + (end_pose - start_pose) * lins[:, None]
    poses = np.concatenate((morph_poses, poses), axis=0)

    start_transl = initial_pose['transl']
    transl -= transl[0] - start_transl[0]   # align the first frame
    end_transl = transl[0]
    morph_transl = start_transl + (end_transl - start_transl) * lins[:, None]
    transl = np.concatenate((morph_transl, transl), axis=0)

    poses = torch.from_numpy(poses).float()
    betas = torch.from_numpy(betas).float()
    betas = betas.expand(num_frames, -1).clone()
    transl = torch.from_numpy(transl).float()

    model = load_my_smpl_model()

    vertices, _ = model.forward(poses, betas, transl)
    triangles = model.faces
    vertices = vertices.detach().cpu().numpy()

    joint_transform = model.forward_joint_transformation(poses, betas, transl)
    joint_transform = joint_transform @ joint_transform[:1].inverse()
    garment_skinning_weights = get_garment_skinning_weights(v_g, vertices[0], triangles, model.weights)

    v_g_deformed = deform_garment(v_g, garment_skinning_weights, joint_transform)

    for i in range(num_frames):
        v_g_deformed[i] = collision_resolve(v_g_deformed[i], vertices[i], triangles)

    print('frametime: ', 1 / amass_seq['mocap_frame_rate'])
    return vertices, triangles, v_g_deformed


def subdivide_mesh(v, f, n_subdivisions=1):
    """
    Subdivide a mesh using Catmull-Clark subdivision.

    Args:
        v (np.ndarray): Vertices of the mesh.
        f (np.ndarray): Faces of the mesh.
        n_subdivisions (int): Number of times to subdivide the mesh.

    Returns:
        np.ndarray: Subdivided vertices.
        np.ndarray: Subdivided faces.
    """
    for _ in range(n_subdivisions):
        S, nF = igl.loop_matrix(f)
        v = batch_mm(S, v)
        f = nF
    return v, f


def batch_mm(matrix, matrix_batch):
    """
    https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(1, 0, 2).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return (matrix @ vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0, 2).astype(np.float32)


def export_to_arcsim(v, f, dir, name):
    os.makedirs(dir, exist_ok=True)

    mesh = trimesh.Trimesh(vertices=v[0], faces=f, process=False)
    mesh.export(osp.join(dir, 'mesh.obj'))

    np.save(osp.join(dir, f'{name}.npy'), v)

    # with open(osp.join(dir, 'motion.bin'), 'wb') as f:
    #     f.write(struct.pack('I', len(v)))  # number of frames
    #     f.write(struct.pack('I', v[0].shape[0]))  # number of vertices per frame
    #     for frame in v:
    #         f.write(frame.astype(np.float32).tobytes())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert an AMASS (SMPL-X G, stageii) sequence into an ArcSim obstacle motion.')
    parser.add_argument('--amass', type=str, required=True, help='e.g. ./AMASS/CMU/01_01_stageii.npz (jump), ./AMASS/CMU/128_02_stageii.npz (running)')
    parser.add_argument('--name', type=str, required=True, help='Output motion name, e.g. jump-hires or running')
    parser.add_argument('--garment', type=str, default=None, help='Optional draped garment OBJ; if given, also export its LBS-deformed sequence')
    parser.add_argument('--out_dir', type=str, default='./arcsim_export')
    parser.add_argument('--subdivisions', type=int, default=1, help='Loop subdivisions of the SMPL mesh (1 = the "hires" body)')
    args = parser.parse_args()

    if args.garment is not None:
        v_g, f_g = igl.read_triangle_mesh(args.garment)
    else:
        v_g, f_g = np.zeros((1, 3)), None
    v, f, v_g_deformed = get_amass_mesh_smpl_and_garment(args.amass, v_g)
    v, f = subdivide_mesh(v, f, n_subdivisions=args.subdivisions)

    # <out_dir>/<name>.npy: (frames, vertices, 3) at the AMASS frame rate (120 fps for CMU).
    # <out_dir>/mesh.obj: rest mesh (first frame), used as the obstacle mesh (A-pose-hires.obj).
    export_to_arcsim(v, f, args.out_dir, args.name)
    if args.garment is not None:
        np.save(osp.join(args.out_dir, f'{args.name}_garment_lbs.npy'), v_g_deformed)
