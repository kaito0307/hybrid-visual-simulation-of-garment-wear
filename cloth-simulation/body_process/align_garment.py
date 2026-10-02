import igl
import torch
import numpy as np


def get_garment_skinning_weights(v, body_v, body_f, skinning_weight_v):
    skinning_weight_f = skinning_weight_v[body_f].mean(axis=1)
    S, I, C, N = igl.signed_distance(v, body_v, body_f)
    return skinning_weight_f[I]


def deform_garment(v, skinning_weight, T):
    """
    :param v: garment vertices
    :param f: garment faces
    :param skinning_weight: garment skinning weight per face
    :param T: body transformation per joint
    :return:
    """

    batch_num = T.shape[0]
    v = torch.from_numpy(v).float()
    v = v.unsqueeze(0)
    v = torch.cat(
        (v, torch.ones((v.shape[0], v.shape[1], 1), dtype=torch.float)), dim=2
    )
    T = torch.tensordot(T, skinning_weight, dims=([1], [1])).permute(0, 3, 1, 2)
    v = torch.matmul(T, torch.reshape(v, (v.shape[0], -1, 4, 1)))
    v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]
    return v.numpy()


def collision_resolve(v, body_v, body_f, offset=0.005):
    S, I, C, _ = igl.signed_distance(v, body_v, body_f)
    N = igl.per_face_normals(body_v, body_f)
    collision = S < 0
    v[collision] += (offset - S[collision])[:, None] * N[I[collision]]
    return v
