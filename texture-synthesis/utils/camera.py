import json

import numpy as np
import torch
import torch.nn.functional as F


# From Kaolin
def generate_transformation_matrix(camera_position, look_at, camera_up_direction):
    r"""Generate transformation matrix for given camera parameters.

    Formula is :math:`\text{P_cam} = \text{P_world} * \text{transformation_mtx}`,
    with :math:`\text{P_world}` being the points coordinates padded with 1.

    Args:
        camera_position (torch.FloatTensor):
            camera positions of shape :math:`(\text{batch_size}, 3)`,
            it means where your cameras are
        look_at (torch.FloatTensor):
            where the camera is watching, of shape :math:`(\text{batch_size}, 3)`,
        camera_up_direction (torch.FloatTensor):
            camera up directions of shape :math:`(\text{batch_size}, 3)`,
            it means what are your camera up directions, generally [0, 1, 0]

    Returns:
        (torch.FloatTensor):
            The camera transformation matrix of shape :math:`(\text{batch_size}, 4, 3)`.
    """
    z_axis = camera_position - look_at
    z_axis /= z_axis.norm(dim=1, keepdim=True)
    # torch.cross don't support broadcast
    # (https://github.com/pytorch/pytorch/issues/39656)
    if camera_up_direction.shape[0] < z_axis.shape[0]:
        camera_up_direction = camera_up_direction.repeat(z_axis.shape[0], 1)
    elif z_axis.shape[0] < camera_up_direction.shape[0]:
        z_axis = z_axis.repeat(camera_up_direction.shape[0], 1)
    x_axis = torch.cross(camera_up_direction, z_axis, dim=1)
    x_axis /= x_axis.norm(dim=1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    rot_part = torch.stack([x_axis, y_axis, z_axis], dim=2)
    trans_part = -camera_position.unsqueeze(1) @ rot_part
    return torch.cat([rot_part, trans_part], dim=1)


def generate_perspective_projection(fovyangle, ratio=1.0, dtype=torch.float):
    r"""Generate perspective projection matrix for a given camera fovy angle.

    Args:
        fovyangle (float):
            field of view angle of y axis, :math:`tan(\frac{fovy}{2}) = \frac{y}{f}`.
        ratio (float):
            aspect ratio :math:`(\frac{width}{height})`. Default: 1.0.

    Returns:
        (torch.FloatTensor):
            camera projection matrix, of shape :math:`(3, 1)`.
    """
    tanfov = np.tan(fovyangle / 2.0)
    return torch.tensor([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]], dtype=dtype)


class PerspectiveCamera:
    """
    This class represents a batch of cameras in 3D space.
    """

    def __init__(
        self,
        fov=60.0,
        elevation: (list, np.array, torch.Tensor) = [0.0],
        azimuth: (list, np.array, torch.Tensor) = [0.0],
        distance: (list, np.array, torch.Tensor) = [2.0],
        look_at: (list, np.array, torch.Tensor) = [0.0, 0.0, 0.0],
        up_vector: (list, np.array, torch.Tensor) = [0.0, 1.0, 0.0],
        k=1.0,
        height=1024,
        width=1024,
        bounds=(1.0, 8.0),
        device: (str, torch.device) = "cuda:0",
        angle_unit="degrees",
    ):
        """
        :param elevation: Elevation angles of the cameras in degrees (list or numpy array)
        :param azimuth: Azimuth angles of the cameras in degrees (list or numpy array)
        :param distance: Distances of the cameras from the origin (list or numpy array)
        :param look_at: Point the camera is looking at (shared by all cameras)
        :param up_vector: Up vector of the camera (shared by all cameras)
        :param fov: Field of view of the camera in degrees (shared by all cameras)
        :param k: k=1.0 means the camera is perspective. As k increases the camera becomes more orthographic
        :param height: Height of the camera image (Number of pixels)
        :param width: Width of the camera image (Number of pixels)
        :param bounds: Tuple of (min, max) world coordinate defining the cube that contains the grid
        :param device: PyTorch device to store the camera data

        The camera class has the following attributes:
        transform_matrix: torch tensor with shape [num_cameras, 4, 3]
        projection_matrix: torch tensor with shape [3, 1]
        position: torch tensor with shape [num_cameras, 3]
        """

        self.k = k
        with torch.no_grad():
            self.height = height
            self.width = width
            self.fov = fov
            self.bounds = bounds

            if not isinstance(elevation, torch.Tensor):
                # Making an assumption that the type of elevation, azimuth, and distance is the same
                self.elevation = torch.tensor(
                    elevation, dtype=torch.float32, device=device
                )
                self.azimuth = torch.tensor(azimuth, dtype=torch.float32, device=device)
                self.distance = (
                    torch.tensor(distance, dtype=torch.float32, device=device)
                )
                self.look_at = torch.tensor(look_at, dtype=torch.float32, device=device)
                self.up_vector = torch.tensor(
                    up_vector, dtype=torch.float32, device=device
                )
            else:
                self.elevation, self.azimuth, self.distance = (
                    elevation,
                    azimuth,
                    distance,
                )
                self.look_at, self.up_vector = look_at, up_vector

            if self.look_at.ndim == 1:
                self.look_at = self.look_at.unsqueeze(0)

            if self.up_vector.ndim == 1:
                self.up_vector = self.up_vector.unsqueeze(0)

            if angle_unit == "degrees":
                self.elevation = self.elevation * torch.pi / 180.0
                self.azimuth = self.azimuth * torch.pi / 180.0
                self.fov = self.fov * torch.pi / 180.0

            self.projection_matrix = generate_perspective_projection(
                self.fov, dtype=torch.float32
            ).to(device)
            self.projection_matrix[:2] *= self.k

            self._update_camera()

    def _update_camera(self):
        device = self.azimuth.device
        x = self.distance * torch.cos(self.elevation) * torch.cos(self.azimuth)
        y = self.distance * torch.sin(self.elevation)
        z = self.distance * torch.cos(self.elevation) * torch.sin(self.azimuth)
        self.position = torch.stack([x, y, z], dim=1)

        self.transform_matrix = generate_transformation_matrix(
            self.position, self.look_at, self.up_vector
        ).to(device)

    def rotateY(self, angle):
        self.azimuth += angle * torch.pi / 180.0
        self._update_camera()

    @staticmethod
    def generate_random_view_cameras(
        num_views, distance=2.5, max_elevation=180.0, max_azimuth=360.0, **kwargs
    ):
        azimuth = (np.random.rand(num_views) - 0.5) * max_azimuth
        elevation = (
            np.arcsin(np.random.rand(num_views) * 2.0 - 1.0) * max_elevation / np.pi
        )
        distance = np.ones(num_views) * distance

        return PerspectiveCamera(
            elevation=elevation, azimuth=azimuth, distance=distance, **kwargs
        )

    def update_trajectory(self, t, trajectory_type="orbit"):
        """
        Updates camera parameters as a function of time to create a trajectory.

        :param t: Scalar float time parameter
        :param trajectory_type: Type of trajectory ('orbit', 'spiral', 'hover', 'wave')
        """
        if trajectory_type == "orbit":
            self.azimuth = (
                torch.tensor(
                    [t * 45.0 % 360], dtype=torch.float32, device=self.azimuth.device
                )
                * torch.pi
                / 180.0
            )
            self.elevation = (
                torch.tensor([20.0], dtype=torch.float32, device=self.azimuth.device)
                * torch.pi
                / 180.0
            )

        elif trajectory_type == "spiral":
            self.azimuth = (
                torch.tensor(
                    [t * 60.0 % 360], dtype=torch.float32, device=self.azimuth.device
                )
                * torch.pi
                / 180.0
            )
            self.elevation = (
                torch.tensor(
                    [15.0 * np.sin(t)], dtype=torch.float32, device=self.azimuth.device
                )
                * torch.pi
                / 180.0
            )
            self.distance = torch.tensor(
                [2.0 + 0.5 * np.sin(t)], dtype=torch.float32, device=self.azimuth.device
            )

        elif trajectory_type == "hover":
            self.azimuth = (
                torch.tensor([30.0], dtype=torch.float32, device=self.azimuth.device)
                * torch.pi
                / 180.0
            )
            self.elevation = (
                torch.tensor(
                    [15.0 + 10.0 * np.sin(t)],
                    dtype=torch.float32,
                    device=self.azimuth.device,
                )
                * torch.pi
                / 180.0
            )

        elif trajectory_type == "wave":
            self.azimuth = (
                torch.tensor(
                    [t * 90.0 % 360], dtype=torch.float32, device=self.azimuth.device
                )
                * torch.pi
                / 180.0
            )
            self.elevation = (
                torch.tensor(
                    [10.0 * np.sin(t * 0.5)],
                    dtype=torch.float32,
                    device=self.azimuth.device,
                )
                * torch.pi
                / 180.0
            )

        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

        self._update_camera()

    def sample_along_rays(
        self, num_samples: int, perturb: bool = False, voxel_bounds=None
    ):
        """Sample points along camera rays, bounded within a cube defined by voxel_bounds.

        The cube is axis-aligned, from voxel_bounds[0] to voxel_bounds[1] in x, y, and z.
        Rays that do not intersect the cube will have NaN samples.

        Returns
        -------
        sample_points : (B, H, W, N, 3)
        t_vals        : (B, H, W, N, 1)
        ray_origins   : (B, H, W, 3)
        ray_dirs      : (B, H, W, 3)
        """

        near, far = self.bounds
        device = self.position.device
        B = self.position.shape[0]
        H, W = self.height, self.width

        # 1. Pixel grid
        i, j = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        focal = 0.5 * W / np.tan(self.fov * 0.5)

        # Perspective camera directions (normalized)
        x_cam = (j + 0.5 - 0.5 * W) / focal
        y_cam = -(i + 0.5 - 0.5 * H) / focal
        z_cam = -torch.ones_like(x_cam)
        dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        dirs_cam = F.normalize(dirs_cam, dim=-1)

        # 2. Interpolation between perspective and orthographic
        # alpha = 1 means fully perspective, alpha = 0 means fully orthographic
        alpha = 1.0 / self.k

        # For orthographic: all rays are parallel to z_cam direction,
        # but origins shift according to pixel coordinates
        ortho_orig_cam = torch.stack([x_cam, y_cam, torch.zeros_like(x_cam)], dim=-1)
        ortho_dirs_cam = torch.tensor([0, 0, -1.0], device=device).expand_as(ortho_orig_cam)

        # Blend camera-space rays
        blended_dirs_cam = F.normalize(alpha * dirs_cam + (1 - alpha) * ortho_dirs_cam, dim=-1)
        blended_orig_cam = alpha * torch.zeros_like(ortho_orig_cam) + (1 - alpha) * ortho_orig_cam

        # 3. Transform to world space
        R = self.transform_matrix[:, :3, :]
        blended_dirs_cam = blended_dirs_cam.view(1, H, W, 3).expand(B, H, W, 3)
        blended_orig_cam = blended_orig_cam.view(1, H, W, 3).expand(B, H, W, 3)

        ray_dirs = torch.einsum("bij,bhwj->bhwi", R, blended_dirs_cam)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        ray_origins = (
            self.position.view(B, 1, 1, 3)
            + torch.einsum("bij,bhwj->bhwi", R, blended_orig_cam)
        )

        # 4. Ray-cube intersection (same as before)
        if voxel_bounds is None:
            t_start, t_end = near, far
        else:
            bounds_min, bounds_max = voxel_bounds
            bounds_min = bounds_min * 1.05
            bounds_max = bounds_max * 1.05
            bounds_min = torch.tensor(bounds_min, device=device)
            bounds_max = torch.tensor(bounds_max, device=device)

            inv_dir = 1.0 / (ray_dirs + 1e-9)
            t_min = (bounds_min - ray_origins) * inv_dir
            t_max = (bounds_max - ray_origins) * inv_dir

            t1 = torch.minimum(t_min, t_max)
            t2 = torch.maximum(t_min, t_max)
            t_near_cube, _ = torch.max(t1, dim=-1, keepdim=True)
            t_far_cube, _ = torch.min(t2, dim=-1, keepdim=True)
            hit_mask = (t_far_cube > t_near_cube) & (t_far_cube > 0)

            t_start = torch.clamp(t_near_cube, min=near, max=far)
            t_end = torch.clamp(t_far_cube, min=near, max=far)
            t_start[~hit_mask] = near
            t_end[~hit_mask] = far

        # 5. Sample along each ray
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        t_vals = t_vals.view(1, 1, 1, num_samples, 1)
        t_vals = t_start.unsqueeze(-2) + (t_end - t_start).unsqueeze(-2) * t_vals

        if perturb and num_samples > 1:
            mids = 0.5 * (t_vals[..., 1:, :] + t_vals[..., :-1, :])
            upper = torch.cat([mids, t_vals[..., -1:, :]], dim=-2)
            lower = torch.cat([t_vals[..., :1, :], mids], dim=-2)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand

        # 6. Compute final 3D sample positions
        sample_points = ray_origins.unsqueeze(-2) + ray_dirs.unsqueeze(-2) * t_vals

        return sample_points, t_vals, ray_origins, ray_dirs

    @staticmethod
    def from_json(json_path: str, device: str = "cuda:0", return_keys=False):
        """Load cameras from a NeRF-style JSON and return a PerspectiveCamera instance."""
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        keys = sorted(meta.keys())
        R_all, t_all = [], []
        for k in keys:
            entry = meta[k]
            R_all.append(np.array(entry["extrinsic"]["rotation"], dtype=np.float32))
            t_all.append(
                np.array(entry["extrinsic"]["translation"], dtype=np.float32).flatten()
            )

        entry0 = meta[keys[0]]
        intrinsics = entry0["intrinsic"]
        H, W = intrinsics["height"], intrinsics["width"]
        focal_px = intrinsics["focal"]
        near, far = map(float, intrinsics["bounds"])

        fov = np.degrees(2 * np.arctan(0.5 * W / focal_px))

        positions = np.stack(t_all)
        dists = np.linalg.norm(positions, axis=1)
        elevs = np.degrees(np.arcsin(positions[:, 2] / dists))
        azims = np.degrees(np.arctan2(positions[:, 0], positions[:, 1]))

        camera = PerspectiveCamera(
            fov=fov,
            elevation=elevs,
            azimuth=azims,
            distance=dists,
            look_at=[0, 0, 0],
            up_vector=[0, 1, 0],
            height=H,
            width=W,
            bounds=(near, far),
            k=1.0,
            device=device,
            angle_unit="degrees",
        )

        print("Successfully loaded cameras from JSON file.")
        if return_keys:
            return camera, keys

        return camera

    def sample_batch(self, batch_size, indices=None):
        """
        Sample a random batch of cameras from the existing camera set.
        """
        N = self.transform_matrix.shape[0]
        if batch_size > N:
            raise ValueError(
                f"Batch size {batch_size} exceeds the number of cameras {N}."
            )

        if indices is None:
            indices = np.random.choice(N, batch_size, replace=False)
        else:
            batch_size = len(indices)

        if self.up_vector.shape[0] == 1:
            up_vector = self.up_vector
            look_at = self.look_at
        else:
            up_vector = self.up_vector[indices]
            look_at = self.look_at[indices]

        return PerspectiveCamera(
            fov=self.fov,
            elevation=self.elevation[indices],
            azimuth=self.azimuth[indices],
            distance=self.distance[indices],
            look_at=look_at,
            up_vector=up_vector,
            k=self.k,
            height=self.height,
            width=self.width,
            bounds=self.bounds,
            device=self.transform_matrix.device,
            angle_unit="radians",
        )

    def __repr__(self):
        return (
            f"Camera("
            f"\n\tNumber of Cameras = {self.transform_matrix.shape[0]},"
            f"\n\tElevation = {self.elevation * 180.0 / torch.pi},"
            f"\n\tAzimuth = {self.azimuth * 180.0 / torch.pi},"
            f"\n\tDistance = {self.distance},"
            f"\n\tField of View = {self.fov * 180.0 / torch.pi},"
            f"\n\tLook At = {self.look_at},"
            f"\n\tUp Vector = {self.up_vector},"
            f"\n)"
        )


if __name__ == "__main__":
    with torch.no_grad():
        # cam = PerspectiveCamera.generate_random_view_cameras(2, distance=2.0)
        alternative_device = "mps" if torch.backends.mps.is_available() else "cpu"
        alternative_device = "cpu"
        # device = torch.device("cuda" if torch.cuda.is_available() else alternative_device)
        # cam = PerspectiveCamera.from_json("../data/radiance_fields/chair/test_camera_params.json", device=device)
        # batch_cam = cam.sample_batch(2)

        # sample_points, sample_depths, _, _ = batch_cam.sample_along_rays(num_samples=10, perturb=True)
        # print(sample_points.shape, sample_depths.shape)

        cam = PerspectiveCamera.generate_random_view_cameras(
            1, distance=3.0, max_elevation=0.0, max_azimuth=0.0, height=1, width=1,
            device=alternative_device,
        )
        sample_points, sample_depths, ray_origins, ray_directions = cam.sample_along_rays(num_samples=5, perturb=False, voxel_bounds=[-100.0, 100.0])
        print("Sample Points:\n", sample_points)
        # print("Sample Depths:\n", sample_depths)
        # print("Ray Origins:\n", ray_origins)
        # print("Ray Directions:\n", ray_directions)
