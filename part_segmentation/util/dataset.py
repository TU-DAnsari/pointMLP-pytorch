import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.spatial import KDTree
import open3d as o3d

class BasePointBlockDataset(Dataset):
    def __init__(self):
        self.point_blocks = None
        self.normal_blocks = None
        self.data_blocks = None

    @staticmethod
    def data_to_blocks(points, 
                       point_data, 
                       num_points, 
                       block_size, 
                       stride, 
                       min_points, 
                       pose_noise=False,
                       n_duplication=3,
                       pose_noise_range=0.1,
                       sensor_noise=False,
                       sensor_noise_std=0.1,
                       voxelize=False,
                       voxel_size=0.1,
                       normal_radius=0.1,
                       normalize=True,
                       seed=42
                       ):
        
        rng_sampling = np.random.default_rng(seed=seed)
        rng_pose   = np.random.default_rng(seed=seed+1)
        rng_sensor = np.random.default_rng(seed=seed+2)
        
        
        point_blocks, data_blocks, normal_blocks = [], [], []

        if pose_noise:
            n_points_orig = points.shape[0]
            points_new = np.zeros((int(n_duplication * n_points_orig), 3))
            point_data_new = []
            for data in point_data:
                if len(data.shape) == 1:
                    data_new = np.tile(data, n_duplication)
                else:
                    data_new = np.tile(data, (n_duplication, 1))
                point_data_new.append(data_new)
             
            for i in range(0, n_duplication*n_points_orig, n_points_orig):
                translation = -1*pose_noise_range + (2 * pose_noise_range) * rng_pose.random(3)
                points_new[i:i+n_points_orig] = translation + points

            points = points_new
            point_data = point_data_new

        if sensor_noise:
            points += rng_sensor.normal(0, sensor_noise_std, points.shape)

        if voxelize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd, trace_map, _ = pcd.voxel_down_sample_and_trace(
                voxel_size,
                pcd.get_min_bound(),
                pcd.get_max_bound()
            )

            points_new = np.asarray(pcd.points)
            point_data_new = [[] for _ in point_data]

            for original_indices in trace_map:
                original_indices = original_indices[original_indices != -1]
                for i, data in enumerate(point_data):
                    averaged_data = np.mean(data[original_indices], axis=0)

                    if len(data.shape) == 1:
                        averaged_data = np.round(averaged_data)

                    point_data_new[i].append(averaged_data)

            points = points_new
            point_data = [np.array(data) for data in point_data_new]

        tree = KDTree(points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
        center = pcd.get_center()
        pcd.orient_normals_towards_camera_location(camera_location=center)

        normals = np.asarray(pcd.normals)
        
        mins, maxes = tree.mins, tree.maxes
        x_range = np.arange(mins[0], maxes[0], stride)
        y_range = np.arange(mins[1], maxes[1], stride)
        z_range = np.arange(mins[2], maxes[2], stride)
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
        centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        for center in centers:
            idx = tree.query_ball_point(center, r=block_size / 2)
            if len(idx) < min_points:
                continue

            # Resampling logic
            replace = len(idx) < num_points
            chosen = rng_sampling.choice(len(idx), num_points, replace=replace)
            
            # Extract and Normalize points
            points_in_block = points[idx][chosen]
            normals_in_block = normals[idx][chosen]
            if normalize:
                points_in_block -= points_in_block.mean(axis=0)
                norm = np.max(np.linalg.norm(points_in_block, axis=1))
                if norm > 0: points_in_block /= norm
            
            point_blocks.append(points_in_block)
            normal_blocks.append(normals_in_block)

            # Extract associated data (labels, normals, etc)
            current_data_group = [d[idx][chosen] for d in point_data]
            data_blocks.append(current_data_group)

        if not point_blocks:
            return np.empty((0, num_points, 3)), []

        # Reorganize data_blocks from List[Blocks][Features] to List[Features][Blocks]
        transposed_data = list(zip(*data_blocks))
        return np.stack(point_blocks), np.stack(normal_blocks), [np.stack(d) for d in transposed_data]

    def __len__(self):
        return len(self.point_blocks) if self.point_blocks is not None else 0
        