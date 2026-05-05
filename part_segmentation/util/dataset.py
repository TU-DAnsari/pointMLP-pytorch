import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import KDTree
import open3d as o3d
import scipy
from tqdm import tqdm

class BasePointBlockDataset(Dataset):
    def __init__(self):
        self.xyz_blocks = None
        self.feature_blocks = None
        self.label_blocks = None
        self.extra_blocks = None

    @staticmethod
    def data_to_blocks(points, 
                       labels,
                       feature_data, 
                       extra_data,
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
        
        if len(extra_data) == 0:
            extra_data = [np.zeros_like(points)]

        xyz_blocks, label_blocks, feature_blocks, extra_blocks = [], [], [], []

        if pose_noise:
            n_points_orig = points.shape[0]
            points_new = np.zeros((int(n_duplication * n_points_orig), 3))

            feature_data_new = []
            for fd in feature_data:
                if len(fd.shape) == 1:
                    fd_new = np.tile(fd, n_duplication)
                else:
                    fd_new = np.tile(fd, (n_duplication, 1))
                feature_data_new.append(fd_new)

            labels_new = np.tile(labels, n_duplication)

            extra_data_new = []
            for ed in extra_data:
                if len(ed.shape) == 1:
                    ed_new = np.tile(ed, n_duplication)
                else:
                    ed_new = np.tile(ed, (n_duplication, 1))
                extra_data_new.append(ed_new)

            for i in range(0, n_duplication * n_points_orig, n_points_orig):
                translation = -1 * pose_noise_range + (2 * pose_noise_range) * rng_pose.random(3)
                points_new[i:i + n_points_orig] = translation + points

            points = points_new
            feature_data = feature_data_new
            labels = labels_new
            extra_data = extra_data_new

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

            n_voxels = len(trace_map)

            vox_ids_list, pt_ids_list = [], []
            for vox_idx, indices in enumerate(trace_map):
                idx = np.asarray(indices)
                idx = idx[idx != -1]
                vox_ids_list.append(np.full(len(idx), vox_idx, dtype=np.int64))
                pt_ids_list.append(idx)

            vox_ids = np.concatenate(vox_ids_list)   
            pt_ids  = np.concatenate(pt_ids_list)    
            counts  = np.bincount(vox_ids, minlength=n_voxels).astype(np.float64)  

            def grouped_mean(arr):
                if arr.ndim == 1:
                    return np.bincount(vox_ids, weights=arr[pt_ids], minlength=n_voxels) / counts
                out = np.empty((n_voxels, arr.shape[1]), dtype=np.float64)
                for j in range(arr.shape[1]):
                    out[:, j] = np.bincount(vox_ids, weights=arr[pt_ids, j], minlength=n_voxels) / counts
                return out

            feature_data = [grouped_mean(fd) for fd in feature_data]

            unique_labels, label_idx = np.unique(labels, return_inverse=True)
            vote_matrix = np.zeros((n_voxels, len(unique_labels)), dtype=np.int32)
            np.add.at(vote_matrix, (vox_ids, label_idx[pt_ids]), 1)
            labels = unique_labels[np.argmax(vote_matrix, axis=1)]

            extra_data_new = []
            for ed in extra_data:
                ed_mean = grouped_mean(ed)
                if ed.ndim == 1:
                    ed_mean = np.round(ed_mean)
                extra_data_new.append(ed_mean)
            extra_data = extra_data_new

            points = np.asarray(pcd.points)

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

            replace = len(idx) < num_points
            chosen = rng_sampling.choice(len(idx), num_points, replace=replace)

            points_in_block = points[idx][chosen]
            normals_in_block = normals[idx][chosen]

            if normalize:
                points_in_block -= points_in_block.mean(axis=0)
                norm = np.max(np.linalg.norm(points_in_block, axis=1))
                if norm > 0: points_in_block /= norm

            xyz_blocks.append(points_in_block)

            current_data_group = [normals_in_block] + [fd[idx][chosen] for fd in feature_data]
            feature_blocks.append(current_data_group)

            label_blocks.append(labels[idx][chosen])

            extra_blocks.append([ed[idx][chosen] for ed in extra_data])

        if not xyz_blocks:
            raise ValueError("NO DATA")
        
        transposed_features = list(zip(*feature_blocks))   # List[Features][Blocks]
        transposed_extra = list(zip(*extra_blocks))         # List[Features][Blocks]

        return (
            np.stack(xyz_blocks),                           # (B, N, 3)
            np.stack(label_blocks),                         # (B, N)
            [np.stack(f) for f in transposed_features],     # List of (B, N, ...)
            [np.stack(e) for e in transposed_extra],        # List of (B, N, ...)
        )

    def __len__(self):
        return len(self.xyz_blocks) if self.xyz_blocks is not None else 0
        