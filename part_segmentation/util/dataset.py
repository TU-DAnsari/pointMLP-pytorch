import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import KDTree
import open3d as o3d
import scipy
from tqdm import tqdm

class BasePointBlockDataset(Dataset):
    def __init__(self):
        self.point_blocks = None
        self.sampling_blocks = None
        self.model_input_blocks = None
        self.label_blocks = None
        self.extra_blocks = None

    @staticmethod
    def tile_list(data_list, n_duplication):
        tiled = []
        for arr in data_list:
            if len(arr.shape) == 1:
                tiled.append(np.tile(arr, n_duplication))
            else:
                tiled.append(np.tile(arr, (n_duplication, 1)))
        return tiled
    
    @staticmethod
    def concat_features(names, feature_map):
        arrays = []
        for name in names:
            f = feature_map[name]
            if len(f) == 0:
                continue
            arrays.append(f if f.ndim == 2 else f[:, np.newaxis])
        return np.concatenate(arrays, axis=1)

    @staticmethod
    def data_to_blocks(points,
                        sampling_inputs, 
                        model_inputs,
                        labels,
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
        rng_pose = np.random.default_rng(seed=seed+1)
        rng_sensor = np.random.default_rng(seed=seed+2)

        if len(extra_data) == 0:
            extra_data = np.zeros_like(points)

        if pose_noise:
            n_points_orig = points.shape[0]

            points_new = np.tile(points, (n_duplication, 1))
            sampling_inputs_new = np.tile(sampling_inputs, (n_duplication, 1))
            model_inputs_new = np.tile(model_inputs, (n_duplication, 1))
            labels_new = np.tile(labels, n_duplication)
            extra_data_new = np.tile(extra_data, (n_duplication, 1))

            for i in range(0, n_duplication * n_points_orig, n_points_orig):
                translation = -pose_noise_range + (2 * pose_noise_range) * rng_pose.random(sampling_inputs.shape[1])
                points_new[i:i + n_points_orig] = sampling_inputs + translation
                sampling_inputs_new[i:i + n_points_orig] = sampling_inputs + translation

            points = points_new
            sampling_inputs = sampling_inputs_new
            model_inputs = model_inputs_new
            labels = labels_new
            extra_data = extra_data_new

        if sensor_noise:
            noise = rng_sensor.normal(0, sensor_noise_std, sampling_inputs.shape)
            points += noise
            sampling_inputs += noise
        
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
                out = np.empty((n_voxels, arr.shape[1]), dtype=np.float64)
                for j in range(arr.shape[1]):
                    out[:, j] = np.bincount(vox_ids, weights=arr[pt_ids, j], minlength=n_voxels) / counts
                return out

            sampling_inputs = grouped_mean(sampling_inputs)
            model_inputs = grouped_mean(model_inputs)
            extra_data = grouped_mean(extra_data)

            unique_labels, label_idx = np.unique(labels, return_inverse=True)
            vote_matrix = np.zeros((n_voxels, len(unique_labels)), dtype=np.int32)
            np.add.at(vote_matrix, (vox_ids, label_idx[pt_ids]), 1)
            labels = unique_labels[np.argmax(vote_matrix, axis=1)]

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

        point_blocks, sampling_blocks, model_input_blocks, label_blocks, extra_blocks = [], [], [], [], []

        for center in centers:
            idx = tree.query_ball_point(center, r=block_size / 2)
            if len(idx) < min_points:
                continue

            replace = len(idx) < num_points
            chosen = rng_sampling.choice(len(idx), num_points, replace=replace)

            points_in_block = points[idx][chosen]
            sampling_in_block = sampling_inputs[idx][chosen]
            model_input_in_block = model_inputs[idx][chosen]
            labels_in_block = labels[idx][chosen]
            extra_in_block = extra_data[idx][chosen]
            normals_in_block = normals[idx][chosen]

            # if normalize:
            #     sampling_in_block -= sampling_in_block.mean(axis=0)
            #     norm = np.max(np.linalg.norm(sampling_in_block, axis=1))
            #     if norm > 0: sampling_in_block /= norm

            if normalize:
                for i in range(model_input_in_block.shape[1]):
                    model_input_in_block[:, i] = BasePointBlockDataset.normalize(data=model_input_in_block[:, i], 
                                                                                 max_bound=model_input_in_block[:, i].max(), 
                                                                                 min_bound=model_input_in_block[:, i].min())
            else:
                for i in range(model_input_in_block.shape[1]):
                    model_input_in_block[:, i] = BasePointBlockDataset.normalize(data=model_input_in_block[:, i], 
                                                                                 max_bound=1.0, 
                                                                                 min_bound=0.0)

            model_input_in_block = np.concatenate([model_input_in_block, normals_in_block], axis=1)

            point_blocks.append(points_in_block)
            sampling_blocks.append(sampling_in_block)
            model_input_blocks.append(model_input_in_block)
            label_blocks.append(labels_in_block)
            extra_blocks.append(extra_in_block)

        if not point_blocks:
            raise ValueError("NO DATA")

        return (
            np.stack(point_blocks),
            np.stack(sampling_blocks),
            np.stack(model_input_blocks),                          
            np.stack(label_blocks),
            np.stack(extra_blocks),
        )
    
    @staticmethod
    def normalize(data, max_bound, min_bound):
        if data.max() > max_bound:
            max_bound = data.max()
        if data.min() < min_bound:
            min_bound = data.min()
        return (data - min_bound) / (max_bound - min_bound)

    def __len__(self):
        return len(self.point_blocks) if self.point_blocks is not None else 0
        