import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from scipy.spatial import KDTree


class ARKitScenesDataset(Dataset):
    def __init__(self, h5_path, split="train", num_points=1024, block_size=1.0, stride=0.5, min_points=256, scene_id=None):
        assert split in ("train", "val")
        self.num_points = num_points
        self.block_size = block_size
        self.min_points = min_points

        point_blocks = []
        data_blocks = []

        with h5py.File(h5_path, "r") as f:
            scene_ids = list(f[split].keys())
            if scene_id is not None:
                scene_ids = [scene_id]

            k = 0

            for scene_id in tqdm(scene_ids, desc=f"pre-processing {split} split"):
                points = np.asarray(f[split][scene_id]["points"], dtype=np.float32)
                normals = np.asarray(f[split][scene_id]["normals"], dtype=np.float32)
                colors = np.asarray(f[split][scene_id]["colors"], dtype=np.float32)
                labels = np.asarray(f[split][scene_id]["labels"], dtype=np.int64)

                point_blocks_scene, data_blocks_scene = ARKitScenesDataset.data_to_blocks(points=points,
                                                                            point_data=[labels, normals, colors],
                                                                            num_points=self.num_points,
                                                                            block_size=self.block_size,
                                                                            stride=stride,
                                                                            min_points=self.min_points,
                                                                            normalize=True)
                point_blocks.append(point_blocks_scene)
                for i, data in enumerate(data_blocks_scene):
                    if i >= len(data_blocks):
                        data_blocks.append([])
                    data_blocks[i].append(data)

                if k == 10:
                    break
                k += 1
        
        self.point_blocks = np.concatenate(point_blocks, axis=0)  # (total_blocks, num_points, 3)
        self.data_blocks = [np.concatenate(data_list, axis=0) for data_list in data_blocks]  # (total_blocks, num_points)

    @staticmethod
    def data_to_blocks(points, point_data, num_points, block_size, stride, min_points, normalize=True):
        point_blocks, data_blocks = [], []

        tree = KDTree(points)

        maxes = tree.maxes
        mins = tree.mins

        x_range = np.arange(mins[0], maxes[0], stride)
        y_range = np.arange(mins[1], maxes[1], stride)
        z_range = np.arange(mins[2], maxes[2], stride)
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
        centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        for cx, cy, cz in centers:
            # query all points within block_size radius of this center
            idx = tree.query_ball_point([cx, cy, cz], r=block_size / 2)
            
            if len(idx) < min_points:
                continue
            
            points_in_block = points[idx]   # (M, 3)
            data_in_block = [data[idx] for data in point_data]
            
            if len(idx) >= num_points:
                chosen = np.random.choice(len(idx), num_points, replace=False)
            else:
                chosen = np.random.choice(len(idx), num_points, replace=True)
            
            points_in_block = points_in_block[chosen]
            data_in_block = [data[chosen] for data in data_in_block]
            
            if normalize:
                points_in_block -= points_in_block.mean(axis=0)
                points_in_block /= np.max(np.linalg.norm(points_in_block, axis=1))
            
            point_blocks.append(points_in_block)
            for i, data in enumerate(data_in_block):
                if i >= len(data_blocks):
                    data_blocks.append([])
                data_blocks[i].append(data)

        return np.stack(point_blocks), [np.stack(data_list) for data_list in data_blocks]

    def __len__(self):
        return len(self.point_blocks)

    def __getitem__(self, idx):
        points = torch.from_numpy(self.point_blocks[idx]).float()       # (num_points, 3)
        labels = torch.from_numpy(self.data_blocks[0][idx]).long()      # (num_points,)
        normals = torch.from_numpy(self.data_blocks[1][idx]).float()    # (num_points,)
        colors = torch.from_numpy(self.data_blocks[1][idx]).float()     # (num_points,)

        points = points.permute(1, 0)                                   # (3, num_points)
        normals = normals.permute(1, 0)                                 # (3, num_points)
        colors = colors.permute(1, 0)                                   # (3, num_points)

        return points, labels, normals, colors
    

class GenericDataset(Dataset):
    def __init__(self, points, point_data, num_points=1024, block_size=1.0, stride=0.5, min_points=256, scene_id=None):
        self.num_points = num_points
        self.block_size = block_size
        self.min_points = min_points

        point_blocks = []
        data_blocks = []

        point_blocks_scene, data_blocks_scene = GenericDataset.data_to_blocks(points=points,
                                                                    point_data=point_data,
                                                                    num_points=self.num_points,
                                                                    block_size=self.block_size,
                                                                    stride=stride,
                                                                    min_points=self.min_points,
                                                                    normalize=True)
        
        point_blocks.append(point_blocks_scene)
        for i, data in enumerate(data_blocks_scene):
            if i >= len(data_blocks):
                data_blocks.append([])
            data_blocks[i].append(data)
        
        # Flatten all blocks from all scenes into a single list
        self.point_blocks = np.concatenate(point_blocks, axis=0)  # (total_blocks, num_points, 3)
        self.data_blocks = [np.concatenate(data_list, axis=0) for data_list in data_blocks]  # (total_blocks, num_points)

    @staticmethod
    def data_to_blocks(points, point_data, num_points, block_size, stride, min_points, normalize=True):
        point_blocks, data_blocks = [], []

        tree = KDTree(points)

        maxes = tree.maxes
        mins = tree.mins

        x_range = np.arange(mins[0], maxes[0], stride)
        y_range = np.arange(mins[1], maxes[1], stride)
        z_range = np.arange(mins[2], maxes[2], stride)
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
        centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        for cx, cy, cz in centers:
            # query all points within block_size radius of this center
            idx = tree.query_ball_point([cx, cy, cz], r=block_size / 2)
            
            if len(idx) < min_points:
                continue
            
            points_in_block = points[idx]   # (M, 3)
            data_in_block = [data[idx] for data in point_data]
            
            if len(idx) >= num_points:
                chosen = np.random.choice(len(idx), num_points, replace=False)
            else:
                chosen = np.random.choice(len(idx), num_points, replace=True)
            
            points_in_block = points_in_block[chosen]
            data_in_block = [data[chosen] for data in data_in_block]
            
            if normalize:
                points_in_block -= points_in_block.mean(axis=0)
                points_in_block /= np.max(np.linalg.norm(points_in_block, axis=1))
            
            point_blocks.append(points_in_block)
            for i, data in enumerate(data_in_block):
                if i >= len(data_blocks):
                    data_blocks.append([])
                data_blocks[i].append(data)

        return np.stack(point_blocks), [np.stack(data_list) for data_list in data_blocks]

    def __len__(self):
        return len(self.point_blocks)

    def __getitem__(self, idx):
        points = torch.from_numpy(self.point_blocks[idx]).float()       # (num_points, 3)
        labels = torch.from_numpy(self.data_blocks[0][idx]).long()      # (num_points,)
        normals = torch.from_numpy(self.data_blocks[1][idx]).float()    # (num_points,)
        colors = torch.from_numpy(self.data_blocks[1][idx]).float()     # (num_points,)

        points = points.permute(1, 0)                                   # (3, num_points)
        normals = normals.permute(1, 0)                                 # (3, num_points)
        colors = colors.permute(1, 0)                                   # (3, num_points)

        return points, labels, normals, colors