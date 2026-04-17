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

        self.blocks_points = []
        self.blocks_labels = []

        with h5py.File(h5_path, "r") as f:
            scene_ids = list(f[split].keys())
            if scene_id is not None:
                scene_ids = [scene_id]
            for scene_id in tqdm(scene_ids, desc=f"pre-processing {split} split"):
                points = np.asarray(f[split][scene_id]["points"], dtype=np.float32)
                labels = np.asarray(f[split][scene_id]["labels"], dtype=np.int64)
                scene_blocks_pts, scene_blocks_lbs = ARKitScenesDataset.data_to_blocks(points=points,
                                                                            labels=labels,
                                                                            num_points=self.num_points,
                                                                            block_size=self.block_size,
                                                                            stride=stride,
                                                                            min_points=self.min_points,
                                                                            normalize=True)
                self.blocks_points.append(scene_blocks_pts)
                self.blocks_labels.append(scene_blocks_lbs)

        # Flatten all blocks from all scenes into a single list
        self.blocks_points = np.concatenate(self.blocks_points, axis=0)  # (total_blocks, num_points, 3)
        self.blocks_labels = np.concatenate(self.blocks_labels, axis=0)  # (total_blocks, num_points)

    @staticmethod
    def data_to_blocks(points, labels, num_points, block_size, stride, min_points, normalize=True):
        blocks_pts, blocks_lbs = [], []

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
            
            block_points = points[idx]   # (M, 3)
            block_labels = labels[idx]   # (M,)
            
            # subsample or pad to num_points
            if len(idx) >= num_points:
                chosen = np.random.choice(len(idx), num_points, replace=False)
            else:
                chosen = np.random.choice(len(idx), num_points, replace=True)
            
            block_points = block_points[chosen]
            block_labels = block_labels[chosen]
            
            if normalize:
                block_points -= block_points.mean(axis=0)
                block_points /= np.max(np.linalg.norm(block_points, axis=1))
            
            blocks_pts.append(block_points)
            blocks_lbs.append(block_labels)

        return np.stack(blocks_pts), np.stack(blocks_lbs) 

    def __len__(self):
        return len(self.blocks_points)

    def __getitem__(self, idx):
        points = torch.from_numpy(self.blocks_points[idx]).float()  # (num_points, 3)
        labels = torch.from_numpy(self.blocks_labels[idx]).long()   # (num_points,)
        points = points.permute(1, 0)                               # (3, num_points) for PointMLP
        return points, labels