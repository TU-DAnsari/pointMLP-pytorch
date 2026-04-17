import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ARKitScenesDataset(Dataset):
    def __init__(self, h5_path, split="train", num_points=2048, block_size=1.0, stride=0.5, min_points=100, scene_id=None):
        assert split in ("train", "val")
        self.num_points = num_points
        self.block_size = block_size
        self.min_points = min_points

        # Pre-compute all blocks at init so __len__ is known and dataloading is fast
        self.blocks_points = []
        self.blocks_labels = []

        with h5py.File(h5_path, "r") as f:
            scene_ids = list(f[split].keys())
            if scene_id is not None:
                scene_ids = [scene_id]
            for scene_id in tqdm(scene_ids, desc=f"pre-processing {split} split"):
                points = np.asarray(f[split][scene_id]["points"], dtype=np.float32)
                labels = np.asarray(f[split][scene_id]["labels"], dtype=np.int64)
                scene_blocks_pts, scene_blocks_lbs = self._scene_to_blocks(points, labels, stride, normalize=True)
                self.blocks_points.append(scene_blocks_pts)
                self.blocks_labels.append(scene_blocks_lbs)

        # Flatten all blocks from all scenes into a single list
        self.blocks_points = np.concatenate(self.blocks_points, axis=0)  # (total_blocks, num_points, 3)
        self.blocks_labels = np.concatenate(self.blocks_labels, axis=0)  # (total_blocks, num_points)

    @staticmethod
    def get_blocks(points, block_size=1.0, stride=0.5, min_points=100, num_points=2048):
        dataset = ARKitScenesDataset.__new__(ARKitScenesDataset)  # Create an uninitialized instance
        dataset.block_size = block_size
        dataset.min_points = min_points
        dataset.num_points = num_points
        labels = np.zeros(points.shape[0], dtype=np.int64)  
        point_blocks, _  = dataset._scene_to_blocks(points, labels, stride, normalize=False)
        return point_blocks


    def _scene_to_blocks(self, points, labels, stride, normalize=True):
        blocks_pts, blocks_lbs = [], []

        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()

        x_start = x_min
        while x_start < x_max:
            y_start = y_min
            while y_start < y_max:
                mask = (
                    (points[:, 0] >= x_start) & (points[:, 0] < x_start + self.block_size) &
                    (points[:, 1] >= y_start) & (points[:, 1] < y_start + self.block_size)
                )
                block_pts = points[mask]
                block_lbs = labels[mask]

                if len(block_pts) >= self.min_points:
                    idx = np.random.choice(
                        len(block_pts),
                        self.num_points,
                        replace=len(block_pts) < self.num_points
                    )
                    
                    sampled_pts = block_pts[idx]

                    if normalize:
                        # Normalize block to zero mean
                        sampled_pts = sampled_pts - sampled_pts.mean(axis=0)
                    blocks_pts.append(sampled_pts)
                    blocks_lbs.append(block_lbs[idx])

                y_start += stride
            x_start += stride

        return np.stack(blocks_pts), np.stack(blocks_lbs)

    def __len__(self):
        return len(self.blocks_points)

    def __getitem__(self, idx):
        points = torch.from_numpy(self.blocks_points[idx]).float()  # (num_points, 3)
        labels = torch.from_numpy(self.blocks_labels[idx]).long()   # (num_points,)
        points = points.permute(1, 0)                               # (3, num_points) for PointMLP
        return points, labels