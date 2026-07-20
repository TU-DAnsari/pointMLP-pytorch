import h5py
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset


class MixedOccupancyDataset(Dataset):
    def __init__(self, 
                 h5_path, 
                 split="train", 
                 num_points=1024,
                 seed=42,
                ):
        
        super().__init__()

        rng_sampling = np.random.default_rng(seed=seed)

        with h5py.File(h5_path, "r") as f:
            g = f[split]

            references = np.asarray(g["reference_points"], dtype=np.float32)
            reference_partials = np.asarray(g["reference_partials"], dtype=np.float32)
            others = np.asarray(g["other_points"], dtype=np.float32)
            other_partials = np.asarray(g["other_partials"], dtype=np.float32)
            mixed_points = np.asarray(g["mixed_points"], dtype=np.float32)
            mixed_labels = np.asarray(g["mixed_labels"], dtype=np.float32)

        _, n_pts_reference, _ = reference_partials.shape
        _, n_pts_mixed, _ = mixed_points.shape

        idx_reference = rng_sampling.choice(n_pts_reference, num_points, replace=n_pts_reference < num_points)
        idx_mixed = rng_sampling.choice(n_pts_mixed, num_points, replace=n_pts_mixed < num_points)

        self.references = references[:, idx_reference, :]
        self.reference_partials = reference_partials[:, idx_reference, :]
        self.others = others[:, idx_reference, :]
        self.other_partials = other_partials[:, idx_reference, :]
        self.mixed_points = mixed_points[:, idx_mixed, :]
        self.mixed_labels = mixed_labels[:, idx_mixed]

    def __len__(self):
        return len(self.references)
    
    def __getitem__(self, index):
        return self.references[index], self.reference_partials[index], self.others[index], self.other_partials[index], self.mixed_points[index], self.mixed_labels[index]