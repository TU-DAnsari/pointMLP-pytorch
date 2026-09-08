import h5py
import numpy as np
from tqdm import tqdm

from .base_dataset import BaseDataSet


class ShapeNetPartDataset(BaseDataSet):
    def __init__(self, 
                 h5_path,
                 split="train",
                 num_points=1024,
                 seed=42,
                ):
        
        super().__init__()

        rng_sampling = np.random.default_rng(seed=seed)

        self.points = []
        self.features = []
        self.labels = []

        with h5py.File(h5_path) as f:
            points = np.asarray(f[split]["points"], dtype=np.float32)
            normals = np.asarray(f[split]["normals"], dtype=np.float32)
            labels_seg = np.asarray(f[split]["label_seg"], dtype=np.float32)

            n_instances, n_points = points.shape[:2]
            replace = n_points < num_points
            chosen = np.stack([
                rng_sampling.choice(n_points, num_points, replace=replace)
                for _ in range(n_instances)
            ])
            rows = np.arange(n_instances)[:, None]

            sampled_points = points[rows, chosen]
            sampled_normals = normals[rows, chosen]
            sampled_labels = labels_seg[rows, chosen]

            self.points.append(sampled_points)
            self.features.append(np.concatenate([sampled_points, sampled_normals], axis=-1))
            self.labels.append(sampled_labels)

        self.points = np.array(self.points)[0]
        self.features = np.array(self.features)[0]
        self.labels = np.array(self.labels)[0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.points[idx], self.features[idx], self.labels[idx]