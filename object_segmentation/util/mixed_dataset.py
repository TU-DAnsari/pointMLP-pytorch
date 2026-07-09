import h5py
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset


class MixedOccupancyDataset(Dataset):
    def __init__(self, h5_path, split="train", num_points=1024,
                 num_ref_points=1024, num_distractors=2, seed=42):
        super().__init__()
        rng = np.random.default_rng(seed=seed)

        with h5py.File(h5_path, "r") as f:
            g = f[split]
            self.class_labels = np.asarray(g["class_label"], dtype=np.int64)
            references = np.asarray(g["full_pcd"], dtype=np.float32)
            partials   = np.asarray(g["partial_pcd"], dtype=np.float32)
            proxies    = np.asarray(g["proxy_points"], dtype=np.float32)
            labels     = np.asarray(g["occupancy_gt"], dtype=np.int8)

        n_inst, n_pts, _ = partials.shape
        idx = rng.choice(n_pts, num_points, replace=n_pts < num_points)
        self.partials = partials[:, idx, :]
        self.proxies  = proxies[:, idx, :]
        self.labels   = labels[:, idx]

        n_ref = references.shape[1]
        ref_idx = rng.choice(n_ref, num_ref_points, replace=n_ref < num_ref_points)
        self.references = references[:, ref_idx, :]

        self.num_points = num_points
        self.num_distractors = num_distractors

        # class -> np.array of instance indices, built once for fast rng.choice
        class_to_indices = defaultdict(list)
        for i, c in enumerate(self.class_labels):
            class_to_indices[int(c)].append(i)
        self.class_to_indices = {c: np.asarray(v) for c, v in class_to_indices.items()}
        self.classes = np.asarray(list(self.class_to_indices.keys()))

    def __len__(self):
        return len(self.partials)

    def __getitem__(self, idx):
        rng = np.random.default_rng()

        target_class = int(self.class_labels[idx])

        # sample distractor classes (excluding target's), then one instance per class
        other_classes = self.classes[self.classes != target_class]
        d_classes = rng.choice(other_classes, size=self.num_distractors, replace=True)
        d_indices = np.array([rng.choice(self.class_to_indices[c]) for c in d_classes])

        all_indices = np.concatenate([[idx], d_indices])          # [1 + n_distractors]

        scene_points = self.partials[all_indices].reshape(-1, 3)   # (1+n_d)*num_points, 3
        match_labels = np.zeros(len(all_indices) * self.num_points, dtype=np.float32)
        match_labels[:self.num_points] = 1.0                       # target block = positives

        perm = rng.permutation(len(scene_points))

        return {
            "reference": self.references[idx],
            "combined": scene_points[perm],
            "labels_match": match_labels[perm],
            "proxy_points": self.proxies[idx],
            "occupancy_labels": self.labels[idx].astype(np.float32),
        }