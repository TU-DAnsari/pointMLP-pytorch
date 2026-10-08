import h5py
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset

from .base_dataset import BaseDataSet


class MixedOccupancyDataset(BaseDataSet):
    def __init__(self, 
                 h5_path, 
                 split="train", 
                 num_points=1024,
                 seed=42,
                 noise_std=-1.0,
                 zero_ref_prob=0.1,          # share of mixes with NO reference points
                 ref_frac_range=(0.0, 1.0),  # range for the positive reference fraction
                ):
        
        super().__init__()

        self.num_points = num_points
        self.zero_ref_prob = zero_ref_prob
        self.ref_frac_range = ref_frac_range

        rng_sampling = np.random.default_rng(seed=seed)

        with h5py.File(h5_path, "r") as f:
            g = f[split]

            references = np.asarray(g["reference_points"], dtype=np.float32)
            reference_partials = np.asarray(g["reference_partials"], dtype=np.float32)
            others = np.asarray(g["other_points"], dtype=np.float32)
            other_partials = np.asarray(g["other_partials"], dtype=np.float32)

        n_instances, n_pts_reference, _ = reference_partials.shape
        _, n_pts_other, _ = other_partials.shape

        idx_reference = rng_sampling.choice(n_pts_reference, num_points, replace=n_pts_reference < num_points)
        idx_other = rng_sampling.choice(n_pts_other, num_points, replace=n_pts_other < num_points)

        self.references = references[:, idx_reference, :]
        self.reference_partials = reference_partials[:, idx_reference, :]
        self.others = others[:, idx_other, :]
        self.other_partials = other_partials[:, idx_other, :]

        if noise_std > 0.0:
            for data in [self.references, self.reference_partials, self.others, self.other_partials]:
                data += rng_sampling.normal(scale=noise_std, size=data.shape)

        self.ref_fracs = self._fixed_fractions(rng_sampling, n_instances)

        mixed = [self._build_mixed(i, self.ref_fracs[i], rng_sampling) for i in range(n_instances)]
        self.mixed_points = np.stack([m[0] for m in mixed])
        self.mixed_labels = np.stack([m[1] for m in mixed])

    # ------------------------------------------------------------------ #
    def _fixed_fractions(self, rng, n_instances):
        """Deterministic split: exactly round(zero_ref_prob * N) instances (at least 1) get fraction 0."""
        lo, hi = self.ref_frac_range
        fracs = rng.uniform(lo, hi, size=n_instances).astype(np.float32)
        if self.zero_ref_prob > 0:
            n_zero = max(1, int(round(self.zero_ref_prob * n_instances)))
            zero_idx = rng.choice(n_instances, n_zero, replace=False)
            fracs[zero_idx] = 0.0
        return fracs

    def _sample_fraction(self, rng):
        """Per-access fraction for training: 0 with prob zero_ref_prob, else uniform in range."""
        if rng.random() < self.zero_ref_prob:
            return 0.0
        lo, hi = self.ref_frac_range
        return rng.uniform(lo, hi)

    def _build_mixed(self, index, frac, rng):
        n = self.num_points
        n_ref = int(round(frac * n))
        if frac > 0:
            n_ref = max(n_ref, 1)  # keep "positive fraction" truly positive
        n_other = n - n_ref

        ref_idx = rng.choice(self.reference_partials.shape[1], n_ref, replace=False)
        other_idx = rng.choice(self.other_partials.shape[1], n_other, replace=False)

        points = np.concatenate([
            self.reference_partials[index, ref_idx],
            self.other_partials[index, other_idx],
        ], axis=0)
        labels = np.concatenate([
            np.ones(n_ref, dtype=np.float32),
            np.zeros(n_other, dtype=np.float32),
        ])

        perm = rng.permutation(n)
        return points[perm], labels[perm]


    def __len__(self):
        return len(self.references)
    
    def __getitem__(self, index):
        return self.references[index], self.reference_partials[index], self.others[index], self.other_partials[index], self.mixed_points[index], self.mixed_labels[index]