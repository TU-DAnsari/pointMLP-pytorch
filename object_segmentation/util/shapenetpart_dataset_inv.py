import h5py
import numpy as np
from scipy.spatial import cKDTree

from .base_dataset import BaseDataSet


def invariant_features(points, normals, k=16, eps=1e-8):
    """Per-point rotation-invariant features.

    points, normals: (N, 3) arrays. Returns (N, 10) float32:
      0 r            distance to global centroid
      1 mean_d       mean distance to k nearest neighbors (local density/scale)
      2 off          distance from point to its local neighborhood mean
      3 cos_n_radial |cos| between normal and direction to centroid
      4 cos_m_c      cos between direction to local mean and direction to centroid
      5 cos_n_m      |cos| between normal and direction to local mean
      6-9            linearity, planarity, scattering, curvature (local PCA eigenvalues)

    Normal terms use |cos| so estimated normals with ambiguous sign are fine.
    """
    points = points.astype(np.float64)
    c = points.mean(0, keepdims=True)
    _, idx = cKDTree(points).query(points, k=k + 1)
    neigh = points[idx[:, 1:]]                      # (N, k, 3), drop the point itself
    m = neigh.mean(1)                               # local mean (N, 3)

    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)

    n = unit(normals.astype(np.float64))
    to_c = c - points
    to_m = m - points

    r = np.linalg.norm(to_c, axis=-1)
    off = np.linalg.norm(to_m, axis=-1)
    mean_d = np.linalg.norm(neigh - points[:, None], axis=-1).mean(1)
    cos_n_radial = np.abs((n * unit(to_c)).sum(-1))
    cos_m_c = (unit(to_m) * unit(to_c)).sum(-1)
    cos_n_m = np.abs((n * unit(to_m)).sum(-1))

    # Eigenvalues of the local covariance are rotation invariant.
    d = neigh - m[:, None]
    cov = np.einsum("nki,nkj->nij", d, d) / k
    l3, l2, l1 = np.clip(np.linalg.eigvalsh(cov), 0.0, None).T   # l1 >= l2 >= l3
    linearity = (l1 - l2) / (l1 + eps)
    planarity = (l2 - l3) / (l1 + eps)
    scattering = l3 / (l1 + eps)
    curvature = l3 / (l1 + l2 + l3 + eps)

    return np.stack([r, mean_d, off, cos_n_radial, cos_m_c, cos_n_m,
                     linearity, planarity, scattering, curvature], -1).astype(np.float32)


class ShapeNetPartDataset(BaseDataSet):
    def __init__(self,
                 h5_path,
                 split="train",
                 num_points=1024,
                 noise_std=-1.0,
                 seed=42,
                 k=16,
                 augment=False,
                 scale_range=(0.8, 1.25),
                 ):
        super().__init__()
        self.k = k
        self.augment = augment
        self.scale_range = scale_range

        rng_sampling = np.random.default_rng(seed=seed)

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

        self.points = points[rows, chosen]
        self.normals = normals[rows, chosen]
        self.labels = labels_seg[rows, chosen]

        if noise_std > 0.0:
            self.points = self.points + rng_sampling.normal(
                scale=noise_std, size=self.points.shape).astype(np.float32)
            # Noise invalidates the stored normals, so re-estimate them.
            self.normals = np.array([BaseDataSet.calc_normals(p) for p in self.points])

        # Without augmentation the features never change, so compute them once.
        self.features = None
        if not augment:
            self.features = np.array([
                invariant_features(p, n, k=k) for p, n in zip(self.points, self.normals)
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.augment:
            return self.points[idx], self.features[idx], self.labels[idx]

        rng = np.random.default_rng()   # fresh per call, so DataLoader workers differ
        pts = self.points[idx] * rng.uniform(*self.scale_range)   # uniform scale keeps normals valid
        feats = invariant_features(pts, self.normals[idx], k=self.k)
        return pts.astype(np.float32), feats, self.labels[idx]