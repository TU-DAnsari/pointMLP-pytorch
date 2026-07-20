import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.spatial import KDTree

class OccupancyDataset(Dataset):
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

            references = np.asarray(g["full_pcd"], dtype=np.float32)
            partials = np.asarray(g["partial_pcd"], dtype=np.float32)
            proxies = np.asarray(g["proxy_points"], dtype=np.float32)
            labels = np.asarray(g["occupancy_gt"], dtype=np.int8)
        
        _, n_pts, _ = partials.shape
        replace = n_pts < num_points

        idx = rng_sampling.choice(n_pts, num_points, replace=replace)

        self.references = references[:, idx, :]
        self.partials = partials[:, idx, :]
        self.proxies  = proxies [:, idx, :]
        self.labels   = labels  [:, idx]
        
    def __len__(self):
        return len(self.references)
    
    def __getitem__(self, index):
        return self.references[index], self.partials[index], self.proxies[index], self.labels[index]
    