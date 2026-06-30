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
                 radius=0.1,
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

        self.partials_noisy, self.labels_noisy = self.__noisy_partials(self.partials,
                                                                       self.proxies,
                                                                       self.labels,
                                                                       radius,
                                                                       seed)

    def __noisy_partials(self, partials_all, proxies_all, labels_all, radius, seed):

        rng_sampling = np.random.default_rng(seed=seed+1)

        n_samples, n_points, _ = partials_all.shape

        partials_noisy = []
        labels_noisy = []

        for i in tqdm(range(n_samples), total=n_samples):
            partial = partials_all[i]
            proxy = proxies_all[i]
            labels = labels_all[i]

            kdtree = KDTree(proxy)
            idx_lists = kdtree.query_ball_point(partial, r=radius)
            idxs = []
            for idx_list in idx_lists:
                idxs.extend(idx_list)

            idxs = np.array(idxs)            
            idxs = np.unique(idxs.reshape(-1))

            replace = len(idxs) < n_points
            idxs = rng_sampling.choice(idxs, n_points, replace=replace)

            partials_noisy.append(proxy[idxs])
            labels_noisy.append(labels[idxs])

        return np.array(partials_noisy), np.array(labels_noisy)
    

    def __len__(self):
        return len(self.references)
    
    def __getitem__(self, index):
        return self.references[index], self.partials_noisy[index], self.labels_noisy[index]