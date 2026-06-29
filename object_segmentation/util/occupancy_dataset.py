import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class OccupancyDataset(Dataset):
    def __init__(self, 
                 h5_path, 
                 split="train", 
                ):
        
        super().__init__()

        self.partials, self.proxies, self.labels = [], [], []
        
        with h5py.File(h5_path, "r") as f:
            g = f[split]

            self.partials = np.asarray(g["partial_pcd"], dtype=np.float32)
            self.proxies = np.asarray(g["proxy_points"], dtype=np.float32)
            self.labels = np.asarray(g["occupancy_gt"], dtype=np.int8)

    def __len__(self):
        return len(self.all_partial)
    
    def __getitem__(self, index):
        return self.partials[index], self.proxies[index], self.labels[index] 