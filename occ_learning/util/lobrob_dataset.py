import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class LobRobDataset(Dataset):
    def __init__(self, 
                 h5_path, 
                 split="train", 
                ):
        
        super().__init__()

        self.all_partial, self.all_proxy, self.all_labels = [], [], []
        
        with h5py.File(h5_path, "r") as f:
            g = f[split]

            self.all_partial = np.asarray(g["partial_pcd"], dtype=np.float32)
            self.all_proxy = np.asarray(g["proxy_points"], dtype=np.float32)
            self.all_labels = np.asarray(g["occupancy_gt"], dtype=np.int8)

    def __len__(self):
        return len(self.all_partial)
    
    def __getitem__(self, index):
        return self.all_partial[index], self.all_proxy[index], self.all_labels[index] 