import h5py
import numpy as np
import torch
from tqdm import tqdm
from .dataset import BasePointBlockDataset

class ARKitScenesDataset(BasePointBlockDataset):
    def __init__(self, 
                 h5_path, 
                 split="train", 
                 num_points=1024, 
                 block_size=1.0, 
                 stride=0.5, 
                 min_points=256):
        
        super().__init__()
        
        all_pts, all_normals, all_data = [], [], []

        with h5py.File(h5_path, "r") as f:
            scene_ids = list(f[split].keys())
            for sid in tqdm(scene_ids, desc=f"Loading {split}"):
                grp = f[split][sid]
                # Load features
                pts = np.asarray(grp["points"], dtype=np.float32)
                feats = [
                    np.asarray(grp["labels"], dtype=np.int64),
                    np.asarray(grp["normals"], dtype=np.float32),
                    np.asarray(grp["colors"], dtype=np.float32)
                ]
                
                pb, nb, db = self.data_to_blocks(pts, feats, 
                                                 num_points, 
                                                 block_size, 
                                                 stride, 
                                                 min_points)
                all_pts.append(pb)
                all_normals.append(nb)
                all_data.append(db)

        self.point_blocks = np.concatenate(all_pts, axis=0)
        self.data_blocks = [np.concatenate([d[i] for d in all_data], axis=0) for i in range(len(all_data[0]))]

    def __getitem__(self, idx):
        # p = torch.from_numpy(self.point_blocks[idx]).float().permute(1, 0)
        # l = torch.from_numpy(self.data_blocks[0][idx]).long()
        # n = torch.from_numpy(self.data_blocks[1][idx]).float().permute(1, 0)
        # c = torch.from_numpy(self.data_blocks[2][idx]).float().permute(1, 0)

        p = self.point_blocks[idx]
        l = self.data_blocks[0][idx]
        n = self.data_blocks[1][idx]
        c = self.data_blocks[2][idx]
        
        return p, l, n, c