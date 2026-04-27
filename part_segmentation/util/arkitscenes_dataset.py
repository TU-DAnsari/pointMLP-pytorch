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
                 min_points=256,
                 pose_noise=False,
                 n_duplication=3,
                 pose_noise_range=0.1,
                 sensor_noise=False,
                 sensor_noise_std=0.1,
                 voxelize=False,
                 voxel_size=0.1,
                 normal_radius=0.1,
                 normalize=True,
                 seed=42
                ):
        
        super().__init__()
        
        all_pts, all_normals, all_data = [], [], []

        with h5py.File(h5_path, "r") as f:
            scene_ids = list(f[split].keys())
            for sid in tqdm(scene_ids, desc=f"Loading {split}"):
                grp = f[split][sid]
                # Load features
                points = np.asarray(grp["points"], dtype=np.float32)
                point_data = [
                    np.asarray(grp["labels"], dtype=np.int64),
                    np.asarray(grp["colors"], dtype=np.float32)
                ]
                
                pb, nb, db = self.data_to_blocks(points=points,
                                                point_data=point_data,
                                                num_points=num_points,
                                                block_size=block_size,
                                                stride=stride,
                                                min_points=min_points,
                                                pose_noise=pose_noise,
                                                n_duplication=n_duplication,
                                                pose_noise_range=pose_noise_range,
                                                sensor_noise=sensor_noise,
                                                sensor_noise_std=sensor_noise_std,
                                                voxelize=voxelize,
                                                voxel_size=voxel_size,
                                                normal_radius=normal_radius,
                                                normalize=normalize,
                                                seed=seed,
                                                )
                all_pts.append(pb)
                all_normals.append(nb)
                all_data.append(db)

        self.point_blocks = np.concatenate(all_pts, axis=0)
        self.normal_blocks = np.concatenate(all_normals, axis=0)
        self.data_blocks = [np.concatenate([d[i] for d in all_data], axis=0) for i in range(len(all_data[0]))]

    def __getitem__(self, idx):
        return self.point_blocks[idx], self.normal_blocks[idx], [self.data_blocks[i][idx] for i in range(len(self.data_blocks))]