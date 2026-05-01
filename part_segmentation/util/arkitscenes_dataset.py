import h5py
import numpy as np
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
        
        xyz_blocks, label_blocks, feature_blocks, extra_blocks = [], [], [], []

        with h5py.File(h5_path, "r") as f:
            scene_ids = list(f[split].keys())
            for sid in tqdm(scene_ids, desc=f"Loading {split}"):
                grp = f[split][sid]

                points = np.asarray(grp["points"], dtype=np.float32)
                labels = np.asarray(grp["labels"], dtype=np.int64)

                feature_data = [
                    
                ]

                extra_data = [
                    np.asarray(grp["colors"], dtype=np.float32)
                ]
                
                xyz_block, label_block, feature_block, extra_block = self.data_to_blocks(points=points,
                                                                                            labels=labels,
                                                                                            feature_data=feature_data,
                                                                                            extra_data=extra_data,
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
                
                xyz_blocks.append(xyz_block)
                label_blocks.append(label_block)
                feature_blocks.append(feature_block)
                extra_blocks.append(extra_block)

        self.xyz_blocks = np.concatenate(xyz_blocks, axis=0)
        self.feature_blocks = [np.concatenate([fb[i] for fb in feature_blocks], axis=0) for i in range(len(feature_blocks[0]))]
        self.label_blocks = np.concatenate(label_blocks, axis=0)
        self.extra_blocks = [np.concatenate([eb[i] for eb in extra_blocks], axis=0) for i in range(len(extra_blocks[0]))]

    def __getitem__(self, idx):
        return self.xyz_blocks[idx], [feature[idx] for feature in self.feature_blocks], self.label_blocks[idx], [extra[idx] for extra in self.extra_blocks]