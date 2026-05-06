import h5py
import numpy as np
from tqdm import tqdm
from .dataset import BasePointBlockDataset

class LobRobDataset(BasePointBlockDataset):
    def __init__(self, 
                 h5_path, 
                 primary_input_names,
                 secondary_input_names,
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
        
        point_blocks, primary_blocks, secondary_blocks, label_blocks, extra_blocks = [], [], [], [], []

        with h5py.File(h5_path, "r") as f:
            scene_ids = list(f[split].keys())
            for sid in tqdm(scene_ids, desc=f"Loading {split}"):
                grp = f[split][sid]

                points = np.asarray(grp["points"], dtype=np.float32)
                labels = np.asarray(grp["labels"], dtype=np.int64)
                ranges = self.normalize(data=np.asarray(grp["ranges"]), max_bound=15.0, min_bound=0.0)
                intensities = self.normalize(data=np.asarray(grp["intensities"]), max_bound=255.0, min_bound=0.0)
                angles_of_incidence = np.asarray(grp["angles_of_incidence"])

                feature_map = {
                    "points": points,
                    "ranges": ranges,
                    "intensities": intensities,
                    "angles_of_incidence": angles_of_incidence,
                }

                

                primary_inputs = LobRobDataset.concat_features(primary_input_names, feature_map)

                if len(secondary_input_names) != 0:
                    secondary_inputs = LobRobDataset.concat_features(secondary_input_names, feature_map)
                else:
                    secondary_inputs = np.zeros_like(points)
                extra_data = []
                
                points_block, primary_block, secondary_block, label_block, extra_block = self.data_to_blocks(points,
                                                                                            primary_inputs,
                                                                                            secondary_inputs,
                                                                                            labels=labels,
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
                
                point_blocks.append(points_block)
                primary_blocks.append(primary_block)
                secondary_blocks.append(secondary_block)
                label_blocks.append(label_block)
                extra_blocks.append(extra_block)

        self.point_blocks = np.concatenate(point_blocks, axis=0)
        self.primary_blocks = np.concatenate(primary_blocks, axis=0)
        self.secondary_blocks = np.concatenate(secondary_blocks, axis=0)
        self.label_blocks = np.concatenate(label_blocks, axis=0)
        self.extra_blocks = np.concatenate(extra_blocks, axis=0)

    def __getitem__(self, idx):
        return self.point_blocks[idx], self.primary_blocks[idx], self.secondary_blocks[idx], self.label_blocks[idx], self.extra_blocks[idx]