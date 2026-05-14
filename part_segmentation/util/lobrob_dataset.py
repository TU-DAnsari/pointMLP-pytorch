import h5py
import numpy as np
from tqdm import tqdm
from .dataset import BasePointBlockDataset

class LobRobDataset(BasePointBlockDataset):
    def __init__(self, 
                 h5_path, 
                 sampling_input_names,
                 model_input_names,
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
                 seed=42,
                 scene_ids=[]
                ):
        
        super().__init__()
        
        point_blocks, sampling_blocks, model_input_blocks, label_blocks, extra_blocks = [], [], [], [], []

        with h5py.File(h5_path, "r") as f:
            if not scene_ids:
                scene_ids = list(f[split].keys())
            for sid in tqdm(scene_ids, desc=f"Loading {split}"):
                grp = f[split][sid]

                points = np.asarray(grp["points"], dtype=np.float32)
                labels = np.asarray(grp["labels"], dtype=np.int64)

                ranges = np.asarray(grp["ranges"], dtype=np.float32)
                intensities = np.asarray(grp["intensities"], dtype=np.float32)
                angles_of_incidence = np.asarray(grp["angles_of_incidence"], dtype=np.float32)

                # points_norm = points - points.mean(axis=0)
                # norm = np.max(np.linalg.norm(points_norm, axis=1))
                # if norm > 0: 
                #     points_norm /= norm

                # ranges_norm = self.normalize(data=np.asarray(grp["ranges"]), max_bound=ranges.max(), min_bound=0.0)
                # intensities_norm = self.normalize(data=np.asarray(grp["intensities"]), max_bound=255.0, min_bound=0.0)

                feature_map = {
                    "points": points,
                    "ranges": ranges,
                    "intensities": intensities,
                    "angles_of_incidence": angles_of_incidence,
                }

                sampling_inputs = LobRobDataset.concat_features(sampling_input_names, feature_map)

                if len(model_input_names) != 0:
                    model_inputs = LobRobDataset.concat_features(model_input_names, feature_map)
                else:
                    model_inputs = np.zeros_like(points)
                extra_data = []
                
                points_block, sampling_block, model_input_block, label_block, extra_block = self.data_to_blocks(points,
                                                                                            sampling_inputs,
                                                                                            model_inputs,
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
                sampling_blocks.append(sampling_block)
                model_input_blocks.append(model_input_block)
                label_blocks.append(label_block)
                extra_blocks.append(extra_block)

        self.point_blocks = np.concatenate(point_blocks, axis=0)
        self.sampling_blocks = np.concatenate(sampling_blocks, axis=0)
        self.model_input_blocks = np.concatenate(model_input_blocks, axis=0)
        self.label_blocks = np.concatenate(label_blocks, axis=0)
        self.extra_blocks = np.concatenate(extra_blocks, axis=0)

    def __getitem__(self, idx):
        return self.point_blocks[idx], self.sampling_blocks[idx], self.model_input_blocks[idx], self.label_blocks[idx], self.extra_blocks[idx]