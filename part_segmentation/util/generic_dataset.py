import h5py
import numpy as np
import torch
from tqdm import tqdm
from .dataset import BasePointBlockDataset

class GenericDataset(BasePointBlockDataset):
    def __init__(self, 
                 points, 
                 point_data,
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
                
        self.point_blocks = pb
        self.normal_blocks = nb
        self.data_blocks = db
        
    def __getitem__(self, idx):
        return self.point_blocks[idx], self.normal_blocks[idx], [self.data_blocks[i][idx] for i in range(len(self.data_blocks))]