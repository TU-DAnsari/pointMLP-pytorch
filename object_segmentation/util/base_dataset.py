import h5py
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import open3d as o3d

class BaseDataSet(Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def calc_normals(points):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        normals = np.asarray(pcd.normals)

        return normals

    @staticmethod
    def normlize_unit_sphere(points):
        points_normalized = points - points.mean(axis=0)
        scale = np.linalg.norm(points_normalized, axis=1).max()
        points_normalized = points_normalized / (scale + 1e-8)
        return points_normalized

    @staticmethod
    def normalize_xyz(points):
        points_normalized = points - points.mean(axis=0)
        maxes = points_normalized.max(axis=0)
        mins = points_normalized.min(axis=0)
        points_normalized = (points_normalized - mins) / (maxes - mins + 1e-5)
        return points_normalized
