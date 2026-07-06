import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.spatial import KDTree
import open3d as o3d
from .octomap_handler import OctomapHandler

class SceneDataset(Dataset):
    def __init__(self, 
                 h5_path,
                 split="train",
                 scene_ids=[],
                 num_points=1024, 
                 min_points=256,
                 voxel_size=0.1,
                 block_size=5.0, 
                 stride=2.5, 
                 label_remap={},
                 normalize=True,
                 seed=42,
                ):
        
        super().__init__()

        self.point_blocks = []
        self.feature_blocks = []
        self.label_blocks = []

        with h5py.File(h5_path, "r") as f:
            if not scene_ids:
                scene_ids = list(f[split].keys())
            for sid in tqdm(scene_ids, desc=f"Loading {split}"):
                grp = f[split][sid]

                points = np.asarray(grp["points"], dtype=np.float32)
                labels = np.asarray(grp["labels"], dtype=np.int64)

                point_blocks, feature_blocks, label_blocks = self.data_to_blocks(points=points,
                                                                                 labels=labels,
                                                                                 voxel_size=voxel_size,
                                                                                 num_points=num_points,
                                                                                 min_points=min_points,
                                                                                 block_size=block_size,
                                                                                 stride=stride,
                                                                                 normalize=normalize,
                                                                                 seed=seed)
                
                if label_remap:
                    for i in range(len(label_blocks)):
                        for j in range(num_points):
                            label_blocks[i][j] = label_remap[label_blocks[i][j]]
                
                self.point_blocks.append(point_blocks)
                self.feature_blocks.append(feature_blocks)
                self.label_blocks.append(label_blocks)

        self.point_blocks = np.concatenate(self.point_blocks, axis=0)
        self.feature_blocks = np.concatenate(self.feature_blocks, axis=0)
        self.label_blocks = np.concatenate(self.label_blocks, axis=0)

    def data_to_blocks(self, 
                       points,
                       labels,
                       voxel_size,
                       num_points,
                       min_points,
                       block_size,
                       stride,
                       normalize,
                       seed):
        
        
        rng_sampling = np.random.default_rng(seed=seed)

        kdtree = KDTree(points)
        octree_handler = OctomapHandler(voxel_size)
        octree_handler.insert_point_cloud_nodes(points)

        mins, maxes = points.min(axis=0), points.max(axis=0)
        x_range = np.arange(mins[0], maxes[0], stride)
        y_range = np.arange(mins[1], maxes[1], stride)
        xx, yy= np.meshgrid(x_range, y_range)
        centers = np.column_stack([xx.ravel(), yy.ravel()])
        z = points[:, 2]
        z_mean = z.mean()

        z_size = 4 * np.max([z.max() - z_mean, z_mean - z.min()])

        centers = np.concatenate([centers, z_mean * np.ones((centers.shape[0], 1))], axis=1)


        point_blocks = []
        feature_blocks = []
        label_blocks = []

        for center in centers:

            points_in_block, _ = octree_handler.get_bbox_points(middle=center, extent=np.array([block_size, block_size, z_size]))

            if len(points_in_block) < min_points:
                continue

            _, idx = kdtree.query(points_in_block, k=1)

            replace = len(idx) < num_points
            chosen = rng_sampling.choice(len(idx), num_points, replace=replace)

            points_in_block = points[idx][chosen]
            labels_in_block = labels[idx][chosen]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_block)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd_center = pcd.get_center()
            pcd.orient_normals_towards_camera_location(camera_location=pcd_center)

            normals_in_block = np.asarray(pcd.normals)

            if normalize:
                features_in_block = np.concatenate([self.normalize_xyz(points_in_block), normals_in_block], axis=1)
            else:
                features_in_block = np.concatenate([points_in_block, normals_in_block], axis=1)


            point_blocks.append(points_in_block)
            feature_blocks.append(features_in_block)
            label_blocks.append(labels_in_block)

        return point_blocks, feature_blocks, label_blocks
    
    
    @staticmethod
    def normalize_xyz(points):

        points_normalized = points - points.mean(axis=0)
        maxes = points_normalized.max(axis=0)
        mins = points_normalized.min(axis=0)
        points_normalized = (points_normalized - mins) / ((maxes - mins) + 1e-8)

        return points_normalized

    def __len__(self):
        return len(self.point_blocks)

    def __getitem__(self, idx):
        return self.point_blocks[idx], self.feature_blocks[idx], self.label_blocks[idx]