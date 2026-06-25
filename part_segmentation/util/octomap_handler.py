import pyoctomap
import numpy as np


class OctomapHandler:
    def __init__(self, resolution: float):
        self.octree = pyoctomap.ColorOcTree(resolution)

    def insert_point_cloud_nodes(self, points: np.ndarray, data: np.ndarray=None):
        for i in range(points.shape[0]):
            self.octree.updateNode(points[i], True)
            if data is not None:
                self.octree.setNodeColor(points[i], data[i, 0], data[i, 1], data[i, 2])

    def get_bbox_iterator(self, middle: np.ndarray, extent: np.ndarray, max_depth: int=0):
        min_bound = middle - extent / 2
        max_bound = middle + extent / 2
        return self.octree.begin_leafs_bbx(min_bound, max_bound, max_depth)

    def get_bbox_points(self, middle: np.ndarray, extent: np.ndarray, max_depth: int=0):
        bbox_iterator = self.get_bbox_iterator(middle, extent, max_depth)
        points = []
        colors = []
        for leaf_it in bbox_iterator:
            coord = leaf_it.getCoordinate()
            if self.octree.isNodeOccupied(leaf_it):
                points.append(coord)
                colors.append(leaf_it.getColor())
        return np.array(points), np.array(colors)

    def get_bounds(self):
        points = self.octree.extractPointCloud()[0]
        return points.min(axis=0), points.max(axis=0)