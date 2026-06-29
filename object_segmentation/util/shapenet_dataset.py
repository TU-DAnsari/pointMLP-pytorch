import h5py
import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm

class ShapeNetDataset(Dataset):
    def __init__(self, 
                 h5_path,
                 classes = [],
                 split="train",
                 num_points=1024,
                 seed=42,
                ):
        
        super().__init__()

        rng_sampling = np.random.default_rng(seed=seed)

        self.points = []
        self.features = []
        self.labels = []

        with h5py.File(h5_path) as f:
            grp = f[split]

            if not classes:
                classes = grp.keys()
            
            for class_name in classes:
                for instance in grp[class_name].keys():
                    points = np.asarray(grp[class_name][instance]["points"], dtype=np.float32)
                    normals = np.asarray(grp[class_name][instance]["normals"], dtype=np.float32)

                    replace = len(points) < num_points
                    chosen = rng_sampling.choice(len(points), num_points, replace=replace)
                    
                    self.points.append(points[chosen])
                    self.features.append(np.concatenate([points[chosen], normals[chosen]], axis=1))
                    self.labels.append(class_name)

        self.points = np.array(self.points)
        self.features = np.array(self.features)

        self.remap = {}

        for i, class_label in enumerate(classes):
            self.remap[class_label] = i

        self.labels = [self.remap[label] for label in self.labels]
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.points[idx], self.features[idx], self.labels[idx]