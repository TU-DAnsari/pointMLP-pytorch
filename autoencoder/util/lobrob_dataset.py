import h5py
import numpy as np
from tqdm import tqdm

class LobRobDataset():
    def __init__(self, 
                 h5_path, 
                 split="train", 
                 num_points=1024, 
                 seed=42,
                ):
        
        super().__init__()

        rng_sampling = np.random.default_rng(seed=seed)

        sources, targets = [], []

        with h5py.File(h5_path, "r") as f:
            samples = list(f[split].keys())


            for sample in tqdm(samples, desc=f"Loading {split}"):
                grp = f[split][sample]

                source = np.asarray(grp["source"], dtype=np.float32)
                target = np.asarray(grp["target"], dtype=np.float32)

                replace = len(source) < num_points
                chosen = rng_sampling.choice(len(source), num_points, replace=replace)

                sources.append(source[chosen])
                targets.append(target[chosen])

        self.sources = np.array(sources)
        self.targets = np.array(targets)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx]