from .base_dataset import BaseDataSet


class SimpleDataset(BaseDataSet):
    def __init__(self, data):
        super().__init__()

        self.data = data

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, index):
        return tuple(data[index] for data in self.data)