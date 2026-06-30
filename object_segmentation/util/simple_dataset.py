from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data):
        super().__init__()

        self.__data = data

    def __len__(self):
        return len(self.__data[0])
    
    def __getitem__(self, index):
        return tuple(data[index] for data in self.__data)