import torch
from torch.utils.data import Dataset

class CoordinateDatasetV1(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item_as_tensor = torch.tensor(self.data[idx], dtype=torch.float)
        return item_as_tensor