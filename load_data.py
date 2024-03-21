import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class PoseDataDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item_as_tensor = torch.tensor(self.data[idx], dtype=torch.float)
        return item_as_tensor
    
def get_dataloader(path):
    df = pd.read_csv(path)

    # Split data into samples of 120 frames
    organized_data = [df.iloc[i : i + 120].values for i in range(0, len(df) - 4, 120)]     # convert data into list of numpy arrs, each dataframe has 120 records
    organized_data = np.array(organized_data)      # convert list into numpy array
    
    # Create Dataset
    train_dataset = PoseDataDataset(organized_data)

    # Create DataLoader with batch size of 32
    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE)
    return train_dataloader