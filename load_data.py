import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class PoseDataDatasetV1(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item_as_tensor = torch.tensor(self.data[idx], dtype=torch.float)
        return item_as_tensor
    
class PoseDataDatasetV2(Dataset):
    # Creates a dataset from pose data where each sequence is 120 records long (4 seconds) by default
    def __init__(self, data: pd.DataFrame, sequence_length=120):
        organized_data = [data.iloc[i : i + sequence_length].values for i in range(0, len(data) - (len(data) % sequence_length), sequence_length)]     # convert data into list of numpy arrays, each array has 120 records (4 seconds)
        organized_data = np.array(organized_data)      # convert list into numpy array
        self.data = organized_data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item_as_tensor = torch.tensor(self.data[idx], dtype=torch.float)
        return item_as_tensor
    

def get_dataloader(path):
    df = pd.read_csv(path)

    # Split data into samples of 120 frames
    organized_data = [df.iloc[i : i + 120].values for i in range(0, len(df) - (len(df) % 120), 120)]     # convert data into list of numpy arrays, each array has 120 records (4 seconds)
    organized_data = np.array(organized_data)      # convert list into numpy array
    
    # Create Dataset
    train_dataset = PoseDataDatasetV1(organized_data)

    # Create DataLoader with batch size of 32
    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE)
    return train_dataloader

def main():
    df = pd.read_csv('data\\stable_data\\090620231100\p7_front_1.csv')
    datasetv2 = PoseDataDatasetV2(df)
    print(f"shape of dataset v2: {len(datasetv2)}")
    print(f"shape of one sample v2: {datasetv2[5].shape}")

# main()