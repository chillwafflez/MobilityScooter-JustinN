import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# fetch data
df = pd.read_csv('data\\090620231100\p7_front_1.csv')
# print(df.head())

# Split data into samples of 120 frames
organized_data = [df.iloc[i : i + 120].values for i in range(0, len(df) - (len(df) % 120), 120)]     # convert data into list of numpy arrays, each array has 120 records (4 seconds)
organized_data = np.array(organized_data)      # convert list into numpy array

print(f"Number of samples: {len(organized_data)}")
print(f"Shape of organized data: {organized_data.shape} -> (num of samples, length of sample, features)")
print(f"Type of one sample: {type(organized_data[0])}")
print(f"Shape of one sample: {organized_data[0].shape}")

# Custom PyTorch dataset class
class CoordinateDatasetV1(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item_as_tensor = torch.tensor(self.data[idx], dtype=torch.float)
        return item_as_tensor

# Create PyTorch dataset of data
train_dataset = CoordinateDatasetV1(organized_data)
# train_dataset[149]

# Create DataLoader with batch size of 32
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE)

print(f"Length of dataset = {len(train_dataset)} | Length of dataloader = {len(train_dataloader)}")

# next(iter(train_dataloader)).shape