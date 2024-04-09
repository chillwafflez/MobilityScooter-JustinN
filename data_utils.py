import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
        # convert data into list of numpy arrays, each array has 120 records (4 seconds)
        organized_data = [data.iloc[i : i + sequence_length].values for i in range(0, len(data) - (len(data) % sequence_length), sequence_length)]     
        organized_data = np.array(organized_data)      # convert list into numpy array
        self.data = organized_data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item_as_tensor = torch.tensor(self.data[idx], dtype=torch.float)
        return item_as_tensor  


# def main():
#     df = pd.read_csv('data\\stable_data\\090620231100\p7_front_1.csv')
#     datasetv2 = PoseDataDatasetV2(df)
#     print(len(datasetv2))

#     train_dataset, test_dataset = train_test_split(datasetv2, test_size=0.2)
#     print(train_dataset[3].shape)
#     print(test_dataset[3].shape)

# main()