import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset from pose data where each sequence is 120 records long (4 seconds) by default. Each sample does not contain overlapping timesteps
class PoseDataDatasetV1(Dataset):
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

# Dataset from pose data where each sequence is 120 records long (4 seconds) by default. Each sample has overlapping timesteps, resulting in more samples
class PoseDataDatasetV2(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length=120):
        self.data = data.to_numpy()                     # convert dataframe to numpy array
        self.seq_len = sequence_length
        
    def __len__(self):
        return len(self.data) - self.seq_len
        
    def __getitem__(self, idx):
        item_as_tensor = torch.tensor(self.data[idx:idx + self.seq_len: ], dtype=torch.float)
        return item_as_tensor  

# df = pd.read_csv('data\\raw_stable_pose_data\\040520231330\FrontView_1.csv')
# df2 = pd.read_csv('data\\raw_stable_pose_data\\040520231330\FrontView_1.csv')
# dataset_v2 = PoseDataDatasetV2(df2)
# print(f"length of datasetV2: {len(dataset_v2)}")
# dataset_v3 = PoseDataDatasetV3(df)
# print(f"length of datasetV3: {len(dataset_v3)}")
# print(dataset_v3[3].shape)
# print(dataset_v3[0])
