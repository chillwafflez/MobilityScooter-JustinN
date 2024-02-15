import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset_class import CoordinateDatasetV1
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# print(torch.__version__)
# print(torch.version.cuda)

def get_dataloader(path):
    df = pd.read_csv(path)
    print(df.head())

    # Split data into samples of 120 frames
    organized_data = [df.iloc[i : i + 120].values for i in range(0, len(df) - 4, 120)]     # convert data into list of dataframes, each dataframe has 120 records
    organized_data = np.array(organized_data)      # convert list into numpy array
    
    # Create Dataset
    train_dataset = CoordinateDatasetV1(organized_data)

    # Create DataLoader with batch size of 32
    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE)
    return train_dataloader


# Transformer Encoder using PyTorch layers
class Transformer_Encoder(nn.Module):
  def __init__(self, num_features, heads, num_layers):
    super().__init__()
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=num_features, # number of features in input
                                                    nhead=heads)          # number of heads in the multiheadattention models
    self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                         num_layers=num_layers)

  def forward(self, x):
    return self.encoder(x)

def train():
    data_path = 'data\\090620231100\p7_front_1.csv'
    dataloader = get_dataloader(data_path)

    EPOCHS = 5
    FEATURES = 18
    HEADS = 6
    LAYERS = 4

    encoder = Transformer_Encoder(num_features=FEATURES,
                                heads=HEADS,
                                num_layers=LAYERS)

    for batch, X in enumerate(dataloader):
        encoding = encoder(X)
        print(f"Encoding: {encoding}")
    
    print(encoding.shape)

train()