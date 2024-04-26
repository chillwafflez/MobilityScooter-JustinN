import torch
from torch import nn
from torch.utils.data import DataLoader
from models.LSTM_AutoencoderV1 import LSTM_Autoencoder
from data_preprocessing import concatData
from train_utils import train_model, plot_loss_during_training
from PoseDataset import *
import pandas as pd
from sklearn.model_selection import train_test_split
import os

path = 'data\yolov7\\raw_stable_pose_data\\051920230915\P1_Front_Track_4.csv'
df = pd.read_csv(path)             # Load CSV files into PyTorch dataset
dataset = PoseDataDatasetV1(df, sequence_length=120)
train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)
print(train_dataset[0])
print()
print(len(validation_dataset))
print(validation_dataset[len(validation_dataset)-1])