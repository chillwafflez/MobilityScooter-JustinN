import torch
from torch import nn
from torch.utils.data import DataLoader
from models.LSTM_AutoencoderV1 import LSTM_Autoencoder
from train_utils import train_model, plot_loss_during_training
from PoseDataset import *
import pandas as pd
from sklearn.model_selection import train_test_split
import os


# ------ Setup Datasets and DataLoaders ------ #
RANDOM_SEED = 42

path = 'data\processed_stable_pose_data\\051920230915\P1_Front_Track_3.csv'
df = pd.read_csv(path)             # Load CSV files into PyTorch dataset
test_dataset = PoseDataDatasetV2(df, sequence_length=120)

BATCH_SIZE = 1
test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

# ----------- Testing ----------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
NUM_FEATURES = 18
EMBEDDING_DIM = 64
SEQUENCE_LENGTH = 120

LSTM_autoencoder = LSTM_Autoencoder(seq_length=SEQUENCE_LENGTH,
                                    input_size=NUM_FEATURES,
                                    embedding_dim=EMBEDDING_DIM).to(device)
LSTM_autoencoder.load_state_dict(torch.load("./saved_models/LSTM_autoencoderV1.pth"))

loss_fn = nn.MSELoss()

print("--- Testing ---")
total_test_loss = []
total = 0
LSTM_autoencoder.eval()
with torch.inference_mode():   
    for batch, X in enumerate(test_dataloader):
        X = X.to(device)

        # Forward method
        output = LSTM_autoencoder(X)

        # Calculate loss
        loss = loss_fn(output, X)
        total += loss.item()
        
        if batch % 100 == 0:
            print(f"Sequence: {batch + 1} | Test loss = {loss.item()}")
            total_test_loss.append(loss.item())
total /= len(test_dataloader)
print(f"Average test loss: {total}")
plot_loss_during_training(total_test_loss)
