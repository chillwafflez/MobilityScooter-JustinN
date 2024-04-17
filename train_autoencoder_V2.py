# Using the time-series anomaly detection LSTM autoencoder methodology from: 
# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
# but changing it up to fit my purpose

import torch
from torch import nn
from torch.utils.data import DataLoader
from models.LSTM_AutoencoderV2 import LSTM_AutoencoderV2
from PoseDataset import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from train_utils import plot_loss_during_training

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, train_dataloader, val_dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    history = dict(train=[], val=[])

    for epoch in range(epochs):
        model.train()

        train_losses = []
        for seq_true in train_dataloader:
            optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = loss_fn(seq_pred, seq_true)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        
        val_losses = []
        model.eval()
        with torch.inference_mode():
            for seq_true in val_dataloader:

                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = loss_fn(seq_pred, seq_true)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f"Epoch: {epoch + 1} | Train loss: {train_loss} | Validation loss: {val_loss}")
    
    return model.eval(), history


# ------ Setup Datasets and DataLoaders ------ #
RANDOM_SEED = 42

df = pd.read_csv('data/raw_stable_pose_data/051920231000/P5_Front_Track_1.csv')
dataset = PoseDataDatasetV1(df)
train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)

BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
validation_dataloader = DataLoader(dataset=validation_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)

# ----------- Training ----------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
INPUT_SIZE = 18
EMBEDDING_DIM = 64
SEQUENCE_LENGTH = 120

LSTM_autoencoder = LSTM_AutoencoderV2(seq_len=SEQUENCE_LENGTH,
                                      n_features=INPUT_SIZE,
                                      embedding_dim=EMBEDDING_DIM).to(device)
# LSTM_autoencoder.load_state_dict(torch.load("./saved_models/LSTM_autoencoderV2.pth"))

LSTM_autoencoder, loss_history = train_model(LSTM_autoencoder, train_dataloader, validation_dataloader, EPOCHS)
plot_loss_during_training(loss_history['train'], loss_history['val'])


# Save the model's parameters (training or inference)
model_save_path = "./saved_models/LSTM_autoencoderV2.pth"
torch.save(obj = LSTM_autoencoder.state_dict(),
           f = model_save_path)