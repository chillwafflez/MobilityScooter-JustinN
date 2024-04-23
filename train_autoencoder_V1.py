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

path = 'data\processed_stable_pose_data\\051920230915\P1_Front_Track_2.csv'
df = pd.read_csv(path)             # Load CSV files into PyTorch dataset
dataset = PoseDataDatasetV2(df, sequence_length=120)
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
EPOCHS = 10
NUM_FEATURES = 18
EMBEDDING_DIM = 64
SEQUENCE_LENGTH = 120

LSTM_autoencoder = LSTM_Autoencoder(seq_length=SEQUENCE_LENGTH,
                                    input_size=NUM_FEATURES,
                                    embedding_dim=EMBEDDING_DIM).to(device)
# Uncomment below to load saved model parameters into model instance
LSTM_autoencoder.load_state_dict(torch.load("./saved_models/LSTM_autoencoderV1.pth"))

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = LSTM_autoencoder.parameters(),
                                lr = 0.001)

print("--- Training ---")
train_loss, validation_loss = train_model(LSTM_autoencoder, EPOCHS, train_dataloader, validation_dataloader, loss_fn, optimizer, device)
plot_loss_during_training(train_loss, validation_loss)


# Save the model's parameters (training or inference)
model_save_path = "./saved_models/LSTM_autoencoderV1.pth"
torch.save(obj = LSTM_autoencoder.state_dict(),
           f = model_save_path)

# Save loss values to text file
full_path = os.path.split(path)
date = os.path.split(full_path[0])[1]
filename = full_path[1].rstrip('.csv') + '.txt'
print(filename)
save_directory = "loss\\" + date
if not os.path.exists(save_directory):
    print("creating directory for loss values")
    os.makedirs(save_directory)
with open(save_directory + "\\" + filename, 'a') as loss_file:
    for i in range(len(train_loss)):
        loss_file.write(f"Epoch: {i + 1} | Train loss: {train_loss[i]} | Validation loss: {validation_loss[i]}\n")