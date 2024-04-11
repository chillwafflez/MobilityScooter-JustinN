import torch
from torch import nn
from torch.utils.data import DataLoader
from models.LSTM_AutoencoderV1 import LSTM_Autoencoder
from train_utils import train_model, plot_loss_during_training
from PoseDataset import PoseDataDatasetV2
import pandas as pd
from sklearn.model_selection import train_test_split


# ------ Setup Datasets and DataLoaders ------ #
RANDOM_SEED = 42

df = pd.read_csv('data\\stable_data\\090620231100\p7_front_1.csv')
# df = pd.read_csv('data\stable_data\\042820231100\P3_Front_Track_2.csv')
dataset = PoseDataDatasetV2(df)
train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)

BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(dataset=validation_dataset,
                                batch_size=BATCH_SIZE)

# ----------- Training ----------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
INPUT_SIZE = 18
EMBEDDING_DIM = 64
SEQUENCE_LENGTH = 120

LSTM_autoencoder = LSTM_Autoencoder(seq_length=SEQUENCE_LENGTH,
                                    input_size=INPUT_SIZE,
                                    embedding_dim=EMBEDDING_DIM).to(device)
# Uncomment below to load saved model parameters into model instance
# LSTM_autoencoder.load_state_dict(torch.load("./saved_models/LSTM_autoencoder.pth"))

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = LSTM_autoencoder.parameters(),
                                lr = 0.001)
# optimizer.load_state_dict(torch.load("./saved_models/opimizer_state"))

print("--- Training ---")
train_loss, validation_loss = train_model(LSTM_autoencoder, EPOCHS, train_dataloader, validation_dataloader, loss_fn, optimizer, device)
plot_loss_during_training(train_loss, validation_loss)


# Save the model's parameters (training or inference) and optimizer's state (training)
model_save_path = "./saved_models/LSTM_autoencoder.pth"
torch.save(obj = LSTM_autoencoder.state_dict(),
           f = model_save_path)

optimizer_save_path = "./saved_models/opimizer_state"
torch.save(obj = optimizer.state_dict(), 
           f = optimizer_save_path)