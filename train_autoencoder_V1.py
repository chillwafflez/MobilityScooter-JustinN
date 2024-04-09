import torch
from torch import nn
from torch.utils.data import DataLoader
from models.LSTM_AutoencoderV1 import LSTM_Autoencoder
from train_utils import train_model, plot_loss
from data_utils import PoseDataDatasetV2
import pandas as pd
from sklearn.model_selection import train_test_split


# ------ Setup Datasets and DataLoaders ------ #
RANDOM_SEED = 42

df = pd.read_csv('data\\stable_data\\090620231100\p7_front_1.csv')

dataset = PoseDataDatasetV2(df)

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)

BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE)
test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE)

# ----------- Training ----------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
INPUT_SIZE = 18
EMBEDDING_DIM = 9
SEQUENCE_LENGTH = 120

LSTM_autoencoder = LSTM_Autoencoder(seq_length=SEQUENCE_LENGTH,
                                    input_size=INPUT_SIZE,
                                    embedding_dim=EMBEDDING_DIM).to(device)


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params = LSTM_autoencoder.parameters(),
                                lr = 0.001)

print("--- Training ---")
train_loss, test_loss, penis, cock = train_model(LSTM_autoencoder, EPOCHS, train_dataloader, test_dataloader, loss_fn, optimizer, device)
print(f"train loss: {train_loss}")
print(f"test loss: {test_loss}")
plot_loss(train_loss, test_loss)