import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PoseDataset import CoordinateDatasetV1
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# print(torch.__version__)
# print(torch.version.cuda)

def get_dataloader(path):
    df = pd.read_csv(path)
    print(df.head())

    # Split data into samples of 120 frames
    organized_data = [df.iloc[i : i + 120].values for i in range(0, len(df) - 4, 120)]     # convert data into list of numpy arrs, each dataframe has 120 records
    organized_data = np.array(organized_data)      # convert list into numpy array
    
    # Create Dataset
    train_dataset = CoordinateDatasetV1(organized_data)

    # Create DataLoader with batch size of 32
    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE)
    return train_dataloader

# Create Transformer Autoencoder using PyTorch layers
class Transformer_Autoencoder(nn.Module):
    def __init__(self, input_dim, heads, num_layers):
        super().__init__()
        # Encoding
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, # number of features in input
                                                        nhead=heads,          # number of heads in the multiheadattention models
                                                        batch_first=True)     # bcuz batch first the input tensor shape should be (batch_size, seq_len, embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=num_layers)

        # Decoding
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim,
                                                        nhead=heads,
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer,
                                            num_layers=num_layers)

        # Turn output of decoder into same shape as input
        # self.output = nn.Linear(input_dim, input_dim)
        self.output = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.ReLU())

    def forward(self, x):
        # Create encoding
        encoding = self.encoder(x)
        decoding = self.decoder(x, encoding)
        output = self.output(decoding)
        return output
        # return decoding
  

def train(data_path):
    train_dataloader = get_dataloader(data_path)

    EPOCHS = 20
    FEATURES = 18
    HEADS = 6
    LAYERS = 4

    autoencoder = Transformer_Autoencoder(input_dim=FEATURES,
                                          heads=HEADS,
                                          num_layers=LAYERS).to(device)
    

    # Training
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params = autoencoder.parameters(),
                                lr = 0.001)
    autoencoder.train()
    for epoch in range(EPOCHS):
        total_loss = 0   # loss per epoch

        for batch, X in enumerate(train_dataloader):
            X = X.to(device)

            # Forward method
            output = autoencoder(X)

            # Calculate loss
            loss = loss_fn(output, X)
            total_loss += loss

            # Optimizer zero grad
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Step the optimizer
            optimizer.step()

        total_loss /= len(train_dataloader)
        print(f"Epoch: {epoch} | Loss: {total_loss}")
    print(output.shape)


def main():
    data_path = 'data\\090620231100\p7_front_1.csv'
    train(data_path)

main()