import torch
from torch import nn
from load_data import get_dataloader
from models.LSTM_Autoencoder import LSTM_Autoencoder
from train_utils import train_LSTM_autoencoder


def main():
    data_path = "data\\090620231100\p7_front_1.csv"
    train_dataloader = get_dataloader(data_path)

    # ------ Test training LSTM autoencoder ------ #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 10
    INPUT_SIZE = 18
    HIDDEN_SIZE = 32
    EMBEDDING_DIM = 9

    LSTM_autoencoder = LSTM_Autoencoder(input_size=INPUT_SIZE,
                                        hidden_dim=HIDDEN_SIZE,
                                        embedding_dim=EMBEDDING_DIM).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params = LSTM_autoencoder.parameters(),
                                 lr = 0.001)

    print("--- Training ---")
    train_LSTM_autoencoder(LSTM_autoencoder, EPOCHS, train_dataloader, device, loss_fn, optimizer)


    # Get embedding of input data using only encoder from LSTM autoencoder
    embeddings = []

    LSTM_autoencoder.eval()
    with torch.inference_mode():
        for batch, X in enumerate(train_dataloader):
            X = X.to(device)
            encoding, state = LSTM_autoencoder.encoder(X)
            embedding = LSTM_autoencoder.compress(encoding)
            embeddings.append(embedding)
    
    # print(embeddings)
    print(f"\nLength of embeddings list: {len(embeddings)} (corresponding to 5 batches in dataloader)")
    print(f"Size of one embedding: {embeddings[0].shape}")

main()