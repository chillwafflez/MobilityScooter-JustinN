# Using the time-series anomaly detection LSTM autoencoder methodology from: 
# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/

from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True)

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True)

    def forward(self, x):
        # x = x.reshape((1, self.seq_len, self.n_features))         # if batch size is not included in input, this just adds 1 for batch size dimension
        # print(f"input into encoder: {x.shape}")
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        # print(f"x = {x.shape}  | hidden_n = {hidden_n.shape} | output of encoder = {hidden_n.reshape((1, self.embedding_dim)).shape}")
        return hidden_n.reshape((1, self.embedding_dim))
        
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True)

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True)

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, 1)
        # print(f"Input into decoder (after repeat): {x.shape}")
        x = x.reshape((1, self.seq_len, self.input_dim))
        # print(f"Input after reshape: {x.shape}")

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        # x = x.reshape((self.seq_len, self.hidden_dim))            # assuming batch size is not specified in input 
        x = x.reshape((1, self.seq_len, self.hidden_dim))

        return self.output_layer(x)
    

class LSTM_AutoencoderV2(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTM_AutoencoderV2, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


# def main():
#   test_tensor = torch.rand((120, 18))
#   model = Encoder(seq_len=120, n_features=18, embedding_dim=64)
#   embedding = model(test_tensor)

#   test_tensor = torch.rand((1, 64))
#   model = Decoder(seq_len=120, input_dim=64, n_features=18)
#   output = model(test_tensor)

#   test_tensor = torch.rand((1, 120, 18))
#   model = LSTM_AutoencoderV2(seq_len=120, n_features=18, embedding_dim=64)
#   output = model(test_tensor)
#   print(f"output of decoder shape: {output.shape}")

# main()