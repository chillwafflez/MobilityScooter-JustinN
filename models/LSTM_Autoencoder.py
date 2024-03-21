from torch import nn

# Create LSTM Autoencoder using PyTorch layers
class LSTM_Autoencoder(nn.Module):
  def __init__(self, input_size, hidden_dim, embedding_dim):
    super(LSTM_Autoencoder, self).__init__()

    self.encoder = nn.LSTM(input_size = input_size,
                                        hidden_size = hidden_dim,
                                        batch_first = True)
    self.compress = nn.Linear(hidden_dim, embedding_dim)            # Linear layer to compress the output of the encoder layer

    self.decoder = nn.LSTM(input_size = embedding_dim,
                           hidden_size = input_size,
                           batch_first = True)

  def forward(self, x):
    encoding, state = self.encoder(x)
    embedding = self.compress(encoding)                             # Compress the output of encoder layer into your defined embedding dimension
    reconstruction, state = self.decoder(embedding)
    return reconstruction