from torch import nn
import torch

# Create LSTM Autoencoder using PyTorch layers
class LSTM_Autoencoder(nn.Module):
  def __init__(self, seq_length, input_size, embedding_dim):
    super(LSTM_Autoencoder, self).__init__()
    self.seq_length = seq_length

    self.encoder_layer_1 = nn.LSTM(input_size = input_size,
                                   hidden_size = 2 * embedding_dim,
                                   batch_first = True)
    self.encoder_layer_2 = nn.LSTM(input_size = 2 * embedding_dim,
                                   hidden_size = embedding_dim,
                                   batch_first = True)

    self.decoder_layer_1 = nn.LSTM(input_size = embedding_dim,
                                   hidden_size = 2 * embedding_dim,
                                   batch_first = True)
    self.decoder_layer_2 = nn.LSTM(input_size = 2 * embedding_dim,
                                   hidden_size = input_size,
                                   batch_first = True)

  def forward(self, x):
    # print(f"input = {x.shape}")

    # -------- Encoder -------- #
    x, (_, _) = self.encoder_layer_1(x)
    _, (embedding, _) = self.encoder_layer_2(x)

    # print(f"embedding shape (hidden state of LSTM): {embedding.shape}")
    x = embedding.repeat(1, self.seq_length, 1)

    # -------- Decoder -------- #
    x, (_, _) = self.decoder_layer_1(x)
    x, (_, _) = self.decoder_layer_2(x)
    # print(f"after decoder: x = {x.shape}")
    return x
  
  
# def main():
#   test_tensor = torch.rand((1, 120, 18))
#   model = LSTM_Autoencoder(seq_length= 120, input_size=18, embedding_dim=64)
#   output = model(test_tensor)

# main()