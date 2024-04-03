def train_LSTM_autoencoder(model, epochs, dataloader, loss_fn, optimizer, device):
  for epoch in range(epochs):
        total_loss = 0   # loss per epoch

        for batch, X in enumerate(dataloader):
            X = X.to(device)

            # Forward method
            output = model(X)

            # Calculate loss
            loss = loss_fn(output, X)
            total_loss += loss

            # Optimizer zero grad
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Step the optimizer
            optimizer.step()

        total_loss /= len(dataloader)
        print(f"Epoch: {epoch} | Loss: {total_loss}")

