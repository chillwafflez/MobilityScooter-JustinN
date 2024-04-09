import torch
import matplotlib.pyplot as plt

def train_model(model, epochs, train_dataloader, test_dataloader, loss_fn, optimizer, device):
  total_train_loss = []
  total_test_loss = []
  
  for epoch in range(epochs):
    # Training
    train_loss = 0   # loss per epoch
    model.train()
    for batch, X in enumerate(train_dataloader):
        X = X.to(device)

        # Forward method
        output = model(X)

        # Calculate loss
        loss = loss_fn(output, X)
        train_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Step the optimizer
        optimizer.step()

    train_loss /= len(train_dataloader)
    total_train_loss.append(train_loss)
        
    # Testing
    test_loss = 0     # test loss per epoch
    model.eval()
    with torch.inference_mode():   
        for batch, X in enumerate(test_dataloader):
            X = X.to(device)

            # Forward method
            output = model(X)

            # Calculate loss
            loss = loss_fn(output, X)
            test_loss += loss.item()

        test_loss /= len(test_dataloader)
        total_test_loss.append(test_loss)
    print(f"Epoch: {epoch} | Train loss = {train_loss} | Test loss = {test_loss}")

  return total_train_loss, total_test_loss

# Plot train and loss curves
def plot_loss(train_loss, test_loss):
   plt.figure(figsize=(10,5))
   plt.plot(train_loss, label="train loss")
   plt.plot(test_loss, label="test loss")
   plt.xlabel("Epochs")
   plt.ylabel("Loss")
   plt.legend()
   plt.show()


# train = [0.0735313066553014, 0.004585355984939573, 0.00410072243733642, 0.003837817346599574, 0.0036556153689161876, 0.0035248944589208503, 0.003426456929688963, 0.003349259872629773, 0.0032869922356136764, 0.0032357451845503723]
# testloss = [0.00508147783887883, 0.004073491331655532, 0.0037328121368773283, 0.0035076297780809304, 0.003354102959080289, 0.003244123082064713, 0.0031610179867129773, 0.0030957222159486266, 0.0030428820677722494, 0.0029991308013753346]
# plot_loss(train, testloss)