import torch
import torch.nn.functional as F

from torchvision.utils import make_grid

import matplotlib.pyplot as plt


def vae_loss(logvar_z, mean_z, output, target, batch_size, mse=True):
  # KL Divergence between the prior and the posterior
  kl_divergence = - 0.5 * (torch.sum(1 + logvar_z - mean_z.pow(2) - logvar_z.exp(), dim=1)).sum()
  # reconstruction loss
  if mse:
    reconstruction_loss = F.mse_loss(output, target, reduction="sum")
  else:
    reconstruction_loss = F.binary_cross_entropy(output, target, reduction="sum")
  loss = (1/batch_size) * (kl_divergence + reconstruction_loss)
  return loss



# simple function to implemenet the reparametrization trick
def reparametrization(mean, logv, device):
  eps = torch.randn_like(mean, device=device)
  z = mean + eps * logv.exp().pow(0.5)
  return z


def train(encoder, decoder, loss, optimizer, dataloader, epochs, testloader, channels=1, height=28, width=28, plot=False, mse=False,  activation=True, data="mnist", plot_freq=10, device='cpu'):
  losses = []
  test_losses = []
  # Main training loop
  for epoch in range(epochs):
    for img, _ in dataloader:
      if data == "freyface":
        img_flattend = img.reshape(-1, (torch.tensor(img.shape[1:])).prod()).to(torch.float32)
      else:
        img_flattend = img.reshape(-1, (torch.tensor(img.shape[1:])).prod())
      mu, logv = encoder(img_flattend.to(device))
      z = reparametrization(mu, logv, device)
      if activation:
        output = decoder(z.to(device), torch.sigmoid)
      else:
        output = decoder(z.to(device))
      loss = vae_loss(logv.to(device), mu.to(device), output.to(device), img_flattend.to(device), len(img), mse=mse)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    losses.append(-loss)

    # plot some results every 10 epochs
    if (epoch+1) % plot_freq == 0 :
      targets = img[:10]
      output_reshaped = output.reshape(-1, channels, height, width)[:10]
      target_grid = make_grid(targets.cpu().detach(), nrow=10)
      if mse:
        output_grid = make_grid(output_reshaped.cpu().detach().to(torch.int32), nrow=10)
      else:
        output_grid = make_grid(output_reshaped.cpu().detach(), nrow=10)
      if plot:
        plt.figure(figsize=(15, 10))
        plt.imshow(target_grid.permute(1, 2, 0))
        plt.figure(figsize=(15, 10))
        plt.imshow(output_grid.permute(1, 2, 0))
        plt.show()

    # evaluate on the test set
    with torch.no_grad():
      for img, _ in testloader:
        if data == "freyface":
          img_flattend = img.reshape(-1, (torch.tensor(img.shape[1:])).prod()).to(torch.float32)
        else:
          img_flattend = img.reshape(-1, (torch.tensor(img.shape[1:])).prod())
        mu, logv = encoder(img_flattend.to(device))
        z = reparametrization(mu, logv, device)
        if activation:
          output = decoder(z.to(device), torch.sigmoid)
        else:
          output = decoder(z.to(device))
        test_loss = vae_loss(logv.to(device), mu.to(device), output.to(device), img_flattend.to(device), len(img), mse=mse)
        # test_loss = vae_loss(logv.to(device), mu.to(device), output.to(device), img_flattend.to(device), 60000, len(img), mse=False)
      test_losses.append(- test_loss)

      print(f"Epoch: {epoch+1}, train loss: {loss}, test loss: {test_loss}")

  return losses, test_losses, target_grid, output_grid


  # plot a batch of images as a grid.
def plot_grid(dataloader):
  images, _ = next(iter(dataloader))
  grid = make_grid(images, )
  plt.figure(figsize=(10, 10))
  plt.imshow(grid.permute(1, 2, 0))
