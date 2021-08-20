import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import requests
from scipy.io import loadmat


# function to load MNIST dataset
def load_mnist(batch_size):
  mnist = MNIST("./", train=True, download=True, transform=transforms.Compose([
                                                                                transforms.ToTensor()]) )
  mnist_test = MNIST("./", train=False, download=True,  transform=transforms.Compose([
                                                                              transforms.ToTensor()]) )
  mnist_dataloader = DataLoader(mnist, batch_size=batch_size)
  mnist_test_dataloader = DataLoader(mnist_test, batch_size=batch_size)
  return mnist_dataloader, mnist_test_dataloader

# function to load frey face dataset
def load_frey_face(batch_size):
  # download the data
  link = "https://cs.nyu.edu/~roweis/data/frey_rawface.mat"
  r = requests.get(link)
  fileName = "frey_rawface.mat"
  open(fileName, 'wb').write(r.content)
  frey_face_mat = loadmat(fileName) # load the mat file
  frey_face_input = torch.tensor( frey_face_mat['ff'].T.reshape(-1, 1, 28, 20))
  dummy_targets = torch.zeros(frey_face_input.shape[0])
  # print(frey_face_input[0])
  size = frey_face_input.shape[0]
  train_size = int(0.9 * size)

  frey_face = TensorDataset((frey_face_input[: train_size]), dummy_targets[: train_size])
  frey_face_test = TensorDataset((frey_face_input[train_size: ]), dummy_targets[train_size:])

  frey_face_dataloader = DataLoader(frey_face, batch_size=batch_size)
  frey_face_test_dataloader = DataLoader(frey_face_test, batch_size=batch_size)
  return frey_face_dataloader, frey_face_test_dataloader

# plot a batch of images as a grid.
def plot_grid(dataloader):
  images, _ = next(iter(dataloader))
  grid = make_grid(images, )
  plt.figure(figsize=(10, 10))
  plt.imshow(grid.permute(1, 2, 0))
