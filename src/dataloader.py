import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from src.config import args

import requests
from scipy.io import loadmat

# function to load MNIST dataset
def load_mnist(batch_size):
  mnist = MNIST(args.mnist_root, train=True, download=True, transform=transforms.Compose([
                                                                                transforms.ToTensor()]) )
  mnist_test = MNIST(args.mnist_root, train=False, download=True,  transform=transforms.Compose([
                                                                              transforms.ToTensor()]) )
  mnist_dataloader = DataLoader(mnist, batch_size=batch_size)
  mnist_test_dataloader = DataLoader(mnist_test, batch_size=batch_size)
  return mnist_dataloader, mnist_test_dataloader


# function to load Fashion MNIST dataset
def load_fmnist(batch_size):
  fmnist = FashionMNIST(args.fmnist_root, train=True, download=True, transform=transforms.Compose([
                                                                                transforms.ToTensor()]) )
  fmnist_test = FashionMNIST(args.fmnist_root, train=False, download=True,  transform=transforms.Compose([
                                                                              transforms.ToTensor()]) )
  fmnist_dataloader = DataLoader(fmnist, batch_size=batch_size)
  fmnist_test_dataloader = DataLoader(fmnist_test, batch_size=batch_size)
  return fmnist_dataloader, fmnist_test_dataloader

  #  function to load frey face dataset
def load_frey_face(batch_size):
  # download the data
  link = "https://cs.nyu.edu/~roweis/data/frey_rawface.mat"
  r = requests.get(link)
  data_directory = args.freyface_root
  fileName = "frey_rawface.mat"
  file_directory = data_directory + "/" + fileName
  open(file_directory, 'wb').write(r.content)
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