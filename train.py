import random

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from src.dataloader import load_mnist, load_fmnist
from src.model import Encoder, Decoder
from src.utils import train, vae_loss
from src.config import args

import matplotlib.pyplot as plt

import argparse


# pass the dataset as a cmd argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
arguments = parser.parse_args()

# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set the seed
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

# function for expirmentation

def experiment(epochs, input_size, hidden_size, bottleneck, height=None, width=None, plot_freq=3, step_size=0.03, data=None, data_test=None, device="cpu"):
  # save image for comparison
  encoder = Encoder(input_size, hidden_size, bottleneck).to(device) # define the encoder
  decoder = Decoder(bottleneck, hidden_size, input_size).to(device) # define the decoder

  optimizer = opt.Adagrad(list(encoder.parameters()) + list(decoder.parameters()) , lr=step_size) # define the optimizer
  if height != None and width != None:
    loss, test_loss, data, output = train(encoder, decoder, vae_loss, optimizer, data, epochs, plot=False, testloader=data_test, height=height, width=width, data="freyface", mse=True, plot_freq=plot_freq, activation=False, device=device)
  else:
    loss, test_loss, data, output = train(encoder, decoder, vae_loss, optimizer, data, epochs, plot=False, testloader=data_test, activation=True, mse=False, plot_freq=plot_freq, device=device)

  return loss, test_loss, data, output

def plot_loss(loss, loss_test, n, data="MNIST"):
  x_labels = [i*10**6 for i in range(1,len(loss)+1)]
  plt.plot(x_labels, loss, '-r', label="AEVB (train)")
  plt.plot(x_labels, loss_test, '--r', label="AEVB (test)")
  plt.xscale('log')
  plt.xlabel("# Training samples evaluated")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(f"{data} N={n}")


# mnist expirement
def MNIST_exp(data_loader, data_test):
    # different size of the latent space
    epochs = args.epochs
    step_size = args.lr
    input_size = 784
    for bottleneck in args.bottlenecks:
      loss, test_loss, data, output = experiment(epochs,input_size, args.hidden_size, bottleneck, step_size=step_size, data=data_loader, data_test=data_test)
      if bottleneck == args.bottlenecks[0]:
        print(f"Original images")
        plt.figure(figsize=(15, 10))
        plt.imshow(data.permute(1, 2, 0))
        plt.savefig("original_image")
        plt.show()
      print(f"MNIST Image Generated with latent space size of {bottleneck}")
      plt.figure(figsize=(15, 10))
      plt.imshow(output.permute(1, 2, 0))
      plt.savefig(f"MNIST Image Generated with latent space size of {bottleneck}")
      plt.show()
      print(f"Losses for bottleneck={bottleneck}")
      plot_loss(loss, test_loss, bottleneck)

# mnist expirement
def FMNIST_exp(data_loader, data_test):
    # different size of the latent space
    epochs = args.epochs
    step_size = args.lr
    input_size = 784
    for bottleneck in args.bottlenecks:
      loss, test_loss, data, output = experiment(epochs,input_size, args.hidden_size, bottleneck, step_size=step_size, data=data_loader, data_test=data_test)
      if bottleneck == args.bottlenecks[0]:
        print(f"Original images")
        plt.figure(figsize=(15, 10))
        plt.imshow(data.permute(1, 2, 0))
        plt.savefig("original_image")
        plt.show()
      print(f"FMNIST Image Generated with latent space size of {bottleneck}")
      plt.figure(figsize=(15, 10))
      plt.imshow(output.permute(1, 2, 0))
      plt.savefig(f"FMNIST Image Generated with latent space size of {bottleneck}")
      plt.show()
      print(f"Losses for bottleneck={bottleneck}")
      plot_loss(loss, test_loss, bottleneck)


# def FreyFace_exp(data, data_test):
#     input_size = 560
#     # different size of the latent space
#     epochs = args.epochs
#     step_size = args.lr
#     for bottleneck in args.bottlenecks:
#       loss, test_loss, data, output = experiment(epochs,input_size, args.hidden_size, bottleneck, step_size=step_size, height=28, width=20, plot_freq=100, data=data, data_test=data_test)
#       if bottleneck == args.bottlenecks[0]:
#         print(f"Original images fery face")
#         plt.figure(figsize=(15, 10))
#         plt.imshow(data.permute(1, 2, 0))
#         plt.savefig("original_image")
#         plt.show()
#       print(f"frey face Image Generated with latent space size of {bottleneck}")
#       plt.figure(figsize=(15, 10))
#       plt.imshow(output.permute(1, 2, 0))
#       plt.savefig(f"frey face Image Generated with latent space size of {bottleneck}")
#       plt.show()
#       print(f"frey face Losses for bottleneck={bottleneck}")
#       plot_loss(loss, test_loss, bottleneck,  data="Frey face")

if __name__ == "__main__":
    dataset = arguments.dataset
    if dataset == "mnist":
        print("-------------------- Loading MNIST --------------------")
        mnist, mnist_test = load_mnist(args.batch_size)
        MNIST_exp(mnist, mnist_test)
    elif dataset == "fmnist":
        print("-------------------- Loading Fashion MNIST --------------------")
        fmnist, fmnist_test = load_fmnist(args.batch_size)
        FMNIST_exp(fmnist, fmnist_test)

