import data
import model
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()

dataset = args.dataset


# Hyperparameters
batch_size = 100
# Networks parameters for MNIST Experemints
hidden_size = 500
input_size = 784
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""# Experimental Setup"""

def experiment(epochs,input_size, hidden_size, bottleneck, height=None, width=None, plot_freq=10, step_size=0.03):
  # save image for comparison
  encoder = model.Encoder(input_size, hidden_size, bottleneck).to(device) # define the encoder
  decoder = model.Decoder(bottleneck, hidden_size, input_size).to(device) # define the decoder

  optimizer = opt.Adagrad(list(encoder.parameters()) + list(decoder.parameters()) , lr=step_size) # define the optimizer
  if height != None and width != None:
    loss, test_loss, data, output = model.train(encoder, decoder, model.vae_loss, optimizer, freyface, epochs, plot=False, testloader=freyface_test, height=height, width=width, data="freyface", mse=True, plot_freq=plot_freq, activation=False, device=device)
  else:
    loss, test_loss, data, output = model.train(encoder, decoder, model.vae_loss, optimizer, mnist, epochs, plot=False, testloader=mnist_test, activation=True, mse=False, plot_freq=plot_freq, device=device)

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


def MNIST_exp():
	# different size of the latent space
	N = [3, 5, 10, 20, 200]
	epochs = 10
	step_size = 0.03
	for bottleneck in N:
	  loss, test_loss, data, output = experiment(epochs,input_size, hidden_size, bottleneck, step_size=step_size)
	  if bottleneck == N[0]:
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
	  print(f"Losses for N={bottleneck}")
	  plot_loss(loss, test_loss, bottleneck)


def FreyFace_exp():
	hidden_size = 100
	input_size = 560
	# different size of the latent space
	N = [2, 5, 10, 20]
	epochs =5000
	dataset_size = 1950
	stepsize = 0.1
	for bottleneck in N:
	  loss, test_loss, data, output = experiment(epochs,input_size, hidden_size, bottleneck, height=28, width=20, plot_freq=100)
	  if bottleneck == N[0]:
	    print(f"Original images fery face")
	    plt.figure(figsize=(15, 10))
	    plt.imshow(data.permute(1, 2, 0))
	    plt.savefig("original_image")
	    plt.show()
	  print(f"frey face Image Generated with latent space size of {bottleneck}")
	  plt.figure(figsize=(15, 10))
	  plt.imshow(output.permute(1, 2, 0))
	  plt.savefig(f"frey face Image Generated with latent space size of {bottleneck}")
	  plt.show()
	  print(f"frey face Losses for N={bottleneck}")
	  plot_loss(loss, test_loss, bottleneck,  data="Frey face")

mnist, mnist_test = data.load_mnist(batch_size)
freyface, freyface_test = data.load_frey_face(batch_size)

# show a batch from the data
# data.plot_grid(mnist)
# plt.show()
# data.plot_grid(freyface)
# plt.show()

if __name__ == '__main__':
	if dataset == 'mnist':
		MNIST_exp()
	else:
		FreyFace_exp()

