from argparse import Namespace

args = Namespace(
# directory of the mnist dataset
mnist_root = "./data/mnist",

# directory of the mnist dataset
fmnist_root = "./data/fmnist",

# set the batch size for pytorch data loader
batch_size = 100,

#size of the hidden 
hidden_size = 100,

# latent space size
bottlenecks = [3, 5, 10, 20, 200],

# epochs
epochs = 15,

# learning rate
lr = 0.03,

# weight decay for l2 regulrization
weight_decay = 1,

# seed
manual_seed = 999
)

