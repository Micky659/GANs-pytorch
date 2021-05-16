# The original vanilla gan architecture proposed in the research paper, with leaky relu as activation
# function and batch normalization.

import torch
from torch import nn

from utils.constants import LATENT_SPACE_DIM, MNIST_IMG_SIZE


def classic_block(in_feat, out_feat, normalize=True, activation=None):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))

    layers.append(nn.LeakyReLU(0.2) if activation is None else activation)
    return layers


class GeneratorNet(torch.nn.Module):

    def __init__(self, img_shape=(MNIST_IMG_SIZE, MNIST_IMG_SIZE)):
        super().__init__()
        self.generated_img_shape = img_shape
        num_neurons_per_layer = [LATENT_SPACE_DIM, 256, 512, 1024, img_shape[0] * img_shape[1]]

        self.net = nn.Sequential(
            *classic_block(num_neurons_per_layer[0], num_neurons_per_layer[1]),
            *classic_block(num_neurons_per_layer[1], num_neurons_per_layer[2]),
            *classic_block(num_neurons_per_layer[2], num_neurons_per_layer[3]),
            *classic_block(num_neurons_per_layer[3], num_neurons_per_layer[4], normalize=False, activation=nn.Tanh())
        )

    def forward(self, latent_vector_batch):
        img_batch_flattened = self.net(latent_vector_batch)
        return img_batch_flattened.view(img_batch_flattened.shape[0], 1, *self.generated_img_shape)


class DiscriminatorNet(torch.nn.Module):

    def __init__(self, img_shape=(MNIST_IMG_SIZE, MNIST_IMG_SIZE)):
        super().__init__()
        num_neurons_per_layer = [img_shape[0] * img_shape[1], 512, 256, 1]

        self.net = nn.Sequential(
            *classic_block(num_neurons_per_layer[0], num_neurons_per_layer[1], normalize=False),
            *classic_block(num_neurons_per_layer[1], num_neurons_per_layer[2], normalize=False),
            *classic_block(num_neurons_per_layer[2], num_neurons_per_layer[3], normalize=False, activation=nn.Sigmoid())
        )

    def forward(self, img_batch):
        img_batch_flattened = img_batch.view(img_batch.shape[0], -1)
        return self.net(img_batch_flattened)
