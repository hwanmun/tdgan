""" Define models for generator and discriminator. """

import torch
from torch import nn

class FCGenerator(nn.Module):
    """ DL generator made of fully connected layers. """
    def __init__(self, latent_dim, feat_dim, hdims=[]):
        super().__init__()
        dims = [latent_dim] + hdims + [feat_dim]
        self.layers = []
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                self.layers.append(nn.BatchNorm1d(dims[i+1]))
                self.layers.append(nn.ReLU())
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FCDiscriminator(nn.Module):
    """ DL discriminator made of fully connected layers. """
    def __init__(self, feat_dim, hdims=[]):
        super().__init__()
        dims = [feat_dim] + hdims + [1]
        self.layers = []
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                self.layers.append(nn.BatchNorm1d(dims[i+1]))
                self.layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
