""" Variational Autoencoder whose decoder is the fixed starter Generator, used to pre-train the GAN generator.
Its decoder state_dict loads directly into model.Generator (same layers and keys). """

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Generator


class Encoder(nn.Module):
    """ MLP encoder (784-1024-512-256) with dropout, mirroring the Discriminator widths.
    Outputs the mean and log-variance of q(z|x). """

    def __init__(self, input_dim=784, latent_dim=100, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.drop = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.drop(F.leaky_relu(self.fc1(x), 0.2))
        h = self.drop(F.leaky_relu(self.fc2(h), 0.2))
        h = self.drop(F.leaky_relu(self.fc3(h), 0.2))
        return self.fc_mu(h), self.fc_logvar(h)


class VAE(nn.Module):
    """ Encoder plus a decoder that is the fixed Generator itself.
    The latent dimension must match the GAN prior (100). """

    def __init__(self, latent_dim=100, data_dim=784):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(data_dim, latent_dim)
        self.decoder = Generator(g_output_dim=data_dim)

    def forward(self, x):
        """ Returns (reconstruction, mu, logvar) using the reparameterisation trick. """
        mu, logvar = self.encoder(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar


def vae_loss(x_rec, x, mu, logvar, beta=1.0, eps=1e-6):
    """ Negative ELBO: pixel-wise BCE (on images rescaled to [0, 1]) + beta * KL to N(0, I).
    Returns (total, recon, kl), each averaged over the batch. """
    b = x.shape[0]
    x01 = (x + 1) / 2  # data and tanh outputs live in [-1, 1]
    r01 = ((x_rec + 1) / 2).clamp(eps, 1 - eps)
    recon = F.binary_cross_entropy(r01, x01, reduction="sum") / b
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return recon + beta * kl, recon, kl


def init_normal(m, std=0.02):
    """ Weight init N(0, std) with zero bias for Linear layers (paper setting); use with model.apply. """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, std)
        nn.init.zeros_(m.bias)
