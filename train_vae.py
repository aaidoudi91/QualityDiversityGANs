""" Pre-trains a VAE whose decoder is the fixed starter Generator (Unbalanced GANs, Ham et al. 2020).
Saves the decoder as checkpoints/vae/G_init.pth, to be used as the GAN's initial generator (--init_g). """

import argparse
import csv
import json
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.utils import save_image

from model import Generator
from vae import VAE, vae_loss, init_normal


def get_device():
    """ Returns the best available device: CUDA, then Apple MPS, then CPU. """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """ Seeds Python, NumPy and PyTorch RNGs for reproducible runs. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist(root, train, device):
    """ Loads MNIST as a flat (N, 784) tensor in [-1, 1], same preprocessing as train.py. """
    ds = datasets.MNIST(root=root, train=train, download=True)
    x = ds.data.float().div(255).sub(0.5).div(0.5).view(-1, 784)
    return x.to(device)


def save_grid(imgs, path, nrow=8):
    """ Saves a batch of flat images in [-1, 1] as a PNG grid. """
    save_image((imgs.view(-1, 1, 28, 28) + 1) / 2, path, nrow=nrow)


@torch.no_grad()
def dump_samples(model, x_test, z_fixed, out_dir, tag):
    """ Saves two grids: test images with their reconstructions, and decodings of fixed prior samples. """
    model.eval()
    rec = model.decoder(model.encoder(x_test[:32])[0])  # reconstruct from the posterior mean
    save_grid(torch.cat([x_test[:32], rec]), os.path.join(out_dir, f"recon_{tag}.png"))
    save_grid(model.decoder(z_fixed), os.path.join(out_dir, f"prior_samples_{tag}.png"))
    model.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-train the VAE whose decoder is the fixed Generator.")
    p.add_argument("--steps", type=int, default=50000, help="Optimisation steps (paper: ~50k).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--kl_weight", type=float, default=1.0, help="beta in recon + beta*KL.")
    p.add_argument("--init", choices=["default", "normal002"], default="normal002",
                   help="normal002 = paper init N(0, 0.02); default = PyTorch init.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--sample_every", type=int, default=10000)
    p.add_argument("--out_dir", type=str, default="checkpoints/vae_pretrain")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    data_root = os.getenv("DATA", "data")
    x_train = load_mnist(data_root, True, device)
    x_test = load_mnist(data_root, False, device)
    n = x_train.shape[0]

    model = VAE(latent_dim=100, data_dim=784).to(device)
    if args.init == "normal002":
        model.apply(init_normal)
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Fixed latent codes, so prior samples are comparable across checkpoints.
    z_fixed = torch.randn(64, 100, generator=torch.Generator().manual_seed(1234)).to(device)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    history, step = [], 0
    run = torch.zeros(3, device=device)  # running sums of (loss, recon, kl) between two logs
    model.train()
    while step < args.steps:
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            x = x_train[perm[i:i + args.batch_size]]
            x_rec, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_rec, x, mu, logvar, beta=args.kl_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            run += torch.stack([loss.detach(), recon.detach(), kl.detach()])
            if step % args.log_every == 0:
                l, r, k = (run / args.log_every).tolist()
                history.append((step, l, r, k))
                print(f"step {step:6d} | loss {l:8.2f} | recon {r:8.2f} | kl {k:6.2f}")
                run.zero_()
            if step % args.sample_every == 0 or step == args.steps:
                dump_samples(model, x_test, z_fixed, args.out_dir, f"{step:06d}")
            if step >= args.steps:
                break

    with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "recon", "kl"])
        w.writerows(history)

    model.eval()
    torch.save(model.state_dict(), os.path.join(args.out_dir, "vae_pretrain.pth"))
    g_path = os.path.join(args.out_dir, "G_init.pth")
    torch.save(model.decoder.state_dict(), g_path)  # decoder keys match Generator's, loads directly

    # Sanity check: the decoder weights load into a fresh Generator, outputs identical.
    G = Generator(g_output_dim=784).to(device)
    G.load_state_dict(torch.load(g_path, map_location=device))
    G.eval()
    with torch.no_grad():
        diff = (G(z_fixed) - model.decoder(z_fixed)).abs().max().item()
    print(f"Transfer check: max |G(z) - Dec(z)| = {diff:.2e}")
    print(f"Saved: {g_path}")
