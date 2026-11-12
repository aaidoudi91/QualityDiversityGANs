""" GAN training on MNIST with the fixed starter Generator (same losses as the starter utils.py).
Optional VAE-decoder init, D dropout, separate G/D learning rates, periodic FID/P/R eval and fine-grained logs. """

import argparse
import csv
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.utils import save_image

from gan.evaluation import Evaluator, get_device, to_images01
from gan.model import Discriminator, Generator


class DropoutDiscriminator(Discriminator):
    """ Starter Discriminator (same layers and state_dict keys) with dropout after each hidden layer,
    as in the paper. """

    def __init__(self, d_input_dim, p):
        super().__init__(d_input_dim)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.drop(F.leaky_relu(self.fc1(x), 0.2))
        x = self.drop(F.leaky_relu(self.fc2(x), 0.2))
        x = self.drop(F.leaky_relu(self.fc3(x), 0.2))
        return torch.sigmoid(self.fc4(x))


def set_seed(seed):
    """ Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducible runs. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_step(x, G, D, G_opt, D_opt, criterion, device):
    """ One D step then one G step (non-saturating BCE), same maths as utils.D_train + utils.G_train.
    Returns detached (d_loss, g_loss, mean D(x), mean D(G(z))) so logging does not force a device sync. """
    b = x.shape[0]
    ones = torch.ones(b, 1, device=device)
    zeros = torch.zeros(b, 1, device=device)

    # Discriminator step: G is not updated, so its output is detached
    D_opt.zero_grad(set_to_none=True)
    d_real = D(x)
    loss_real = criterion(d_real, ones)
    z = torch.randn(b, 100, device=device)
    with torch.no_grad():
        x_fake = G(z)
    d_fake = D(x_fake)
    loss_fake = criterion(d_fake, zeros)
    d_loss = loss_real + loss_fake
    d_loss.backward()
    D_opt.step()

    # Generator step on fresh noise: G is trained to make D output "real"
    G_opt.zero_grad(set_to_none=True)
    z = torch.randn(b, 100, device=device)
    g_loss = criterion(D(G(z)), ones)
    g_loss.backward()
    G_opt.step()
    return d_loss.detach(), g_loss.detach(), d_real.detach().mean(), d_fake.detach().mean()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GAN training: optional G init, D dropout, TTUR, fine-grained logs.")
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_g", type=float, default=None, help="Defaults to --lr.")
    p.add_argument("--lr_d", type=float, default=None, help="Defaults to --lr.")
    p.add_argument("--beta1", type=float, default=0.9, help="0.9 = Adam default (train.py); paper uses 0.5.")
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_g", type=str, default=None)
    p.add_argument("--d_dropout", type=float, default=0.0, help="0 = starter Discriminator.")
    p.add_argument("--eval_every", type=int, default=10, help="0 = only at the last epoch.")
    p.add_argument("--eval_n", type=int, default=10000)
    p.add_argument("--eval_mode", choices=["png_clamp", "rescale"], default="png_clamp")
    p.add_argument("--n_real", type=int, default=20000)
    p.add_argument("--no_eval_ep0", action="store_true")
    p.add_argument("--no_eval", action="store_true", help="Skip all evaluations (still saves G_final).")
    p.add_argument("--log_steps", type=int, default=0, help="Window (in steps) for fine logs; 0 = off.")
    p.add_argument("--log_until", type=int, default=2814, help="Fine logs for the first N steps (~3 epochs).")
    p.add_argument("--data_root", default=os.getenv("DATA", "data"))
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Built before seeding: the Evaluator has its own RNG usage and must not shift the training stream
    evaluator = None if args.no_eval else Evaluator(device, args.data_root, args.n_real)
    set_seed(args.seed)

    # Whole dataset kept on the device, same normalisation as train.py (pixels in [-1, 1])
    ds = datasets.MNIST(root=args.data_root, train=True, download=True)
    X = ds.data.float().div(255).sub(0.5).div(0.5).view(-1, 784).to(device)
    N = X.shape[0]

    # D is created after G in every run, so a given seed gives the same D init across configurations
    G = Generator(g_output_dim=784).to(device)
    D = (DropoutDiscriminator(784, args.d_dropout) if args.d_dropout > 0 else Discriminator(784)).to(device)
    if args.init_g:
        G.load_state_dict(torch.load(args.init_g, map_location=device))
        print(f"G initialised from {args.init_g}")

    criterion = nn.BCELoss()
    betas = (args.beta1, args.beta2)
    G_opt = optim.Adam(G.parameters(), lr=args.lr_g or args.lr, betas=betas)
    D_opt = optim.Adam(D.parameters(), lr=args.lr_d or args.lr, betas=betas)
    z_fixed = torch.randn(64, 100, generator=torch.Generator().manual_seed(1234)).to(device)  # fixed grid noise

    hist_path = os.path.join(run_dir, "history.csv")
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "d_loss", "g_loss", "d_real", "d_fake", "fid", "precision",
                                "recall", "clf_conf", "class_entropy", "elapsed_s", "eval_s"])
    if args.log_steps:
        steps_path = os.path.join(run_dir, "steps.csv")
        with open(steps_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "d_loss", "g_loss", "d_real", "d_fake"])

    t0 = time.time()

    def do_eval(epoch, stats):
        """ Evaluate G (FID, P, R, class stats), append a row to history.csv, save G and a sample grid. """
        te = time.time()
        r = evaluator.evaluate(G, args.eval_n, seed=0, mode=args.eval_mode)
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, *stats, r["fid"], r["precision"], r["recall"], r["clf_conf"],
                                    r["class_entropy"], round(time.time() - t0, 1), round(time.time() - te, 1)])
        torch.save(G.state_dict(), os.path.join(run_dir, f"G_ep{epoch:03d}.pth"))
        with torch.no_grad():
            save_image(to_images01(G(z_fixed), args.eval_mode).view(-1, 1, 28, 28),
                       os.path.join(run_dir, f"samples_ep{epoch:03d}.png"), nrow=8)
        print(f"[eval ep {epoch:3d}] FID {r['fid']:.2f} | P {r['precision']:.3f} | R {r['recall']:.3f} "
              f"| conf {r['clf_conf']:.3f} | class-entropy {r['class_entropy']:.3f} | eval {time.time() - te:.0f}s")

    if not args.no_eval and not args.no_eval_ep0:
        do_eval(0, [float("nan")] * 4)  # quality of the initial G (random or VAE decoder)

    step = 0
    win, wn = torch.zeros(4, device=device), 0  # running sums for the fine-grained step logs
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(N, device=device)
        ep_sum, nb = torch.zeros(4, device=device), 0
        for i in range(0, N, args.batch_size):
            x = X[perm[i:i + args.batch_size]]
            stats = torch.stack(train_step(x, G, D, G_opt, D_opt, criterion, device))
            ep_sum += stats
            nb += 1
            step += 1
            if args.log_steps and step <= args.log_until:
                win += stats
                wn += 1
                if wn == args.log_steps:
                    with open(steps_path, "a", newline="") as f:
                        csv.writer(f).writerow([step, *(win / wn).tolist()])
                    win.zero_()
                    wn = 0
        ep_stats = (ep_sum / nb).tolist()  # single device sync per epoch
        print(f"epoch {epoch:3d} | D {ep_stats[0]:.3f} | G {ep_stats[1]:.3f} | D(x) {ep_stats[2]:.2f} "
              f"| D(G(z)) {ep_stats[3]:.2f} | {time.time() - t0:.0f}s")
        last = epoch == args.epochs
        periodic = args.eval_every > 0 and epoch % args.eval_every == 0
        if not args.no_eval and (periodic or last):
            do_eval(epoch, ep_stats)

    torch.save(G.state_dict(), os.path.join(run_dir, "G_final.pth"))
    torch.save(D.state_dict(), os.path.join(run_dir, "D_final.pth"))
    print("Done.")
