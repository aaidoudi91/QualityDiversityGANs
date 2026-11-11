""" Local evaluation of a Generator: FID, k-NN precision/recall on Inception features, class-balance diagnostics.
Usage: python evaluation.py --g_path <G.pth> [--compare_modes]; also imported by train_gan*.py. """

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
from torchvision import datasets

from model import Generator


def get_device():
    """ Return CUDA, MPS or CPU, in that order of preference. """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_images01(x, mode):
    """ Map generator outputs in [-1, 1], shape (B, 784), to 8-bit-quantised images in [0, 1].
    png_clamp mimics the starter generate.py (save_image clips negatives); 'rescale' uses (x + 1) / 2. """
    if mode == "png_clamp":
        img = x.clamp(0, 1)
    elif mode == "rescale":
        img = (x + 1) / 2
    else:
        raise ValueError(mode)
    return (img * 255).round() / 255


def sqrtm_psd(sigma):
    """ Symmetric square root of a PSD matrix via eigendecomposition. """
    w, v = np.linalg.eigh(sigma)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def frechet_distance(mu1, sigma1, mu2, sigma2, sqrt_sigma2):
    """ FID between two Gaussians, with sqrt(sigma2) precomputed.
    Uses tr sqrt(S1 S2) = sum sqrt(eig(S2^1/2 S1 S2^1/2)), a symmetric problem (faster than scipy sqrtm). """
    diff = mu1 - mu2
    m = sqrt_sigma2 @ sigma1 @ sqrt_sigma2
    eig = np.linalg.eigvalsh((m + m.T) / 2)
    tr_covmean = np.sqrt(np.clip(eig, 0, None)).sum()
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class SmallCNN(nn.Module):
    """ Small MNIST classifier, used only for diagnostics (class frequencies, confidence). """

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        """ Logits for flattened images in [0, 1], shape (B, 784) -> (B, 10). """
        x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def pairwise_dist(a, b):
    """ Euclidean distance matrix between rows of a and b, via the dot-product expansion. """
    d2 = (a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1)[None] - 2 * a @ b.T).clamp_min(0)
    return d2.sqrt()


def knn_radii(feats, k, bs=1000):
    """ Distance from each point to its k-th nearest neighbour in the same set (the manifold radii). """
    out = []
    for i in range(0, feats.shape[0], bs):
        d = pairwise_dist(feats[i:i + bs], feats)
        out.append(torch.topk(d, k + 1, dim=1, largest=False).values[:, -1])  # k+1 skips the point itself
    return torch.cat(out)


def coverage(queries, refs, ref_radii, bs=1000):
    """ Fraction of query points lying inside at least one reference ball (center refs, radius ref_radii). """
    hits = []
    for i in range(0, queries.shape[0], bs):
        d = pairwise_dist(queries[i:i + bs], refs)
        hits.append((d <= ref_radii[None]).any(dim=1))
    return torch.cat(hits).float().mean().item()


class Evaluator:
    """ FID + k-NN precision/recall (Kynkaanniemi et al. style) on Inception pool features,
    plus class-balance diagnostics. Real features and the classifier are computed once and cached. """

    def __init__(self, device, data_root="data", n_real=20000, k=3, real_seed=0, cache_dir="eval_cache"):
        self.device, self.k, self.cache_dir = device, k, cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

        # Fixed random subset of the MNIST train set as the real reference
        ds = datasets.MNIST(data_root, train=True, download=True)
        x = ds.data.float().div(255).view(-1, 784)
        idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(real_seed))[:n_real]
        path = os.path.join(cache_dir, f"real_feats_{n_real}_{real_seed}.pt")
        if os.path.exists(path):
            self.real_feats = torch.load(path)
        else:
            print(f"Computing real Inception features ({n_real} images), one-time cost...")
            self.real_feats = self.features(x[idx])
            torch.save(self.real_feats, path)
        r = self.real_feats.double().numpy()
        self.mu_r, self.sigma_r = r.mean(0), np.cov(r, rowvar=False)
        self.sqrt_sigma_r = sqrtm_psd(self.sigma_r)
        self.clf = self._get_classifier(data_root)

    def _get_classifier(self, data_root):
        """ Load the cached SmallCNN, or train it (3 epochs on MNIST train) and cache it. """
        path = os.path.join(self.cache_dir, "mnist_cnn.pth")
        clf = SmallCNN().to(self.device)
        if os.path.exists(path):
            clf.load_state_dict(torch.load(path, map_location=self.device))
            return clf.eval()
        ds = datasets.MNIST(data_root, train=True, download=True)
        x = ds.data.float().div(255).view(-1, 784).to(self.device)
        y = ds.targets.to(self.device)
        opt = torch.optim.Adam(clf.parameters(), 1e-3)
        g = torch.Generator().manual_seed(0)  # local generator: leaves the global RNG untouched
        for _ in range(3):
            perm = torch.randperm(x.shape[0], generator=g).to(self.device)
            for i in range(0, x.shape[0], 128):
                idx = perm[i:i + 128]
                loss = F.cross_entropy(clf(x[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        torch.save(clf.state_dict(), path)
        return clf.eval()

    @torch.no_grad()
    def features(self, imgs01, bs=250):
        """ Inception pool features (2048-d, on CPU) for flattened grayscale images in [0, 1]. """
        out = []
        for i in tqdm(range(0, imgs01.shape[0], bs), desc="Inception features", leave=False):
            b = imgs01[i:i + bs].to(self.device).view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
            out.append(self.inception(b)[0].squeeze(-1).squeeze(-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def evaluate(self, G, n=10000, seed=0, mode="png_clamp", bs=1000):
        """ Generate n images from G (fixed latent seed) and return FID, precision, recall,
        classifier confidence, class entropy (normalised to [0, 1]) and per-class frequencies. """
        G.eval()
        gen = torch.Generator().manual_seed(seed)
        imgs = []
        for i in range(0, n, bs):
            z = torch.randn(min(bs, n - i), 100, generator=gen).to(self.device)
            imgs.append(to_images01(G(z), mode).cpu())
        imgs = torch.cat(imgs)

        f = self.features(imgs)
        fd = f.double().numpy()
        fid = frechet_distance(fd.mean(0), np.cov(fd, rowvar=False), self.mu_r, self.sigma_r, self.sqrt_sigma_r)

        # Precision: fake points inside the real manifold; recall: real points inside the fake manifold
        n_pr = min(n, self.real_feats.shape[0])
        fake = f[:n_pr].to(self.device)
        real = self.real_feats[:n_pr].to(self.device)
        precision = coverage(fake, real, knn_radii(real, self.k))
        recall = coverage(real, fake, knn_radii(fake, self.k))

        probs = torch.cat([F.softmax(self.clf(imgs[i:i + 1000].to(self.device)), 1).cpu()
                           for i in range(0, n, 1000)])
        freq = torch.bincount(probs.argmax(1), minlength=10).float() / n
        ent = -(freq[freq > 0] * freq[freq > 0].log()).sum().item() / np.log(10)
        return {
            "fid": float(fid), "precision": precision, "recall": recall,
            "clf_conf": probs.max(1).values.mean().item(),
            "class_entropy": ent, "class_freq": [round(v, 4) for v in freq.tolist()],
        }


def load_generator(path, device):
    """ Load a Generator state_dict (tolerating a DataParallel 'module.' prefix) in eval mode. """
    G = Generator(g_output_dim=784).to(device)
    sd = torch.load(path, map_location=device)
    G.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    return G.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--g_path", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["png_clamp", "rescale"], default="png_clamp")
    ap.add_argument("--compare_modes", action="store_true", help="Evaluate both image conversions.")
    ap.add_argument("--n_real", type=int, default=20000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--data_root", default=os.getenv("DATA", "data"))
    a = ap.parse_args()

    dev = get_device()
    ev = Evaluator(dev, a.data_root, a.n_real, a.k)
    G = load_generator(a.g_path, dev)
    for m in (["png_clamp", "rescale"] if a.compare_modes else [a.mode]):
        print(m, json.dumps(ev.evaluate(G, a.n, a.seed, m)))
