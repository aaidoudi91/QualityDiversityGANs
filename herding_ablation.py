"""Ablation of Inception-feature kernel herding (A2, group 24).

Compares, on ONE fixed pool of 20,000 candidates from checkpoints/G.pth
(same latent stream as generate.py: CPU RNG seed 42, chunks of 256, clamp to [0,1]):
  random           10,000 candidates drawn uniformly (several seeds -> noise level)
  attraction_only  top-10,000 candidates by m(x) = mean_r k(x, r): no repulsion,
                   i.e. a pure "closest to real data" quality filter
  herding          greedy Eq. 8 of Chen et al. (2010), unique selection
  herding_stoch    same, but each step only scores a random subset of candidates
  submitted        the indices in checkpoints/kh_selected_ids.npy (sanity check)

Herding uses Inception-2048 features, an RBF kernel with the median-distance sigma,
and MNIST TRAIN reference images disjoint from the 20,000 used by evaluation.py.
Every selection is scored with evaluation.Evaluator (FID, k-NN P/R, class entropy).

Run from the repository root:
  python herding_ablation.py            # writes results/herding_ablation.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST

from evaluation import Evaluator, coverage, frechet_distance, get_device, knn_radii
from kernel_herding import estimate_sigma, generate_candidates, kernel_target_mean


def quantise(x: torch.Tensor) -> torch.Tensor:
    """Same values as the saved 8-bit PNGs."""
    return (x * 255).round() / 255


@torch.no_grad()
def score(ev: Evaluator, imgs: torch.Tensor, feats: torch.Tensor) -> dict:
    """Mirror of evaluate_samples_folder.evaluate_folder, from tensors."""
    fd = feats.double().numpy()
    fid = frechet_distance(fd.mean(0), np.cov(fd, rowvar=False), ev.mu_r, ev.sigma_r, ev.sqrt_sigma_r)
    n_pr = min(len(imgs), ev.real_feats.shape[0])
    real = ev.real_feats[:n_pr].to(ev.device)
    fake = feats[:n_pr].to(ev.device)
    precision = coverage(fake, real, knn_radii(real, ev.k))
    recall = coverage(real, fake, knn_radii(fake, ev.k))
    preds = torch.cat([F.softmax(ev.clf(imgs[i:i + 1000].to(ev.device)), 1).argmax(1).cpu()
                       for i in range(0, len(imgs), 1000)])
    freq = torch.bincount(preds, minlength=10).float() / len(imgs)
    nz = freq[freq > 0]
    entropy = -(nz * nz.log()).sum().item() / math.log(10)
    return {"fid": round(float(fid), 2), "precision": round(precision, 3), "recall": round(recall, 3),
            "class_entropy": round(entropy, 3), "class_freq": [round(v, 4) for v in freq.tolist()]}


@torch.no_grad()
def herd(cand: torch.Tensor, m: torch.Tensor, n_select: int, sigma: float,
         subset: int | None = None, seed: int = 0) -> np.ndarray:
    """Greedy Eq. 8 with unique selection; if subset is set, each step scores only
    `subset` random unused candidates (stochastic variant)."""
    n = len(cand)
    norms2 = cand.square().sum(1)
    used = torch.zeros(n, dtype=torch.bool, device=cand.device)
    penalty = torch.zeros(n, device=cand.device)
    ids = np.empty(n_select, dtype=np.int64)
    gen = torch.Generator().manual_seed(seed)
    for t in range(n_select):
        s = (m - penalty / (t + 1)).masked_fill(used, float("-inf"))
        if subset is None:
            idx = int(s.argmax())
        else:
            free = (~used).nonzero().squeeze(1).cpu()
            pick = free[torch.randperm(len(free), generator=gen)[:subset]]
            idx = int(pick[int(s[pick.to(cand.device)].argmax())])
        ids[t] = idx
        used[idx] = True
        d2 = (norms2 + norms2[idx] - 2.0 * cand @ cand[idx]).clamp_min_(0)
        penalty.add_((-d2 / (2.0 * sigma ** 2)).exp_())
        if (t + 1) % 2000 == 0:
            print(f"  step {t + 1}/{n_select}", flush=True)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/G.pth")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--n_candidates", type=int, default=20000)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--n_reference", type=int, default=10000)
    ap.add_argument("--subset", type=int, default=1000, help="candidates scored per step, stochastic variant")
    ap.add_argument("--random_seeds", type=int, default=5)
    ap.add_argument("--skip_full", action="store_true", help="skip the full (slow) herding run")
    ap.add_argument("--output", default="results/herding_ablation.json")
    a = ap.parse_args()

    dev = get_device()
    print(f"Device: {dev}", flush=True)
    ev = Evaluator(dev, a.data_root)  # same protocol as evaluation.py (20k real, k=3, seed 0)

    # 1) Candidate pool, identical to generate.py
    pool = quantise(generate_candidates(Path(a.checkpoint), a.n_candidates, 256, torch.device("cpu"), 42))
    print("Inception features of the pool...", flush=True)
    pool_f = ev.features(pool)

    # 2) Reference: MNIST train images NOT used by the evaluator
    train = MNIST(a.data_root, train=True, download=True).data.float().div(255).view(-1, 784)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
    ref = train[perm[20000:20000 + a.n_reference]]
    print("Inception features of the reference...", flush=True)
    ref_f = ev.features(ref)

    cand = pool_f.to(dev)
    sigma = estimate_sigma(torch.cat((pool_f[:512], ref_f[:512])), 42)
    m = kernel_target_mean(cand, ref_f.to(dev), sigma)
    print(f"sigma = {sigma:.3f}", flush=True)

    res = {"config": vars(a) | {"sigma": sigma}, "runs": {}, "time_s": {}}

    def add(name, ids):
        ids = torch.as_tensor(ids)
        res["runs"][name] = score(ev, pool[ids], pool_f[ids]) | {"n_unique": int(len(ids.unique()))}
        print(name, json.dumps(res["runs"][name]), flush=True)

    # random baseline, several seeds: gives the noise level of each metric
    rnd = []
    for s in range(a.random_seeds):
        ids = np.random.default_rng(s).choice(a.n_candidates, a.n_samples, replace=False)
        add(f"random_s{s}", ids)
        rnd.append(res["runs"][f"random_s{s}"])
    res["random_mean_std"] = {k: [round(float(np.mean([r[k] for r in rnd])), 3),
                                  round(float(np.std([r[k] for r in rnd])), 3)]
                              for k in ("fid", "precision", "recall", "class_entropy")}
    print("random mean/std", res["random_mean_std"], flush=True)

    t0 = time.time()
    add("attraction_only", torch.topk(m, a.n_samples).indices.cpu().numpy())
    res["time_s"]["attraction_only"] = round(time.time() - t0, 1)

    t0 = time.time()
    print(f"Stochastic herding (subset={a.subset})...", flush=True)
    add("herding_stoch", herd(cand, m, a.n_samples, sigma, subset=a.subset))
    res["time_s"]["herding_stoch"] = round(time.time() - t0, 1)

    if not a.skip_full:
        t0 = time.time()
        print("Full herding...", flush=True)
        add("herding", herd(cand, m, a.n_samples, sigma))
        res["time_s"]["herding"] = round(time.time() - t0, 1)

    sub = Path("checkpoints/kh_selected_ids.npy")
    if sub.is_file():
        add("submitted", np.load(sub))

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
