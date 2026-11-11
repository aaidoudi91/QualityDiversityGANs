"""Evaluate 28x28 PNG samples with the *same* Inception FID and k-NN P/R
protocol as this group's evaluation.py, without regenerating GAN images.

Install optional evaluation dependencies: python -m pip install -r requirements-dev.txt
Run from repository root:
  python evaluate_samples_folder.py --samples_dir samples_random --output results/random_eval.json
  python evaluate_samples_folder.py --samples_dir samples_kh --output results/herding_eval.json

Uses only MNIST TRAIN for the reference data, as in evaluation.Evaluator.
No VAE or training is involved. This script doesn't modify project files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from evaluation import Evaluator, coverage, frechet_distance, get_device, knn_radii


def load_pngs(folder: Path, expected_count: int) -> torch.Tensor:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    paths = sorted(folder.glob("*.png"), key=lambda p: (not p.stem.isdecimal(), int(p.stem) if p.stem.isdecimal() else p.stem))
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} PNGs in {folder}; found {len(paths)}")

    data = np.empty((len(paths), 784), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as image:
            if image.size != (28, 28):
                raise ValueError(f"Expected 28x28, got {image.size}: {p}")
            if image.mode not in ("L", "1"):
                raise ValueError(f"Expected grayscale L/1 PNG, got {image.mode}: {p}")
            data[i] = np.asarray(image.convert("L"), dtype=np.uint8).reshape(784)
    return torch.from_numpy(data.copy()).float().div_(255)


@torch.no_grad()
def evaluate_folder(folder: Path, ev: Evaluator, n_expected: int) -> dict:
    imgs = load_pngs(folder, n_expected)
    # This is exactly evaluation.Evaluator.features(imgs), which reads [0,1]
    # pixel values and converts 1x28x28 -> 3x28x28 for Inception.
    fake_feats = ev.features(imgs)
    fake_np = fake_feats.double().numpy()
    fid = frechet_distance(fake_np.mean(axis=0), np.cov(fake_np, rowvar=False),
                           ev.mu_r, ev.sigma_r, ev.sqrt_sigma_r)

    # Match the group's Evaluator.evaluate(): PR uses first n samples
    # from both feature matrices, and k-neighbourhoods with k=ev.k.
    n_pr = min(len(imgs), ev.real_feats.shape[0])
    real = ev.real_feats[:n_pr].to(ev.device)
    fake = fake_feats[:n_pr].to(ev.device)
    precision = coverage(fake, real, knn_radii(real, ev.k))
    recall = coverage(real, fake, knn_radii(fake, ev.k))

    predictions = []
    confidences = []
    for start in range(0, len(imgs), 1000):
        probs = F.softmax(ev.clf(imgs[start:start + 1000].to(ev.device)), dim=1)
        predictions.append(probs.argmax(dim=1).cpu())
        confidences.append(probs.max(dim=1).values.cpu())
    counts = torch.bincount(torch.cat(predictions), minlength=10)
    freq = counts.float() / len(imgs)
    nonzero = freq[freq > 0]
    entropy = -(nonzero * nonzero.log()).sum().item() / math.log(10)

    return {
        "samples_dir": str(folder),
        "n_samples": len(imgs), "n_real": ev.real_feats.shape[0],
        "k": ev.k,
        "fid": float(fid),
        "precision": float(precision),
        "recall": float(recall),
        "clf_conf": float(torch.cat(confidences).mean()),
        "class_entropy": float(entropy),
        "class_freq": [round(v, 4) for v in freq.tolist()],
        "metric_note": "Inception-2048 FID and kNN P/R using evaluation.py's evaluator",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FID, precision and recall for existing 28x28 PNG folder")
    ap.add_argument("--samples_dir", type=Path, required=True)
    ap.add_argument("--expected", type=int, default=10000, help="Expected number of PNG files")
    ap.add_argument("--data_root", default=os.getenv("DATA", "data"))
    ap.add_argument("--cache_dir", default="eval_cache")
    ap.add_argument("--n_real", type=int, default=20000, help="Same default as evaluation.py")
    ap.add_argument("--k", type=int, default=3, help="Same default as evaluation.py")
    ap.add_argument("--real_seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.expected < 4 or args.n_real < 4 or not 1 <= args.k < min(args.expected, args.n_real):
        ap.error("Need >=4 images, >=4 real references, and 1 <= k < min(expected,n_real)")

    print("Loading evaluation features (on CPU this can be slow)...", flush=True)
    dev = get_device()
    print(f"Evaluation device: {dev}", flush=True)
    ev = Evaluator(dev, data_root=args.data_root, n_real=args.n_real, k=args.k,
                   real_seed=args.real_seed, cache_dir=args.cache_dir)
    result = evaluate_folder(args.samples_dir, ev, args.expected)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
