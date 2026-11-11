# VAE Pre-training and Kernel Herding on MNIST

Léo Gruchociak · Mahmoud Ait-selam · Aaron Aidoudi

We study generation quality and diversity while keeping the supplied
generator architecture fixed.

Two approaches are investigated:
- VAE pre-training to initialise the GAN generator.
- Kernel herding to select generated samples after training.

The full methodology, comparisons and limitations are in [report.pdf](report.pdf).

## Main results

Selected local results reported in the report, using 10,000 generated images:

| Configuration | FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|
| Epoch-150 GAN, random selection | 24.362 | 0.549 | 0.371 |
| Epoch-150 GAN + kernel herding | 12.414 | 0.612 | 0.375 |
| Joint GAN/VAE-GAN herding, 50/50 pool | 8.132 | 0.582 | 0.474 |


## Repository guide

- `model.py`: supplied generator and discriminator architectures.
- `vae.py`, `train_vae.py`: VAE and decoder pre-training.
- `train_gan.py`: GAN training, optional VAE initialisation and D regularisation.
- `evaluation.py`, `evaluate_samples_folder.py`: local FID, precision/recall and class diagnostics.
- `herding_ablation.py`: random, attraction-only and herding comparisons.
- `kernel_herding*.py`, `kh_overnight.py`: exploratory selection scripts.
- `results/`: saved configurations, training histories, sample grids and herding ablation results.
- `generate.py`, `checkpoints/`: generation entry point, GAN weights and saved herding indices.
- `train.py`, `utils.py`, `scripts/`: starter training utilities and Slurm scripts.

## Generate the submitted samples

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python generate.py
```

This produces 10,000 grayscale PNGs in `samples/`, replacing existing PNGs. It loads `checkpoints/G.pth` and replays the saved GAN-only herding selection from `checkpoints/kh_selected_ids.npy` on CPU.

It does not recompute herding or generate the joint GAN/VAE-GAN result.

## Training and local evaluation

Install the evaluation dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Example commands for the original VAE/GAN comparison:

```bash
# Baseline GAN
python train_gan.py --name baseline_s0 --seed 0 --no_eval_ep0

# VAE pre-training and decoder export
python train_vae.py --seed 0 --out_dir checkpoints/vae

# GAN initialised with the pretrained decoder
python train_gan.py --name vae_init_s0 --seed 0 \
  --init_g checkpoints/vae/G_init.pth

# Evaluate the supplied GAN checkpoint
python evaluation.py --g_path checkpoints/G.pth
```

Training outputs are saved under `runs/<name>/`. MNIST is downloaded automatically; the data location defaults to `data/` and can be set with the `DATA` environment variable.

Training selects CUDA, Apple MPS or CPU automatically. Local evaluation uses Inception-2048 features, 20,000 real MNIST training images for FID, and 10,000 real images for k-NN precision/recall with k = 3.

## References

Based on the course starter repository, using PyTorch/torchvision and pytorch-fid.

- H. Ham, T. J. Jun and D. Kim. Unbalanced GANs: Pre-training the Generator of Generative Adversarial Network using Variational Autoencoder. arXiv:2002.02112, 2020.
- Y. Chen, M. Welling and A. Smola. Super-samples from kernel herding. UAI, pp. 109–116, 2010.