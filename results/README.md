# Results

Machine: M1 Pro (MPS), seed 0, 10,000 images generated per evaluation.

Local evaluation: `evaluation.py` (FID + k-NN precision/recall, k=3, Inception pool3 features,
20,000 real reference images, `png_clamp` conversion identical to generate.py).

| Folder | Description | Command |
|---|---|---|
| baseline_s0 | Repository GAN, 100 epochs | `python train_gan.py --name baseline_s0 --seed 0` |
| vae | VAE pre-training (50k steps) | `python train_vae.py` |
| vae_init_s0 | GAN initialized with VAE decoder | `python train_gan.py --name vae_init_s0 --seed 0 --init_g checkpoints/vae/G_init.pth` |
| base_e300_s0 | Baseline, 300 epochs, eval every 50 epochs | `python train_gan.py --name base_e300_s0 --seed 0 --epochs 300 --eval_every 50` |
| drop_b05_s0 | Baseline with 0.25 dropout in D and beta1=0.5 | `python train_gan.py --name drop_b05_s0 --seed 0 --d_dropout 0.25 --beta1 0.5 --eval_every 10 --log_steps 50` |
| vae_drop_b05_s0 | Same with VAE initialization | `python train_gan.py --name vae_drop_b05_s0 --seed 0 --init_g checkpoints/vae/G_init.pth --d_dropout 0.25 --beta1 0.5 --eval_every 10 --log_steps 50` |

Notes:
- base_e300_s0 reproduces baseline_s0 up to epoch 100.
- Evaluations at epoch 0 for runs without VAE are very slow (>20 min), they should be skipped with --no_eval_ep0.