
"""A2 inference: replay a precomputed Kernel Herding selection.

The indices were obtained with:
- GAN checkpoint: checkpoints/G.pth
- 20,000 candidates, seed 42
- Inception-2048 RBF Kernel Herding
- alpha=1, sigma_scale=1, unique selection

No MNIST data or Inception network is needed at inference time.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import Generator


N_CANDIDATES = 20_000
N_SAMPLES = 10_000
SEED = 42

# Match the latent-generation chunk size used in kh_overnight.py.
LATENT_CHUNK_SIZE = 256


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")

    # Read indices computed during the Kernel Herding experiment.
    ids = np.load(
        "checkpoints/kh_selected_ids.npy",
        allow_pickle=False,
    )

    assert ids.shape == (N_SAMPLES,)
    assert len(np.unique(ids)) == N_SAMPLES
    assert ids.min() >= 0
    assert ids.max() < N_CANDIDATES

    # Map each candidate index to its output PNG number.
    output_position = np.full(
        N_CANDIDATES,
        -1,
        dtype=np.int64,
    )
    output_position[ids] = np.arange(N_SAMPLES)

    # Use CPU to reproduce the original candidate generation.
    device = torch.device("cpu")

    G = Generator(g_output_dim=784).to(device)

    state = torch.load(
        "checkpoints/G.pth",
        map_location=device,
        weights_only=True,
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    state = {
        k.removeprefix("module."): v
        for k, v in state.items()
    }

    G.load_state_dict(state, strict=True)
    G.eval()

    output_dir = Path("samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.glob("*.png"):
        old_file.unlink()

    rng = torch.Generator(device="cpu").manual_seed(SEED)

    saved = 0

    # Reconstruct exactly the candidate latent stream.
    for start in range(0, N_CANDIDATES, LATENT_CHUNK_SIZE):
        count = min(LATENT_CHUNK_SIZE, N_CANDIDATES - start)

        z_chunk = torch.randn(
            count,
            100,
            generator=rng,
        )

        # Allow any requested inference batch size while keeping
        # latent generation independent of that batch size.
        for offset in range(0, count, args.batch_size):
            z = z_chunk[offset:offset + args.batch_size]

            images = G(z).clamp(0, 1).cpu()

            for j, image in enumerate(images):
                candidate_index = start + offset + j
                output_index = output_position[candidate_index]

                if output_index < 0:
                    continue

                png = (
                    image.reshape(28, 28)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .numpy()
                )

                Image.fromarray(png, mode="L").save(
                    output_dir / f"{output_index}.png"
                )

                saved += 1

        if (start + count) % 5000 == 0:
            print(
                f"Processed {start + count}/{N_CANDIDATES} "
                f"GAN candidates",
                flush=True,
            )

    assert saved == N_SAMPLES, (
        f"Expected {N_SAMPLES} PNGs, got {saved}"
    )

    print(
        f"Saved {saved} Kernel Herding images to {output_dir}/",
        flush=True,
    )


if __name__ == "__main__":
    main()
