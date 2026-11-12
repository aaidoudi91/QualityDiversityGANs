"""DataLab A2: replay the *joint* GAN/VAE-GAN 50/50 Kernel Herding result.

Uses two fixed-architecture Generator checkpoints and the saved selection
mapping from a2_kh_mixed_gan_vae.py (seed=42, 20k candidates -> 10k images).
No MNIST, Inception, feature cache, or training data is needed at inference.

Required project-root files:
  model.py
  checkpoints/G.pth
  checkpoints/G_vae_finetuned.pth
  checkpoints/mixed_50_50_kh_selected_ids.npy
  checkpoints/mixed_50_50_candidate_source.npy
  checkpoints/mixed_50_50_candidate_index.npy

Run: python generate.py --batch_size 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gan.model import Generator


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
N_CANDIDATES = 20_000  # 20k original latent indices shared by both sources
N_SELECTED = 10_000
SEED = 42
# Both original candidate caches used generate_candidates(... batch_size=128).
# Replaying the same block shape avoids subtle batch-dependent rounding.
LATENT_CHUNK_SIZE = 128



def read_selection() -> tuple[np.ndarray, np.ndarray]:
    """Return output PNG position for each original GAN/VAE candidate index.

    The joint KH ids index a *mixed pool*, so they cannot be directly used
    as GAN or VAE source-pool indices.
    """
    kh_ids = np.load(CHECKPOINTS / "mixed_50_50_kh_selected_ids.npy", allow_pickle=False)
    sources = np.load(CHECKPOINTS / "mixed_50_50_candidate_source.npy", allow_pickle=False)
    original_indices = np.load(CHECKPOINTS / "mixed_50_50_candidate_index.npy", allow_pickle=False)

    if kh_ids.shape != (N_SELECTED,) or sources.shape != (N_CANDIDATES,) or original_indices.shape != (N_CANDIDATES,):
        raise ValueError("Wrong saved KH/source/index array shapes (expected 10k/20k/20k)")
    kh_ids = kh_ids.astype(np.int64)
    sources = sources.astype(np.int64)
    original_indices = original_indices.astype(np.int64)

    if not (np.all((0 <= kh_ids) & (kh_ids < N_CANDIDATES))
            and len(np.unique(kh_ids)) == N_SELECTED):
        raise ValueError("KH selection must contain 10,000 unique valid mixed-pool indices")
    if not (np.all(np.isin(sources, (0, 1)))
            and np.count_nonzero(sources == 0) == N_CANDIDATES // 2):
        raise ValueError("Expected exactly 10,000 GAN and 10,000 VAE candidates")
    if (original_indices.min() < 0 or original_indices.max() >= N_CANDIDATES
            or len(np.unique(original_indices)) != N_CANDIDATES):
        raise ValueError("Invalid mixed-pool original indices")

    # In the mixed pool, 0=GAN and 1=VAE. Map each selected original
    # latent index to its *output* position (0.png, ..., 9999.png).
    output_positions = [np.full(N_CANDIDATES, -1, dtype=np.int64) for _ in range(2)]
    for out_pos, mixed_id in enumerate(kh_ids):
        src = int(sources[mixed_id])
        original_id = int(original_indices[mixed_id])
        if output_positions[src][original_id] != -1:
            raise ValueError("Duplicate output source/original index")
        output_positions[src][original_id] = out_pos

    selected_gan = int(np.count_nonzero(output_positions[0] != -1))
    selected_vae = int(np.count_nonzero(output_positions[1] != -1))
    if (selected_gan, selected_vae) != (5029, 4971):
        raise ValueError(
            "Unexpected mixed selection: expected 5029 GAN / 4971 VAE, "
            f"found {selected_gan} / {selected_vae}. "
            "Check that the files are from gan10000_vae10000."
        )
    return output_positions[0], output_positions[1]


def load_generator(checkpoint: Path) -> Generator:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing generator checkpoint: {checkpoint}")
    gen = Generator(g_output_dim=784).to("cpu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Not a generator state dict: {checkpoint}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    gen.load_state_dict(state, strict=True)
    gen.eval()
    return gen


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--check_only", action="store_true", help="Check saved selection and checkpoints without writing PNGs")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")

    positions = read_selection()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    generators = [
        load_generator(CHECKPOINTS / "G.pth"),
        load_generator(CHECKPOINTS / "G_vae_finetuned.pth"),
    ]
    print("[OK] GAN + VAE-GAN checkpoints; joint KH 50/50 indices (5029 / 4971)", flush=True)
    if args.check_only:
        return

    output_dir = ROOT / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.png"):
        old_file.unlink()

    rng = torch.Generator(device="cpu").manual_seed(SEED)
    saved = 0
    # Reconstruct the 20k source-pool latent positions.  Every source used the
    # *same* seed-42 latent stream, but the joint pool partitioned the indices.
    for start in range(0, N_CANDIDATES, LATENT_CHUNK_SIZE):
        count = min(LATENT_CHUNK_SIZE, N_CANDIDATES - start)
        z_chunk = torch.randn(count, 100, generator=rng)
        for source_id, G in enumerate(generators):
            selected_in_chunk = np.flatnonzero(positions[source_id][start:start + count] >= 0)
            if len(selected_in_chunk) == 0:
                continue

            # Compute with the same full 128-candidate block as the cached
            # original run, even when only some are selected.
            batches = []
            for offset in range(0, count, min(args.batch_size, LATENT_CHUNK_SIZE)):
                z = z_chunk[offset:offset + min(args.batch_size, LATENT_CHUNK_SIZE)]
                batches.append(G(z).clamp(0, 1).cpu())
            images = torch.cat(batches, dim=0)

            for within_chunk in selected_in_chunk:
                source_index = start + int(within_chunk)
                final_index = int(positions[source_id][source_index])
                img_u8 = (images[int(within_chunk)].reshape(28, 28)
                          .mul(255).round().to(torch.uint8).numpy())
                Image.fromarray(img_u8, mode="L").save(output_dir / f"{final_index}.png")
                saved += 1
        if (start + count) % 5000 == 0 or start + count == N_CANDIDATES:
            print(f"Processed {start + count}/{N_CANDIDATES} latent indices; saved {saved}/{N_SELECTED} PNGs", flush=True)

    if saved != N_SELECTED:
        raise RuntimeError(f"Expected {N_SELECTED} PNGs, actually saved {saved}")
    print(f"[DONE] {saved} PNGs (28x28) in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
