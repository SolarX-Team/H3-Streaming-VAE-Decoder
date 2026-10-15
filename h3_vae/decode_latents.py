#!/usr/bin/env python3
"""Decode saved H3 latents latent by latent and write the frames as a video.

    python -m h3_vae.decode_latents --latents latents.pt --output stream.mp4 --fps 24

--latents is a .pt holding a [1, C, T, h, w] tensor, or a dict with it under "latents" (and
optionally "fps"), as reconstruct.py --save-latents writes it; a .safetensors or .npy works
too. In a file with several tensors, --key picks one. The latents go in as decode_base takes
them; DiT latents, saved as (z - latents_mean) / latents_std, need --normalized. --official
also decodes them all at once with decode_base and compares the two. The report is printed
as JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import h3_vae


def load_latents(path: Path, key: str | None = None) -> tuple[torch.Tensor, float | None]:
    fps = None
    if path.suffix == ".npy":
        latents = torch.from_numpy(np.load(path))
    elif path.suffix == ".safetensors":
        from safetensors.torch import load_file

        latents = load_file(str(path))
    else:
        latents = torch.load(path, map_location="cpu")
    if isinstance(latents, dict):
        fps = latents.get("fps")
        fps = None if fps is None else float(fps)
        tensors = {name: value for name, value in latents.items() if torch.is_tensor(value) and value.dim() in (4, 5)}
        if key is None:
            key = next(iter(tensors)) if len(tensors) == 1 else "latents"
        if key not in tensors:
            raise KeyError(f"no latents {key!r} in {path}; pass --key, one of {sorted(tensors)}")
        latents = tensors[key]
    if latents.dim() == 4:
        latents = latents.unsqueeze(0)
    if latents.dim() != 5 or latents.shape[0] != 1:
        raise ValueError(f"expected latents [1, C, T, h, w], got {list(latents.shape)}")
    return latents, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latents", type=Path, required=True, help=".pt, .safetensors or .npy")
    parser.add_argument("--key", help="the tensor to decode in a file with several (default: the only one, else 'latents')")
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="the latents are DiT latents, (z - latents_mean) / latents_std: multiply by std and add mean first",
    )
    parser.add_argument("--output", type=Path, required=True, help="streaming decode, .mp4")
    parser.add_argument("--fps", type=float, help="default: the fps saved with the latents, else 24")
    parser.add_argument("--official", type=Path, help="also decode with decode_base, write it here and compare")
    h3_vae.add_decoder_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    latents, saved_fps = load_latents(args.latents, args.key)
    if args.normalized:
        mean, std = h3_vae.latent_stats(args.h3_root)
        latents = latents.float() * std + mean
    fps = args.fps or saved_fps or 24.0
    vae = h3_vae.load_vae(args.h3_root, device)
    latents = latents.to(device)
    report = {
        "latents": list(latents.shape),
        "normalized": args.normalized,
        "variant": args.variant,
        "compile": not args.no_compile,
    }

    official = None
    if args.official is not None:
        official, seconds = h3_vae.official_decode(vae, latents)
        h3_vae.write_video(args.official, official, fps)
        report["official"] = {"frames": int(len(official)), "decode_s": seconds, "fps": len(official) / seconds}

    decoder = h3_vae.make_decoder(vae, args)
    frames, timing = h3_vae.stream_decode(vae, decoder, latents)
    h3_vae.write_video(args.output, frames, fps)
    report["stream"] = timing
    if official is not None:
        report["stream"]["psnr_vs_official"] = h3_vae.psnr(official, frames)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
