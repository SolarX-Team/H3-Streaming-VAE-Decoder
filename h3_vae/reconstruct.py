#!/usr/bin/env python3
"""Encode a video with the official H3 VAE, decode it latent by latent, report speed and PSNR.

    python -m h3_vae.reconstruct --video in.mp4 --output stream.mp4 --official official.mp4

The video is area-resized and center-cropped to --size (1344x768 by default, the setting
the brightness tables were fitted at), then trimmed to whole causal chunks. --official also
decodes all latents at once with decode_base and compares the two. The report is printed
as JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import h3_vae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="streaming reconstruction, .mp4")
    parser.add_argument("--official", type=Path, help="also decode with decode_base, write it here and compare")
    parser.add_argument("--save-latents", type=Path, help="save the latents for decode_latents.py, .pt")
    parser.add_argument("--size", default="1344x768", help="WIDTHxHEIGHT to resize and crop to, or 'native'")
    parser.add_argument("--max-frames", type=int)
    h3_vae.add_decoder_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    vae = h3_vae.load_vae(args.h3_root, device)
    video, fps = h3_vae.read_video(args.video, args.max_frames)
    if args.size != "native":
        width, height = (int(v) for v in args.size.lower().split("x"))
        video = h3_vae.fit_size(video, width, height)
    source = h3_vae.trim(vae, video)
    del video
    with torch.inference_mode():
        latents = vae.encode_base(h3_vae.to_pixels(vae, source), False)
    torch.cuda.empty_cache()
    if args.save_latents is not None:
        args.save_latents.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"latents": latents.cpu(), "fps": fps}, args.save_latents)

    report = {
        "video": str(args.video),
        "frames": int(len(source)),
        "size": f"{source.shape[2]}x{source.shape[1]}",
        "latents": list(latents.shape),
        "variant": args.variant,
        "compile": not args.no_compile,
    }
    official = None
    if args.official is not None:
        official, seconds = h3_vae.official_decode(vae, latents)
        official = official[: len(source)]
        h3_vae.write_video(args.official, official, fps)
        report["official"] = {
            "decode_s": seconds,
            "fps": len(official) / seconds,
            "psnr_vs_source": h3_vae.psnr(source, official),
        }

    decoder = h3_vae.make_decoder(vae, args)
    frames, timing = h3_vae.stream_decode(vae, decoder, latents)
    frames = frames[: len(source)]
    h3_vae.write_video(args.output, frames, fps)
    report["stream"] = dict(timing, psnr_vs_source=h3_vae.psnr(source, frames))
    if official is not None:
        report["stream"]["psnr_vs_official"] = h3_vae.psnr(official, frames)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
