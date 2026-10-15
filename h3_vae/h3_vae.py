#!/usr/bin/env python3
"""The official MiniMax H3 video VAE, videos in and out of its pixel space, and a timed stream.

load_vae needs a MiniMax-H3 download (Hugging Face MiniMaxAI/MiniMax-H3 or ModelScope
MiniMax/MiniMax-H3) with FL2VA/video_vae in it: the VAE code is imported from there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .stream_decoder import StreamingDecoder

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "v1": dict(first_refresh=False, compensate=False),
    "v2": dict(first_refresh=False, compensate=True),
    "v3": dict(first_refresh=True, compensate=True),
}


def find_h3_root(path: str | Path | None = None) -> Path:
    """The MiniMax-H3 download: `path`, then $H3_ROOT, then a parent folder that already contains FL2VA."""
    for candidate in (path, os.environ.get("H3_ROOT"), HERE.parent, HERE.parent.parent):
        if candidate and (Path(candidate) / "FL2VA" / "video_vae" / "config.json").is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError("no MiniMax-H3 download found: pass --h3-root or set H3_ROOT to the folder holding FL2VA/video_vae")


def _cast_inputs(module, args):
    dtype = next(module.parameters()).dtype
    return tuple(
        value.to(dtype=dtype) if torch.is_tensor(value) and value.is_floating_point() and value.dtype != dtype else value
        for value in args
    )


def _posterior_mean(self, generator=None):
    return self.mean


def load_vae(h3_root: str | Path | None = None, device: str | torch.device = "cuda"):
    """MiniMaxH3VideoVAE(...).model in bfloat16; encode_base returns the posterior mean."""
    root = find_h3_root(h3_root)
    if str(root / "FL2VA") not in sys.path:
        sys.path.insert(0, str(root / "FL2VA"))
    from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
    from video_vae.vae_module import DiagonalGaussianDistribution

    DiagonalGaussianDistribution.sample = _posterior_mean
    model = MiniMaxH3VideoVAE.from_pretrained(str(root / "FL2VA" / "video_vae")).model
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    for name in ("encoder", "quant_conv", "post_quant_conv", "decoder"):
        getattr(model, name).register_forward_pre_hook(_cast_inputs)
    model.stack_tiling = False
    return model


def latent_stats(h3_root: str | Path | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """latents_mean and latents_std of the VAE config as [1, C, 1, 1, 1] float32.

    The H3 DiT works on (z - mean) / std; its latents go into the VAE as latents * std + mean.
    """
    config = json.loads((find_h3_root(h3_root) / "FL2VA" / "video_vae" / "config.json").read_text())
    mean, std = (torch.tensor(config[key], dtype=torch.float32).view(1, -1, 1, 1, 1) for key in ("latents_mean", "latents_std"))
    return mean, std


def read_video(path: str | Path, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """RGB frames [T, H, W, 3] uint8 and the frame rate."""
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        if max_frames is not None and len(frames) >= max_frames:
            break
    container.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0), fps


def write_video(path: str | Path, frames: np.ndarray, fps: float) -> None:
    """H.264 at qp 0, yuv420p."""
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"qp": "0", "preset": "veryfast"}
    for rgb in frames:
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def fit_size(frames: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-resize so the frames cover width x height, then center-crop. Needs opencv."""
    import cv2

    count, src_h, src_w = frames.shape[:3]
    scale = max(width / src_w, height / src_h)
    new_w = max(width, int(round(src_w * scale)))
    new_h = max(height, int(round(src_h * scale)))
    x0 = (new_w - width) // 2
    y0 = (new_h - height) // 2
    out = np.empty((count, height, width, 3), dtype=np.uint8)
    for index in range(count):
        resized = cv2.resize(frames[index], (new_w, new_h), interpolation=cv2.INTER_AREA)
        out[index] = resized[y0 : y0 + height, x0 : x0 + width]
    return out


def trim(vae, frames: np.ndarray) -> np.ndarray:
    """The frames the VAE encodes: whole causal chunks, sides cut down to a multiple of 16."""
    used = vae.processor.get_suitable_video_length(int(frames.shape[0]))
    height, width = vae.processor._align_to_total_patch_size(int(frames.shape[1]), int(frames.shape[2]))
    return np.ascontiguousarray(frames[:used, :height, :width])


def to_pixels(vae, frames: np.ndarray) -> torch.Tensor:
    """uint8 RGB [T, H, W, 3] to the [1, 3, T, H, W] input of encode_base."""
    device = next(vae.parameters()).device
    tensor = vae.processor.convert_numpy_to_tensor(frames, device)
    return vae.processor.transform_tensor(tensor).transpose(0, 1).unsqueeze(0).contiguous()


def to_uint8(vae, pixels: torch.Tensor) -> np.ndarray:
    """Decoder output [1, 3, T, H, W] to uint8 RGB [T, H, W, 3]."""
    pixels = vae.processor.revert_tensor(pixels)
    pixels = pixels[0].detach().float().clamp(0, 1).cpu()
    return (pixels.permute(1, 2, 3, 0).numpy() * 255.0).round().astype(np.uint8)


def psnr(reference: np.ndarray, frames: np.ndarray) -> float:
    """RGB PSNR from the mean squared error over all frames."""
    count = min(len(reference), len(frames))
    sse = 0.0
    for a, b in zip(reference[:count], frames[:count]):
        sse += float(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    mse = sse / (count * reference[0].size)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0**2 / mse))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def official_decode(vae, latents: torch.Tensor) -> tuple[np.ndarray, float]:
    """decode_base on all latents at once; uint8 frames and seconds."""
    device = latents.device
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        recon = vae.decode_base(latents)
    _sync(device)
    seconds = time.perf_counter() - started
    frames = to_uint8(vae, recon)
    del recon
    torch.cuda.empty_cache()
    return frames, seconds


def stream_decode(vae, decoder: StreamingDecoder, latents: torch.Tensor, warmup: int = 8) -> tuple[np.ndarray, dict]:
    """push() the latents one at a time; uint8 frames and the time of every step.

    The first `warmup` latents are decoded once beforehand, so torch.compile and the CUDA
    libraries are ready. Only push() is timed; the conversion to uint8 is not.
    """
    device = latents.device
    if warmup:
        for _ in decoder.stream(latents[:, :, :warmup]):
            pass
        decoder.reset()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    frames, steps = [], []
    for index in range(latents.shape[2]):
        _sync(device)
        tick = time.perf_counter()
        piece = decoder.push(latents[:, :, index : index + 1])
        _sync(device)
        steps.append(time.perf_counter() - tick)
        frames.append(to_uint8(vae, piece))
    frames = np.concatenate(frames, axis=0)
    shifts = [t for i, t in enumerate(steps) if i >= decoder.window and i % decoder.chunk == 2]
    timing = {
        "latents": len(steps),
        "frames": int(len(frames)),
        "decode_s": float(sum(steps)),
        "fps": float(len(frames) / sum(steps)),
        "first_frame_s": float(steps[0]),
        "first_chunk_s": float(sum(steps[: decoder.chunk])),
        "step_median_s": float(np.median(steps)),
        "step_max_s": float(np.max(steps)),
        "shift_step_median_s": float(np.median(shifts)) if shifts else None,
        "slot_passes": decoder.slot_passes,
    }
    if device.type == "cuda":
        timing["peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    return frames, timing


def add_decoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h3-root", type=Path, help="MiniMax-H3 download holding FL2VA/video_vae (default: $H3_ROOT, then ..)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="v3",
        help="v3: final version; v2: without the slot-0 refresh; v1: without the brightness compensation either",
    )
    parser.add_argument("--no-compile", action="store_true", help="run the blocks eagerly, without torch.compile")


def make_decoder(vae, args: argparse.Namespace) -> StreamingDecoder:
    return StreamingDecoder(vae, compile=not args.no_compile, **VARIANTS[args.variant])
