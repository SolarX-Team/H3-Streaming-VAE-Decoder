# H3 Streaming VAE Decoder

Streaming decode for the MiniMax H3 video VAE. Each latent is decoded as soon as it arrives. Supports both the official H3 VAE decoder and the StreamRender-H3 VAE decoder.

## Install

```bash
pip install -e .            # StreamingDecoder and CausalViTDecoder
pip install -e ".[fused]"   # also the fused kernels
```

`h3_vae` also needs a MiniMax-H3 download that contains `FL2VA/`:

```bash
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
```

## H3 VAE

Without fused kernels. `torch.compile` runs on the machine that executes it. `--no-compile` turns that off. `--official` also writes the official decode. Saved normalized DiT latents need `--normalized`.

```bash
python -m h3_vae.reconstruct --h3-root /path/to/MiniMax-H3 --video input.mp4 --output stream.mp4
python -m h3_vae.decode_latents --h3-root /path/to/MiniMax-H3 --latents latents.pt --output stream.mp4
```

```python
from h3_vae import load_vae, StreamingDecoder

vae = load_vae("/path/to/MiniMax-H3", "cuda")
decoder = StreamingDecoder(vae)
for latent in generator:          # [1, C, 1, h, w]
    frames = decoder.push(latent) # [1, 3, n, H, W], n is 1 or 4
```

With fused kernels. Needs a Blackwell GPU and `pip install -e ".[fused]"`. The kernels compile on that GPU at the first call.

```python
from h3_vae import load_vae, FusedFFNDecoder

vae = load_vae("/path/to/MiniMax-H3", "cuda")
decoder = FusedFFNDecoder(vae)   # push() is the same as above
```

## StreamRender-H3

The student weights are NVIDIA's distilled 24-layer decoder, `causal_decoder/causal_vit24_step6000.safetensors` in [Efficient-Large-Model/StreamRender-H3](https://huggingface.co/Efficient-Large-Model/StreamRender-H3). That file stays under the MiniMax H3 community license. The rest of the repo is the renderer and is not used here.

```bash
hf download Efficient-Large-Model/StreamRender-H3 --include "causal_decoder/causal_vit24_step6000.safetensors" --local-dir StreamRender-H3
```

Without fused kernels. `chunk=1` by default; `chunk=2` is the original pair step. `latents` is normalized, `[1, 1, 24, H, W]`.

```python
import torch
from safetensors.torch import load_file
from streamrender_h3 import CausalViTDecoder, load_config

student = CausalViTDecoder.from_config(load_config(), chunk=1)
checkpoint = load_file("StreamRender-H3/causal_decoder/causal_vit24_step6000.safetensors", device="cpu")
student.load_state_dict(checkpoint, strict=True, assign=True)
student = student.cuda().eval()
state = student.init_state()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    frames = student.decode_chunk(latents, state)
```

With fused kernels. Same checkpoint and the same normalized latents. `H` and `W` are the latent size; 1344×768 is 48×84. Needs Blackwell, FlashAttention-4, and quack.

```python
from safetensors.torch import load_file
from streamrender_h3 import CausalViTDecoder, FusedStreamingCausalViT, load_config

student = CausalViTDecoder.from_config(load_config(), chunk=2)
checkpoint = load_file("StreamRender-H3/causal_decoder/causal_vit24_step6000.safetensors", device="cpu")
student.load_state_dict(checkpoint, strict=True, assign=True)
student = student.cuda().eval()
decoder = FusedStreamingCausalViT(student, H=48, W=84)
frames = decoder.decode_chunk(latents)
```

This repo is Apache-2.0. The official H3 weights and `FL2VA` stay under the [MiniMax H3 community license](https://github.com/MiniMax-AI/MiniMax-H3). `streamrender_h3/decoder` follows the public [StreamRender-H3](https://github.com/NVlabs/Sana/tree/sol-engine/models/streamrender_h3) deployment path. `streamrender_h3/h3_video_vae` keeps the Apache-2.0 headers those modules were published with. The student checkpoint above is not in this repo.
