import json
from pathlib import Path

from .decoder.causal_vit_decoder import CausalViTDecoder

def load_config():
    return json.loads((Path(__file__).resolve().parent / "config.json").read_text())

def __getattr__(name):
    if name == "FusedStreamingCausalViT":
        from .kernels.sr_fused_final import FusedStreamingCausalViT
        return FusedStreamingCausalViT
    raise AttributeError(name)
