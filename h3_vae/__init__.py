from .h3_vae import (
    add_decoder_args,
    fit_size,
    latent_stats,
    load_vae,
    make_decoder,
    official_decode,
    psnr,
    read_video,
    stream_decode,
    to_pixels,
    trim,
    write_video,
)
from .stream_decoder import StreamingDecoder

def __getattr__(name):
    if name == "FusedFFNDecoder":
        from .kernels.v3_ffn import FusedFFNDecoder
        return FusedFFNDecoder
    raise AttributeError(name)
