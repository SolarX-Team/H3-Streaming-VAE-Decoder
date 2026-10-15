"""Fuse V3 output unpack, original overlap rules, crop and compensation.

Preserves every BF16 multiply/add round in klvae.blend and FP32 compensation.
Horizontal blending reads the raw previous tile, exactly as the reference.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _read(P,tile,c,t,y,x):
    row=(y//16)*16+x//16
    col=((c*4+t)*16+y%16)*16+x%16
    return tl.load(P+tile*256*3072+row*3072+col).to(tl.float32)

@triton.jit
def _output(P,XMAP,YMAP,A,B,C,O,H:tl.constexpr,W:tl.constexpr,COLS:tl.constexpr,F:tl.constexpr,HASC:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<3*F*H*W
    x=i%W;y=i//W%H;t=i//(H*W)%F+(4-F);c=i//(F*H*W)
    tx=tl.load(XMAP+x*3);lx=tl.load(XMAP+x*3+1);ox=tl.load(XMAP+x*3+2)
    ty=tl.load(YMAP+y*3);ly=tl.load(YMAP+y*3+1);oy=tl.load(YMAP+y*3+2)
    # Final padded lanes use a safe channel. Store is still masked.
    c=tl.minimum(c,2);tile=ty*COLS+tx;v=_read(P,tile,c,t,ly,lx)
    ya=_read(P,tl.maximum(tile-COLS,0),c,t,tl.minimum(256-tl.maximum(oy,1)+ly,255),lx)
    wy=(ly.to(tl.bfloat16).to(tl.float32)/tl.maximum(oy,1)).to(tl.bfloat16).to(tl.float32)
    ay=(1.-wy).to(tl.bfloat16).to(tl.float32)
    mix=((ya*ay).to(tl.bfloat16).to(tl.float32)+(v*wy).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    v=tl.where((ty>0)&(ly<oy),mix,v)
    xa=_read(P,tl.maximum(tile-1,0),c,t,ly,tl.minimum(256-tl.maximum(ox,1)+lx,255))
    wx=(lx.to(tl.bfloat16).to(tl.float32)/tl.maximum(ox,1)).to(tl.bfloat16).to(tl.float32)
    ax=(1.-wx).to(tl.bfloat16).to(tl.float32)
    mix=((xa*ax).to(tl.bfloat16).to(tl.float32)+(v*wx).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    v=tl.where((tx>0)&(lx<ox),mix,v)
    off=c*F+(t-(4-F));a=tl.load(A+off);b=tl.load(B+off);shift=a+b*v
    if HASC:shift=shift+tl.load(C+off)*v*v
    tl.store(O+i,(v-shift).to(tl.bfloat16),mask)

def mapping(lengths,overlaps,device):
    rows=[]
    for i,length in enumerate(lengths):
        keep=length-(overlaps[i] if i<len(lengths)-1 else 0)
        rows.extend((i,j,overlaps[i-1] if i else 0) for j in range(keep))
    return torch.tensor(rows,device=device,dtype=torch.int32)

def render_output(d,raw,index,slot):
    p=d._plan;F=d.frames_of(index)
    if not hasattr(d,'_output_maps'):
        d._output_maps=(mapping([256]*p['cols'],p['x_overlap'],raw.device),mapping([256]*p['rows'],p['y_overlap'],raw.device))
    xm,ym=d._output_maps;H,W=ym.shape[0],xm.shape[0]
    if index<=d.chunk:
        first=d.first_frame(index);table=d._table('first',raw.device)
        vals={k:v[:,first:first+F].contiguous() for k,v in table.items() if k in ('a','b','c')}
        hasc=d._first_c is not None and bool(d._first_c[first:first+F].any())
    else:
        table=d._table('steady',raw.device);vals={k:v[slot,:,4-F:].contiguous() for k,v in table.items() if k in ('a','b','c')};hasc='c' in table
    out=torch.empty((1,3,F,H,W),device=raw.device,dtype=raw.dtype)
    _output[(triton.cdiv(out.numel(),256),)](raw,xm,ym,vals['a'],vals['b'],vals.get('c'),out,H,W,p['cols'],F,hasc,256,enable_fp_fusion=False)
    return out
