"""FFN GEMM epilogue that matches the compiled V3 SiLU rounding.

Quack runs the GEMM. The epilogue rounds the accumulator before the bias, then
rounds SiLU and the up branch separately before the multiply. The installed
quack package is not modified.
"""
import torch
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass import Float32,BFloat16,const_expr
from cutlass.cutlass_dsl import dsl_user_op
import quack.gemm_act as ga
from quack.activation import gate_fn_map,silu,F32_or_F32x2
from quack.gemm_interface import gemm_act,gated_to_pytorch_fn_map
from .. import stream_decoder as base
from .v3_fused import FusedOutputDecoder

@dsl_user_op
def v3_swiglu(x:F32_or_F32x2,y:F32_or_F32x2,*,loc=None,ip=None)->F32_or_F32x2:
    if const_expr(not isinstance(x,tuple)):
        return Float32(BFloat16(silu(x)))*Float32(BFloat16(y))
    else:
        sx=silu(x)
        return (Float32(BFloat16(sx[0]))*Float32(BFloat16(y[0])),Float32(BFloat16(sx[1]))*Float32(BFloat16(y[1])))

class V3GatedSm100(ga.GemmGatedSm100):
    @cute.jit
    def epi_visit_subtile(self,params,epi_loop_tensors,tRS_rD,tRS_rC=None):
        for i in cutlass.range(cute.size(tRS_rD),unroll_full=True,vectorize=True):
            tRS_rD[i]=Float32(BFloat16(tRS_rD[i]))
        return ga.GemmGatedMixin.epi_visit_subtile(self,params,epi_loop_tensors,tRS_rD,tRS_rC)

gate_fn_map['v3_bf16_swiglu']=v3_swiglu
gated_to_pytorch_fn_map['v3_bf16_swiglu']=lambda x,y:F.silu(x).bfloat16().float()*y.bfloat16().float()
ga.GemmGatedSm100=V3GatedSm100

def _pre(tokens,attn,w,b,scale,norm,eps):
    tokens=tokens+F.linear(attn.reshape(tokens.shape),w,b)*scale
    return tokens,base._rms(tokens,norm,eps).to(tokens.dtype)
def _end(tokens,gate,w,b,scale):
    return tokens+F.linear(gate.reshape(*tokens.shape[:-1],-1),w,b)*scale

def post(tokens,attn,w,b,scale,norm,eps,w1,b1,w2,b2,scale2):
    tokens,normed=base._compiled(_pre)(tokens,attn,w,b,scale,norm,eps)
    _,gate=gemm_act(normed.reshape(-1,normed.shape[-1]),w1.T,bias=b1,activation='v3_bf16_swiglu',store_preact=False,tuned=False,concat_layout=('B','bias'))
    return base._compiled(_end)(tokens,gate,w2,b2,scale2)

class FusedFFNDecoder(FusedOutputDecoder):
    post=staticmethod(post)
