"""One kernel for V3 Q/K RMSNorm, bf16 RoPE, and the suffix-first K/V store.

36 blocks and the spatial tiles stay as in StreamingDecoder.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from .. import stream_decoder as base
from .v3_graph import GraphDecoder

@triton.jit
def _qk_rope_store(X,COS,SIN,Q,KC,VC,QW,KW,L:tl.constexpr,H:tl.constexpr,D:tl.constexpr,R:tl.constexpr,
                  CACHE:tl.constexpr,NF:tl.constexpr,FIRST:tl.constexpr,S:tl.constexpr,
                  QE:tl.constexpr,KE:tl.constexpr,QAFF:tl.constexpr,KAFF:tl.constexpr,G:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)*G+tl.arange(0,G)[:,None];j=tl.arange(0,D)[None,:]
    batch=row//L;pos=row%L;off=row*(H*3*D)+head*3*D+j
    q=tl.load(X+off).to(tl.float32);k=tl.load(X+off+D).to(tl.float32);v=tl.load(X+off+2*D)
    q=q*tl.rsqrt(tl.sum(q*q,1)[:,None]/D+QE)
    k=k*tl.rsqrt(tl.sum(k*k,1)[:,None]/D+KE)
    if QAFF:q=q*tl.load(QW+j).to(tl.float32)
    if KAFF:k=k*tl.load(KW+j).to(tl.float32)
    q=q.to(tl.bfloat16).to(tl.float32);k=k.to(tl.bfloat16).to(tl.float32)
    c=tl.load(COS+pos*R+j,j<R,0).to(tl.bfloat16).to(tl.float32)
    s=tl.load(SIN+pos*R+j,j<R,0).to(tl.bfloat16).to(tl.float32)
    other=tl.minimum(tl.maximum(tl.where(j<R//2,j+R//2,j-R//2),0),D-1)
    qo=tl.gather(q,tl.broadcast_to(other,(G,D)),1)*tl.where(j<R//2,-1.,1.)
    ko=tl.gather(k,tl.broadcast_to(other,(G,D)),1)*tl.where(j<R//2,-1.,1.)
    qr=((q*c).to(tl.bfloat16).to(tl.float32)+(qo*s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    kr=((k*c).to(tl.bfloat16).to(tl.float32)+(ko*s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    q=tl.where(j<R,qr,q).to(tl.bfloat16);k=tl.where(j<R,kr,k).to(tl.bfloat16)
    dest=tl.where(pos<NF,S+FIRST+pos,pos-NF)
    tl.store(Q+row*H*D+head*D+j,q)
    tl.store(KC+(batch*CACHE+dest)*H*D+head*D+j,k)
    tl.store(VC+(batch*CACHE+dest)*H*D+head*D+j,v)

def _linear(tokens,norm,eps,w,b):
    return F.linear(base._rms(tokens,norm,eps).to(tokens.dtype),w,b)

class FusedDecoder(GraphDecoder):
    def _layers(self,span,z,fresh,slot,rope):
        d=self.decoder;p=self._plan;hw=p['hw'];batch=z.shape[0];nf=len(fresh)*hw
        assert fresh==list(range(fresh[0],fresh[0]+len(fresh)))
        hidden=d.x_embedder(z.permute(0,2,3,4,1).reshape(batch,nf,z.shape[1]))
        parts=[hidden]
        if d.register_tokens is not None:parts.append(d.register_tokens.expand(batch,-1,-1).to(hidden.dtype))
        parts.append(hidden.new_zeros(batch,1,hidden.shape[-1]));tokens=torch.cat(parts,1)
        cos,sin=rope;visible=self.suffix+(slot+1)*hw
        linear=base._compiled(_linear);post=getattr(self,'post',None) or base._compiled(base._attn_out)
        for layer,b in enumerate(d.transformer_blocks):
            a=b.attn;H=a.heads;D=a.dim_head;L=tokens.shape[1]
            qkv=linear(tokens,b.norm1.weight,base._eps(b.norm1),a.to_qkv.weight,a.to_qkv.bias)
            query=torch.empty((batch,L,H,D),device=z.device,dtype=z.dtype)
            k=self._k[layer][span];v=self._v[layer][span]
            _qk_rope_store[(batch*L,H//16)](qkv,cos,sin,query,k,v,a.norm_q.weight,a.norm_k.weight,L,H,D,cos.shape[-1],k.shape[1],nf,fresh[0]*hw,self.suffix,base._eps(a.norm_q),base._eps(a.norm_k),a.norm_q.weight is not None,a.norm_k.weight is not None,16,num_warps=4,enable_fp_fusion=False)
            attn=F.scaled_dot_product_attention(query.transpose(1,2),k[:,:visible].transpose(1,2),v[:,:visible].transpose(1,2),dropout_p=0.).transpose(1,2)
            tokens=post(tokens,attn,a.to_out.weight,a.to_out.bias,b.scale1,b.norm2.weight,base._eps(b.norm2),b.ff.w1.weight,b.ff.w1.bias,b.ff.w2.weight,b.ff.w2.bias,b.scale2)
        own=fresh.index(slot)*hw
        output=d.proj_out(d.norm_out(tokens[:,own:own+hw].contiguous()))
        if getattr(self,'raw_output',False):return output
        c=d.config
        return base._unpack(output,c.patch_size,c.patch_size_t,c.patch_size_t,p['h']*c.patch_size,p['w']*c.patch_size)

class FusedOutputDecoder(FusedDecoder):
    raw_output=True
    def _render(self,z,fresh,slot,rope,index):
        from .v3_output import render_output
        batch=max(1,self.token_budget//(len(fresh)*self._plan['hw']+self.suffix))
        spans=[slice(b,min(b+batch,z.shape[0])) for b in range(0,z.shape[0],batch)]
        pieces=[self._layers(span,z[span],fresh,slot,rope) for span in spans]
        raw=pieces[0] if len(pieces)==1 else torch.cat(pieces,0)
        return render_output(self,raw,index,slot)
