"""Fused kernels for the StreamRender-H3 deployment decoder.

Same 24 blocks, shifted 16x16 windows, 4 historical latents plus the current one,
global suffix attention, BF16 linear weights, FP32 residuals and output.
"""
import torch
import triton
import triton.language as tl
from quack.gemm_interface import gemm_act
from ..decoder.causal_vit_stream import StreamingCausalViT,flash_attn_varlen_func,BF16
from ..decoder.causal_vit_decoder import PATCH,PATCH_T,OUT_CH

@triton.jit
def _qkv_cache(QKV,COS,SIN,SCOS,SSIN,PERM,WIDX,SIDX,Q,QS,KP,VP,
               N:tl.constexpr,S:tl.constexpr,NW:tl.constexpr,HEADS:tl.constexpr,D:tl.constexpr,ROT:tl.constexpr,EPS:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)*16+tl.arange(0,16)[:,None];j=tl.arange(0,D)[None,:]
    is_s=row>=N
    src=tl.load(PERM+row,row<N,0)
    src=tl.where(is_s,row,src)
    off=src*(HEADS*3*D)+head*(3*D)+j
    q=tl.load(QKV+off).to(tl.float32);k=tl.load(QKV+off+D).to(tl.float32);v=tl.load(QKV+off+2*D)
    qr=(q*tl.rsqrt(tl.sum(q*q,1)[:,None]/D+EPS)).to(tl.bfloat16).to(tl.float32)
    kr=(k*tl.rsqrt(tl.sum(k*k,1)[:,None]/D+EPS)).to(tl.bfloat16).to(tl.float32)
    c=tl.load(COS+src*ROT+j,(row<N)&(j<ROT),0)
    si=tl.load(SIN+src*ROT+j,(row<N)&(j<ROT),0)
    cs=tl.load(SCOS+(row-N)*ROT+j,is_s&(j<ROT),0)
    ss=tl.load(SSIN+(row-N)*ROT+j,is_s&(j<ROT),0)
    c=tl.where(is_s,cs,c).to(tl.bfloat16).to(tl.float32);si=tl.where(is_s,ss,si).to(tl.bfloat16).to(tl.float32)
    other=tl.where(j<ROT//2,j+ROT//2,j-ROT//2)
    other=tl.minimum(tl.maximum(other,0),D-1)
    qo=tl.gather(qr,tl.broadcast_to(other,(16,D)),1)*tl.where(j<ROT//2,-1.,1.)
    ko=tl.gather(kr,tl.broadcast_to(other,(16,D)),1)*tl.where(j<ROT//2,-1.,1.)
    qrot=((qr*c).to(tl.bfloat16).to(tl.float32)+(qo*si).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    krot=((kr*c).to(tl.bfloat16).to(tl.float32)+(ko*si).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    qr=tl.where(j<ROT,qrot,qr).to(tl.bfloat16);kr=tl.where(j<ROT,krot,kr).to(tl.bfloat16)
    tl.store(Q+row*HEADS*D+head*D+j,qr,row<N)
    tl.store(QS+(row-N)*HEADS*D+head*D+j,qr,is_s)
    dest=tl.load(WIDX+row,row<N,0)
    tl.store(KP+dest*HEADS*D+head*D+j,kr,row<N)
    tl.store(VP+dest*HEADS*D+head*D+j,v,row<N)
    if is_s:
        for win in range(NW):
            dest=tl.load(SIDX+win*S+row-N)
            tl.store(KP+dest*HEADS*D+head*D+j,kr)
            tl.store(VP+dest*HEADS*D+head*D+j,v)

@triton.jit
def _suffix_partial(Q,K,V,BIAS,P,L,NK:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SPLITS:tl.constexpr,B:tl.constexpr):
    h=tl.program_id(0);part=tl.program_id(1)
    m=tl.arange(0,16);d=tl.arange(0,D);k=part*B+tl.arange(0,B)
    q=tl.load(Q+m[:,None]*H*D+h*D+d[None,:],m[:,None]<5,0)
    kk=tl.load(K+k[None,:]*H*D+h*D+d[:,None],k[None,:]<NK,0)
    scores=tl.dot(q,kk)*D**-0.5
    bias=tl.load(BIAS+k,k<NK,float('-inf'));scores+=bias[None,:]
    mx=tl.max(scores,1);mx=tl.where(mx==float('-inf'),0.,mx)
    p=tl.exp(scores-mx[:,None]);z=tl.sum(p,1)
    vv=tl.load(V+k[:,None]*H*D+h*D+d[None,:],k[:,None]<NK,0)
    val=tl.dot(p.to(tl.bfloat16),vv)/tl.where(z[:,None]>0,z[:,None],1.)
    tl.store(P+((h*SPLITS+part)*5+m[:,None])*D+d[None,:],val,m[:,None]<5)
    tl.store(L+(h*SPLITS+part)*5+m,mx+tl.log(z),m<5)

@triton.jit
def _suffix_merge(P,L,O,H:tl.constexpr,D:tl.constexpr,SPLITS:tl.constexpr,B:tl.constexpr):
    h=tl.program_id(0);m=tl.program_id(1);s=tl.arange(0,B);d=tl.arange(0,D)
    l=tl.load(L+(h*SPLITS+s)*5+m,s<SPLITS,float('-inf'));mx=tl.max(l,0);p=tl.exp(l-mx);den=tl.sum(p,0)
    v=tl.load(P+((h*SPLITS+s[:,None])*5+m)*D+d[None,:],s[:,None]<SPLITS,0)
    # Merge normalized per-partition attention using log-sum-exp weights.
    val=tl.sum(v*p[:,None],0)/den
    tl.store(O+m*H*D+h*D+d,val.to(tl.bfloat16))

@triton.jit
def _head_hilo(X,W,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=64):
    m=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
    acc=tl.full((BM,BN),0.,tl.float32)
    for ki in range(tl.cdiv(K,BK)):
        kk=ki*BK+k
        a=tl.load(X+m[:,None]*K+kk[None,:],(m[:,None]<M)&(kk[None,:]<K),0)
        b=tl.load(W+n[None,:]*K+kk[:,None],(n[None,:]<N)&(kk[:,None]<K),0)
        ah=a.to(tl.bfloat16);bh=b.to(tl.bfloat16)
        al=(a-ah.to(tl.float32)).to(tl.bfloat16);bl=(b-bh.to(tl.float32)).to(tl.bfloat16)
        acc=tl.dot(ah,bl,acc);acc=tl.dot(al,bh,acc);acc=tl.dot(ah,bh,acc)
    bias=tl.load(BIAS+n,n<N,0);tl.store(Y+m[:,None]*N+n[None,:],acc+bias[None,:],(m[:,None]<M)&(n[None,:]<N))

def head_hilo(x,head):
    y=torch.empty((x.shape[0],head.weight.shape[0]),device=x.device,dtype=torch.float32)
    _head_hilo[(triton.cdiv(y.shape[0],128),triton.cdiv(y.shape[1],64))](x,head.weight,head.bias,y,y.shape[0],y.shape[1],x.shape[1],BM=128,BN=64,BK=64,num_warps=8,num_stages=3)
    return y

def _qkv_linear(blk,x):return blk.attn.to_qkv(blk.norm1(x).to(BF16))
def _merge_project(blk,x,o,os,inv):
    return blk.attn.to_out(torch.cat([o.flatten(1)[inv],os],0))

@triton.jit
def _resnorm(X,Y,SCALE,WEIGHT,OUT,NORM,D:tl.constexpr,EPS:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,D)
    x=tl.load(X+row*D+i);y=tl.load(Y+row*D+i).to(tl.float32)
    scale=tl.load(SCALE+i);w=tl.load(WEIGHT+i)
    x=x+y*scale
    norm=x*tl.rsqrt(tl.sum(x*x,0)/D+EPS)*w
    tl.store(OUT+row*D+i,x);tl.store(NORM+row*D+i,norm.to(tl.bfloat16))

def resnorm(x,y,scale,norm):
    out=torch.empty_like(x);n=torch.empty_like(x,dtype=BF16)
    _resnorm[(x.shape[0],)](x,y,scale,norm.weight,out,n,x.shape[1],norm.eps,num_warps=4,enable_fp_fusion=False)
    return out,n

def _meta(r,t,C,H,W,s,S,bank,n,row_slot,dup,ns):
    ts=(torch.arange(C,device=t.device,dtype=torch.float32)+t+0.5)/r.time_norm*2-1
    hy=(((torch.arange(H,device=t.device)+s)%r.window).float()+0.5)/r.window*2-1
    wx=(((torch.arange(W,device=t.device)+s)%r.window).float()+0.5)/r.window*2-1
    grid=torch.stack(torch.meshgrid(ts,hy,wx,indexing='ij'),-1).reshape(1,-1,3)
    cos,sin=r.pos_embed(grid)
    scos,ssin=r._suffix_ids(t,S,torch.float32,t.device)
    filled=torch.minimum(t+C,t.new_tensor(ns))
    idx=bank.index_select(0,((t%ns)//C).reshape(1)).squeeze(0)
    used=(S+filled*n).to(torch.int32)
    bias=dup+torch.where((row_slot>=filled)&(row_slot<ns),float('-inf'),0.)
    return cos,sin,scos,ssin,idx,used,bias

def _suffix_torch(qs,kp,vp,kbias,dh):
    sc=torch.matmul(qs.transpose(0,1),kp.permute(1,2,0)).float()*dh**-.5
    p=(sc+kbias).softmax(-1).to(BF16)
    return torch.matmul(p,vp.transpose(0,1)).transpose(0,1).flatten(1)

class FusedStreamingCausalViT(StreamingCausalViT):
    def __init__(self,r,H,W,*,cuda_graph=True,variant='fused'):
        super().__init__(r,H,W,cuda_graph=cuda_graph)
        self.variant=variant
        self._linear=torch.compile(_qkv_linear,dynamic=False)
        self._combined_project=torch.compile(_merge_project,dynamic=False)
        self._meta=torch.compile(_meta,dynamic=False,fullgraph=True)
        self.clock=torch.zeros((),device=r.register_tokens.device,dtype=torch.int64)
        self.banks={s:torch.stack(list(g['widx'].values())) for s,g in self.geo.items()}
        # Normalize the first token rows once; later RMSNorm is fused with residual.
        self._first_norm=torch.compile(lambda norm,x:norm(x).to(BF16),dynamic=False)
        self._suffix=torch.compile(_suffix_torch,dynamic=False)

    def _set_inputs(self,z):
        self.z_in.copy_(z)
        self.clock.fill_(self.t)

    @torch.no_grad()
    def _step(self):
        r,C,H,W=self.r,self.C,self.H,self.W;N=C*H*W;D=self.heads*self.dh
        with torch.autocast('cuda',dtype=BF16):
            raw=self.z_in*r.latents_std.view(1,1,-1,1,1)+r.latents_mean.view(1,1,-1,1,1)
            raw=r.post_quant_conv(raw.transpose(1,2))
        x=r.x_embedder(raw.permute(0,2,3,4,1).float()).to(BF16).float().reshape(N,-1)
        suf=torch.cat([r.register_tokens[0],r.register_tokens.new_zeros(1,D)],0).to(BF16).float()
        x=torch.cat([x,suf],0)
        metadata={}
        for s,g in self.geo.items():
            metadata[s]=self._meta(r,self.clock,C,H,W,s,self.S,self.banks[s],g['n'],g['row_slot'],g['dup'],self.nslot)
            cos,sin,scos,ssin,idx,used,bias=metadata[s]
            self.cos[s],self.sin[s],self.scos,self.ssin=cos,sin,scos,ssin
            g['widx_cur'].copy_(idx);g['seqused'].copy_(used);g['kbias'].copy_(bias)
        xn=self._first_norm(self.blocks[0].norm1,x)
        for li,blk in enumerate(self.blocks):
            s=self.shifts[li];g=self.geo[s];kp,vp=self.kpool[li],self.vpool[li]
            qkv=blk.attn.to_qkv(xn)
            Q=torch.empty((N,self.heads,self.dh),device=x.device,dtype=BF16);qs=torch.empty((self.S,self.heads,self.dh),device=x.device,dtype=BF16)
            _qkv_cache[(N+self.S,self.heads//16)](qkv,self.cos[s],self.sin[s],self.scos,self.ssin,g['qperm'],g['widx_cur'],g['sidx'],Q,qs,kp,vp,N,self.S,g['nW'],self.heads,self.dh,self.cos[s].shape[-1],blk.attn.norm_q.eps,num_warps=4,enable_fp_fusion=False)
            O=flash_attn_varlen_func(Q,kp,vp,cu_seqlens_q=g['cu_q'],cu_seqlens_k=g['cu_k'],max_seqlen_q=g['max_q'],max_seqlen_k=g['max_k'],seqused_k=g['seqused'])
            if isinstance(O,tuple):O=O[0]
            if 'split' in self.variant:
                nk=kp.shape[0];splits=triton.cdiv(nk,256)
                pp=torch.empty((self.heads,splits,5,self.dh),device=x.device,dtype=torch.float32)
                ll=torch.empty((self.heads,splits,5),device=x.device,dtype=torch.float32)
                os_=torch.empty((5,self.heads,self.dh),device=x.device,dtype=BF16)
                _suffix_partial[(self.heads,splits)](qs,kp,vp,g['kbias'],pp,ll,nk,self.heads,self.dh,splits,256,num_warps=4)
                _suffix_merge[(self.heads,5)](pp,ll,os_,self.heads,self.dh,splits,triton.next_power_of_2(splits),num_warps=4)
                os_=os_.flatten(1)
            else:os_=self._suffix(qs,kp,vp,g['kbias'],self.dh)
            projected=self._combined_project(blk,x,O,os_,g['inv'])
            x,fn=resnorm(x,projected,blk.scale1,blk.norm2)
            _,gate=gemm_act(fn,blk.ff.w1.weight.T,bias=blk.ff.w1.bias,activation='swiglu',store_preact=False,tuned=False,concat_layout=('B','bias'))
            down=blk.ff.w2(gate)
            nxt=self.blocks[li+1].norm1 if li+1<len(self.blocks) else r.norm_out
            x,xn=resnorm(x,down,blk.scale2,nxt)
        x=r.norm_out(x[:N])
        out=head_hilo(x,r.proj_out) if 'head' in self.variant else r.proj_out(x)
        out=out.view(C,H,W,OUT_CH,PATCH_T,PATCH,PATCH).permute(3,0,4,1,5,2,6)
        return out.reshape(1,OUT_CH,C*PATCH_T,H*PATCH,W*PATCH)

    @torch.no_grad()
    def decode_chunk(self,z):
        assert z.shape==(1,self.C,24,self.H,self.W)
        self._set_inputs(z)
        if not self.cuda_graph or (self.graph is None and self.t<self.nslot):
            out=self._step()
        else:
            if self.graph is None:
                torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):self.out=self._step()
            self.graph.replay();out=self.out
        keep=[i for i in range(self.C*PATCH_T) if not ((self.t+i//PATCH_T)%self.r.block_latents==0 and i%PATCH_T<self.r.frames_to_trim)]
        self.t+=self.C
        return out[:,:,keep] if len(keep)<self.C*PATCH_T else out.clone()
