"""CUDA graphs for fixed V3 step shapes. Attention and bf16 math stay the same.

Tile/history management stays on the host. Each graph includes all 36 blocks,
output projection, original overlap blending and brightness compensation.
Graph outputs are cloned before returning so callers may retain frame tensors.
"""
import torch
from .. import stream_decoder as original

class GraphDecoder(original.StreamingDecoder):
    def __init__(self,*a,graph=True,**kw):
        super().__init__(*a,**kw)
        self.use_graph=graph;self.graphs={};self.pool=torch.cuda.graph_pool_handle()

    def _render(self,z,fresh,slot,rope,index):
        batch=max(1,self.token_budget//(len(fresh)*self._plan['hw']+self.suffix))
        spans=[slice(b,min(b+batch,z.shape[0])) for b in range(0,z.shape[0],batch)]
        tiles=torch.cat([self._layers(span,z[span],fresh,slot,rope) for span in spans],0)
        out=self._blend(tiles)[:,:,self.ratio-self.frames_of(index):]
        return self._compensate(index,slot,out)

    @torch.inference_mode()
    def push(self,latent):
        index=self.received
        z=self._tile(latent,new_stream=index==0);self.received+=1
        start=self._window_start(index)
        if start!=self._start:
            self._start=start;self._valid=[False]*self.window
            self._latents={i:v for i,v in self._latents.items() if i>=start}
        self._latents[index]=z;slot=index-start
        fresh={s for s in range(slot) if not self._valid[s]}|{slot}
        if self.first_refresh and start==0 and slot==1:fresh.add(0)
        fresh=sorted(fresh)
        z=torch.cat([self._latents[start+s] for s in fresh],2)
        rope=self._rope_of(fresh)
        if not self.use_graph:out=self._render(z,fresh,slot,rope,index)
        else:
            key=(tuple(z.shape),tuple(fresh),slot,index if index<=self.chunk else 'steady')
            if key not in self.graphs:
                buf=z.clone();stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):self._render(buf,fresh,slot,rope,index)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=self.pool):
                    result=self._render(buf,fresh,slot,rope,index)
                self.graphs[key]=(buf,graph,result)
            buf,graph,result=self.graphs[key]
            buf.copy_(z);graph.replay();out=result.clone()
        for s in fresh:self._valid[s]=True
        self.slot_passes+=len(fresh)
        return out
