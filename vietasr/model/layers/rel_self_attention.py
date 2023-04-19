import torch
from torch import nn, einsum

from einops import rearrange


# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class Rel_MultiheadAttention(nn.Module):

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        max_pos_emb: int = 512,
        dropout: int = 0.1,
    ):
        super(Rel_MultiheadAttention, self).__init__()
        
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

    def forward(self, x, context = None, mask = None, context_mask = None):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        # print(q.shape, k.shape, v.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # print(q.shape, k.shape, v.shape)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # print(dots.shape)

        # shaw's relative positional embedding
        
        # print("shaw's relative positional embedding")
        seq = torch.arange(n, device = device)
        
        # print(seq.shape)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        
        # print(dist.shape)
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        
        # print(dist.shape)
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        
        # print(rel_pos_emb.shape)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        
        # print(pos_attn.shape)
        
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out