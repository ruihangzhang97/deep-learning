import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.input_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.output_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads # dimension of each head: each head watches a part of the embedding
    
    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (batch_size, seq_len, dim)
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        intermim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        
        # (batch_size, seq_len, dim) -> (batch_size, seg_len, dim * 3) -> 3 tensors of (batch_size, seg_len, dim)
        q, k, v = self.input_proj(x).chunk(3, dim=-1)
        
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_heads, d_dead=dim/height) -> (batch_size, n_heads, seq_len, d_dead=dim/height)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (batch_size, n_heads, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # mask where the upper triangle(above the principal diagoanl) is made up with 1's
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=1)

        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, dim/height) -> (batch_size, n_heads, seq_len, dim/height)
        # recall in batched matmul, mul is performed on the last two dimensions of the tensors while preserving the leading dimensions
        output = weight @ v

        # (batch_size, n_heads, seq_len, dim/height) -> (batch_size, seq_len, n_heads, dim/height)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.output_proj(output)

        # (batch_size, seq_len, dim)
        return output
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x(latent): (batch_size, seq_len_Q, dim_Q)
        # y(context/prompt): (batch_size, seq_len_KV, dim_KV) = (batch_size, 77, 768)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # multiply query by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.k_proj(y)

        q = q.view(interim_shape.transpose(1, 2))
        k = k.view(interim_shape.transpose(1, 2))
        v = v.view(interim_shape.transpose(1, 2))

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1) # no casual mask since pixels and words are related

        output = weight @ v

        output = output.transpose(1, 2).continous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output