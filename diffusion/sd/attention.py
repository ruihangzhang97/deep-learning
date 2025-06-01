import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    # in_proj_bias indicates whether the input projection layer should have a bias term
    # out_proj_bias indicates whether the output projection layer should have a bias term
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (batch_size, seq_len, d_embed)
        batch_size, seq_len, d_embed = x.shape()

        assert d_embed % self.n_heads == 0, "d_embed must be divisible by n_heads"

        intermim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # (batch_size, seq_len, d_embed) -> (batch_size, seq_len, 3 * d_embed)
        # Split into query, key, value tensors
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, d_embed) -> (batch_size, seq_len, self.n_heads, self.d_head) -> (batch_size, n_heads, seq_len, d_head)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # (batch_size, n_heads, seq_len, seq_len)
        weight = q @ k.transpose(-2, -1) * (self.d_head ** -0.5)

        if casual_mask:
            # mask where the upper triangular part is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(diagonal=1)
            # set the upper triangular part to -inf
            weight.masked_fill_(mask, float('-inf'))

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_head) -> (batch_size, n_heads, seq_len, d_head)
        output = weight @ v

        # (batch_size, n_heads, seq_len, d_head) -> (batch_size, seq_len, n_heads, d_head)
        output = output.transpose(1, 2)

        # resahpe to (batch_size, seq_len, d_embed), the input shape
        output = output.reshape(batch_size, seq_len, d_embed)

        output = self.out_proj(output)

        return output



