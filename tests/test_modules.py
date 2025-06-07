from dataclasses import dataclass
import torch
from torch import nn
from nanolm.modules import SlowMha


# reference impl from nanogpt
@dataclass
class Config:
    n_embd: int
    n_head: int
    bias: bool = False
    dropout: float = 0.


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


def test_mha():
    B,T,D = (2, 4, 8)
    nheads = 2
    x = torch.rand(B,T,D)

    mha_torch = nn.MultiheadAttention(D, nheads, batch_first=True, bias=False)
    config = Config(D, nheads)
    mha_nano = CausalSelfAttention(config)
    mha = SlowMha(D, nheads, T, D // nheads)

    # convert weights
    mha_torch.in_proj_weight.data.copy_(mha.qkv_proj.data.view_as(mha_torch.in_proj_weight.data))
    mha_torch.out_proj.weight.data.copy_(mha.out_proj.weight.data)
    mha_nano.c_attn.weight.data.copy_(mha.qkv_proj.data.reshape(3*D, D))
    mha_nano.c_proj.weight.data.copy_(mha.out_proj.weight.data)

    attn_output, attn_weights = mha_torch(x, x, x, attn_mask=mha.causal_mask.logical_not(), is_causal=True, average_attn_weights=False)
    nano_output = mha_nano(x)
    output, weights = mha(x)

    assert torch.allclose(attn_weights, weights)
    assert torch.allclose(attn_output, output)
    assert torch.allclose(nano_output, output)

