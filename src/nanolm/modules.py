import torch
from torch import Tensor, nn


class SlowMha(nn.Module):
    def __init__(self, dim: int, nheads: int, maxseqlen: int, head_dim=128):
        super().__init__()
        self.nheads = nheads
        self.head_dim = head_dim
        hdim = nheads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkv_proj = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        # causal_mask[target, source]: can target attend to source
        self.causal_mask = nn.Buffer(torch.ones(maxseqlen, maxseqlen, dtype=torch.bool))
        for i in range(1, maxseqlen):
            self.causal_mask.diagonal(i).logical_not_()
        self.out_proj = nn.Linear(hdim, dim, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = x.shape
        qkv = torch.einsum("btd,ahd->btah", x, self.qkv_proj)
        q, k, v = qkv.chunk(3, dim=-2) # leaves singleton split dim
        q = q.view(B, T, self.nheads, self.head_dim)
        k = k.view(B, T, self.nheads, self.head_dim)
        v = v.view(B, T, self.nheads, self.head_dim)
        logits = torch.einsum("btnh,bsnh->bnts", q, k) * (self.head_dim ** -0.5)
        masked_logits = logits.masked_fill(self.causal_mask.logical_not(), float("-inf"))
        attn = masked_logits.softmax(-1)
        output = torch.einsum("bnts,bsnd->btnd", attn, v)
        return self.out_proj(output.reshape(B, T, -1)), attn


class Ffn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.proj1 = nn.Linear(dim, hdim)
        self.proj2 = nn.Linear(hdim, dim)
        self.proj2.weight.detach().zero_()

    def forward(self, x: Tensor) -> Tensor:
        # x: B x T x D
        x = self.proj1(x)
        x = F.relu(x).square()
        x = self.proj2(x)
        return x

