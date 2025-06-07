import torch
from torch import Tensor, nn


class Mha(nn.Module):
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
        for i in range(0, maxseqlen):
            self.causal_mask.diagonal(i).add_(-1)

    def forward(self, x: Tensor) -> Tensor:
        # x: B x T x D
        qkv = torch.einsum("btd,shd->btsh", x, self.qkv_proj)
        q, k, v = qkv.chunk(-2)
        logits = torch.einsum("bth,bsh->bts", q, k)
        # mask
        logits.masked_fill()
        import pdb; pdb.set_trace()


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

