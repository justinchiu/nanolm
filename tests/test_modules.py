import torch
from torch import nn
from nanolm.modules import Mha


def test_mha():
    B,T,D = (2, 4, 3)
    nheads = 2
    x = torch.rand(B,T,D)

    mha_torch = nn.MultiheadAttention(D, nheads)

    attn_output, attn_weights = mha_torch(x, x, x)
    import pdb; pdb.set_trace()

