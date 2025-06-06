import torch
from nanolm.modules import Mha


def test_mha():
    B,T,D = (2, 4, 3)
    nheads = 2
    x = torch.rand(B,T,D)

