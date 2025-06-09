"""Implementations of sharded FFN and MHA
FFN:
* Input-parallel
* Output-parallel
MHA:
* Sequence-parallel

Follows Megatron
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Ffn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.proj1 = nn.Linear(dim, hdim)
        self.proj2 = nn.Linear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj1(x)
        x = F.relu(x)
        x = self.proj2(x)
        return x


class InputParallelLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, shards: int):
        super().__init__()
        self.weights = [
            nn.Parameter(torch.randn(in_dim // shards, out_dim)) for n in range(shards)
        ]

    def set_weights(self, W: Tensor):
        shards = len(self.weights)
        for i, weight in enumerate(W.chunk(shards, dim=0)):
            self.weights[i].data.copy_(weight)

    def forward(self, x: Tensor) -> Tensor:
        outputs = [
            x @ W
            for x, W in zip(
                x.chunk(len(self.weights), dim=-1),
                self.weights,
            )
        ]
        return torch.stack(outputs, dim=0).sum(0)

    @torch.no_grad()
    def backward(self, x: Tensor, grad_output: Tensor) -> list[Tensor]:
        shards = x.chunk(len(self.weights), dim=-1)
        x_grads = []
        W_grads = []
        for xi, W in zip(shards, self.weights):
            W_grads.append(xi.T @ grad_output)
            x_grads.append(grad_output @ W.T)
        x_grad = torch.cat(x_grads, dim=-1)
        return [x_grad] + W_grads


class OutputParallelLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, shards: int):
        super().__init__()
        self.weights = [
            nn.Parameter(torch.randn(in_dim, out_dim // shards)) for n in range(shards)
        ]

    def set_weights(self, W: Tensor):
        shards = len(self.weights)
        for i, weight in enumerate(W.chunk(shards, dim=1)):
            self.weights[i].data.copy_(weight)

    def forward(self, x: Tensor) -> Tensor:
        outputs = [x @ W for W in self.weights]
        return torch.cat(outputs, dim=-1)

    @torch.no_grad()
    def backward(self, x: Tensor, grad_output: Tensor) -> list[Tensor]:
        grad_shards = grad_output.chunk(len(self.weights), dim=-1)
        x_grad = torch.zeros_like(x)
        W_grads = []
        for dW, W in zip(grad_shards, self.weights):
            W_grads.append(x.T @ dW)
            x_grad.add_(dW @ W.T)
        return [x_grad] + W_grads
