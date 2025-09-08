import torch
from torch import Tensor, nn
import torch.nn.functional as F
from pydantic import BaseModel

from nanolm.sample import KvCache


class TransformerOutput(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    logprobs: Tensor
    attentions: list[Tensor]
    lengths: list[int]


class Rotary(nn.Module):
    """Llama implementation"""

    def __init__(self, dim: int, maxseqlen: int, theta: float = 10_000):
        super().__init__()
        self.dim = dim
        self.maxseqlen = maxseqlen
        # angular speed
        speed = theta ** -(torch.arange(0, dim, 2)[: dim // 2].float() / dim)
        t = torch.arange(maxseqlen)
        # rotations at each timestep
        freqs = torch.outer(t, speed)
        # time x dim//2 (x complex)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, x: Tensor, shift: Tensor | None) -> Tensor:
        """Apply rotations at each timesteps"""
        B, T, heads, dim = x.shape
        # batch x time x nheads x head_dim (x complex)
        xc = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
        if shift is None:
            # if sequence below max timestep, get up to :T
            return torch.view_as_real(xc * self.freqs_cis[None, :T, None, :]).flatten(
                -2
            )
        else:
            # shift by the required indices when encoding
            import pdb

            pdb.set_trace()
            return torch.view_as_real(xc * self.freqs_cis[None, :T, None, :]).flatten(
                -2
            )


class SlowMha(nn.Module):
    """Self attention only"""

    def __init__(
        self,
        dim: int,
        nheads: int,
        maxseqlen: int,
        blockidx: int,
        head_dim: int = 128,
        pos: bool = True,
    ):
        super().__init__()
        self.blockidx = blockidx
        self.nheads = nheads
        self.head_dim = head_dim
        self.pos = pos
        hdim = nheads * head_dim
        std = 0.5 * (dim**-0.5)
        bound = (3**0.5) * std
        self.qkv_proj = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.rotary = Rotary(head_dim, maxseqlen)
        # causal_mask[target, source]: can target attend to source
        # should be lower triangular
        self.causal_mask = nn.Buffer(torch.ones(maxseqlen, maxseqlen, dtype=torch.bool))
        for i in range(1, maxseqlen):
            self.causal_mask.diagonal(i).logical_not_()
        self.out_proj = nn.Linear(hdim, dim, bias=False)

    def forward(
        self, x: Tensor, kvcache: KvCache | None, seqids: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        B, T, D = x.shape
        qkv = torch.einsum("btd,ahd->btah", x, self.qkv_proj)
        q, k, v = qkv.view(B, T, 3 * self.nheads, self.head_dim).chunk(3, dim=-2)

        shift = None
        if kvcache is not None:
            # compute shift
            import pdb

            pdb.set_trace()
        if self.pos:
            q = self.rotary(q, shift)
            k = self.rotary(k, shift)

        # append kv cache
        kvlengths = None
        if kvcache is not None and seqids is not None:
            k, v, kvlengths = kvcache.extend(k, v, seqids, self.blockidx)

        logits = torch.einsum("btnh,bsnh->bnts", q, k) * (self.head_dim**-0.5)
        masked_logits = logits.masked_fill(
            self.causal_mask.logical_not()[:T, :T], float("-inf")
        )
        attn = masked_logits.softmax(-1)
        output = torch.einsum("bnts,bsnd->btnd", attn, v)

        if kvcache is None:
            # reshape is bad
            return self.out_proj(output.reshape(B, T, -1)), attn, kvlengths
        else:
            return self.out_proj(output.reshape(B, T, -1)[:, -1]), attn, kvlengths


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


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],))


class Block(nn.Module):
    def __init__(self, dim: int, nheads: int, maxseqlen: int, blockidx: int):
        super().__init__()
        self.mha = SlowMha(dim, nheads, maxseqlen, blockidx=blockidx, head_dim=dim)
        self.ffn = Ffn(dim)

    def forward(
        self, x: Tensor, kvcache: KvCache | None, seqids: list[int] | None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        attn_out, attn, lengths = self.mha(norm(x), kvcache, seqids)
        x = x + attn_out
        x = x + self.ffn(norm(x))
        return x, attn, lengths


class Transformer(nn.Module):
    def __init__(self, vocab: int, nblocks: int, dim: int, nheads: int, maxseqlen: int):
        super().__init__()
        self.vocab = vocab
        self.nblocks = nblocks
        self.dim = dim
        self.nheads = nheads
        self.maxseqlen = maxseqlen

        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, nheads, maxseqlen, i) for i in range(nblocks)]
        )
        self.output_proj = nn.Linear(dim, vocab, bias=False)
        self.output_proj.weight.data.zero_()

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Tensor | None,
        kvcache: KvCache | None,
        seqids: list[int] | None,
    ) -> TransformerOutput:
        x = self.emb(input_seq)
        attns = []
        all_lengths = []
        for block in self.blocks:
            x, attn, lengths = block(x, kvcache, seqids)
            attns.append(attn)
            all_lengths.append(lengths)
        x = self.output_proj(x).log_softmax(-1)
        logprobs = x.log_softmax(-1)
        if target_seq is not None:
            # ideally use sparse softmax
            logprobs = x.gather(-1, target_seq[:, :, None]).squeeze()
        # lp = F.cross_entropy(x.view(-1, x.shape[-1]), target_seq.view(-1), reduction="none")
        # assert torch.allclose(logprobs, lp)
        return TransformerOutput(
            logprobs=logprobs, attentions=attns, lengths=all_lengths
        )


if __name__ == "__main__":
    transformer = Transformer(128, 2, 8, 2, 5)
    transformer(torch.arange(5)[None], torch.arange(1, 6)[None])
