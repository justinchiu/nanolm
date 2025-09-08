import torch


class KvCache:
    def __init__(
        self, batchsize: int, nblocks: int, maxlen: int, nheads: int, hidden: int
    ):
        self.maxlen = maxlen
        self.kcache = torch.zeros(batchsize, maxlen, nblocks, nheads, hidden)
        self.vcache = torch.zeros(batchsize, maxlen, nblocks, nheads, hidden)
        self.lengths = torch.zeros(batchsize, nblocks, dtype=torch.int32)

    def extend(
        self, new_k: torch.Tensor, new_v: torch.Tensor, ids: torch.Tensor, blockidx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # for simplicity, only one block at a time!
        self.kcache[ids, self.lengths[ids, blockidx], blockidx] = new_k[:, -1]
        self.vcache[ids, self.lengths[ids, blockidx], blockidx] = new_v[:, -1]
        self.lengths[ids, blockidx] += 1
        maxlen = self.lengths[ids, blockidx].max()
        return (
            self.kcache[ids, :maxlen, blockidx],
            self.vcache[ids, :maxlen, blockidx],
            self.lengths[ids, blockidx],
        )


def print_generations(generations: list[list[int]], tokenizer):
    for gen in generations:
        print(tokenizer.decode(gen))


@torch.no_grad()
def sample(
    num_steps: int,
    batch_size: int,
    input_ids: list[torch.Tensor],
    kvcache: KvCache,
    model: torch.nn.Module,
    eos_id: int,
    tokenizer,
) -> list[list[int]]:
    """For simplicity, iterate over all num_steps.
    Prefill will be done one step at a time. Inefficient but easier to interleave prefill and generation.
    Always check whether we are in prefill or generation at each timestep and token.
    """

    not_started = [True for _ in range(len(input_ids))]
    completed = [False for _ in range(len(input_ids))]
    generations = [[int(input_ids[x][0].item())] for x in range(len(input_ids))]
    prefix_lens = [len(x) for x in input_ids]

    batch_ids = torch.tensor([x[0] for x in input_ids[:batch_size]], dtype=torch.long)[
        :, None
    ]
    current_seqs = list(range(batch_size))
    for seq in current_seqs:
        not_started[seq] = False
    for i in range(num_steps):
        # always compute logprobs and sample
        output = model(batch_ids, None, kvcache, current_seqs)
        next_token_logprobs = output.logprobs[:, -1]
        next_tokens = torch.multinomial(next_token_logprobs.exp(), num_samples=1)

        # check if generation or prefill
        # TODO: can be batched with tensor ops
        for batchidx in range(batch_size):
            seqid = current_seqs[batchidx]
            genlen = len(generations[seqid])
            if completed[seqid]:
                continue
            elif genlen >= prefix_lens[seqid]:
                generations[seqid].append(int(next_tokens[batchidx].item()))
            else:
                true_token = input_ids[seqid][genlen]
                generations[seqid].append(int(true_token.item()))
                next_tokens[batchidx] = true_token

        # Debug: print next tokens
        print()
        print_generations(generations, tokenizer)

        # check if any sequences finished
        is_complete = (next_tokens == eos_id) | (next_tokens == 0)
        for batchidx in range(batch_size):
            if is_complete[batchidx]:
                seqid = current_seqs[batchidx]
                completed[seqid] = True

        # assign new sequences for prefill
        for batchidx in range(batch_size):
            if is_complete[batchidx] and any(not_started):
                seqid = not_started.index(True)
                not_started[seqid] = False
                current_seqs[batchidx] = seqid
                # populate next_tokens, which must exist (at least bos)
                next_tokens[batchidx] = input_ids[seqid][0]

        if all(completed):
            break

        # Set next inputs
        batch_ids = next_tokens
    return generations


if __name__ == "__main__":
    kvcache = KvCache(4, 1, 8, 1, 3)
    kvcache.extend(
        torch.ones(2, 1, 1, 3), torch.ones(2, 1, 1, 3), torch.tensor([1, 2]), blockidx=0
    )

    # test cases
    assert kvcache.kcache.sum() == 6
    assert (kvcache.kcache[1, 0] == 1).all()
    assert (kvcache.kcache[2, 0] == 1).all()
    assert kvcache.kcache[0].sum() == 0
    assert kvcache.kcache[3].sum() == 0
