import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict, IterableDataset, IterableDatasetDict
from typing import Optional
import tiktoken


class TextDataset(Dataset):
    """Simple text dataset using tiktoken for tokenization"""

    def __init__(self, texts: list[str], seq_len: int, tokenizer):
        # single long text
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.tokens = [
            torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in texts
        ]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        chunk = self.tokens[idx]
        if len(chunk) < self.seq_len + 1:
            # Pad if necessary
            chunk = torch.cat(
                [chunk, torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)]
            )
        return chunk
        # index into chunk after copying to gpu to save memory
        # return chunk[:-1], chunk[1:]  # input, target
        # technically better to have a persistent buffer(s)


def get_dataloader(
    dataset_name: str = "roneneldan/TinyStories",
    split: str = "train",
    seq_len: int = 2048,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
):
    """Create a dataloader from HuggingFace datasets"""

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Handle different dataset types
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        raise ValueError(f"Expected a single dataset split, got {type(dataset)}")

    if isinstance(dataset, IterableDataset):
        # For iterable datasets, we need to materialize them
        if max_samples:
            texts = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                texts.append(item["text"])
        else:
            texts = [item["text"] for item in dataset]
    else:
        # For regular datasets
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        # Extract texts
        texts = dataset["text"]

    # Create dataset
    text_dataset = TextDataset(texts, seq_len, tokenizer)

    # Create dataloader
    dataloader = DataLoader(
        text_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return dataloader, tokenizer
