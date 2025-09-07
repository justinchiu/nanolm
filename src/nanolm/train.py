import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import json
from typing import Optional
import tiktoken

from nanolm.modules import Transformer, AttentionOutput


class TextDataset:
    """Simple text dataset using tiktoken for tokenization"""

    def __init__(self, texts: list[str], seq_len: int, tokenizer):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Tokenize all texts and concatenate
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self):
        return max(1, len(self.tokens) // self.seq_len - 1)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target
        chunk = self.tokens[start:end]

        if len(chunk) < self.seq_len + 1:
            # Pad if necessary
            chunk = torch.cat(
                [chunk, torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)]
            )

        return chunk[:-1], chunk[1:]  # input, target


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
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Extract texts
    texts = [item["text"] for item in dataset]

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


def register_gradient_clipping(model: nn.Module, max_norm: float = 1.0):
    """Register gradient clipping hooks on model parameters"""

    def clip_grad(grad):
        grad_norm = grad.norm(2)
        if grad_norm > max_norm:
            return grad * (max_norm / grad_norm)
        return grad

    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(clip_grad)


def train_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    step: int = 0,
) -> dict:
    """Single training step with gradient accumulation"""

    input_seq, target_seq = batch
    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    # Forward pass
    output: AttentionOutput = model(input_seq, target_seq)

    # Calculate loss (negative log likelihood)
    loss = -output.logprobs.mean() / grad_accum_steps

    # Backward pass (gradients are clipped via hooks)
    loss.backward()

    # Update weights if accumulation complete
    if (step + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    return {
        "loss": loss.item() * grad_accum_steps,
        "perplexity": torch.exp(-output.logprobs.mean()).item(),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> dict:
    """Evaluate model on validation set"""

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, (input_seq, target_seq) in enumerate(dataloader):
            if i >= max_batches:
                break

            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            output: AttentionOutput = model(input_seq, target_seq)

            batch_loss = -output.logprobs.sum().item()
            batch_tokens = output.logprobs.numel()

            total_loss += batch_loss
            total_tokens += batch_tokens

    model.train()

    avg_loss = total_loss / total_tokens
    return {"val_loss": avg_loss, "val_perplexity": np.exp(avg_loss)}


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    step: int,
    metrics: dict,
    checkpoint_dir: Path,
):
    """Save model checkpoint"""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save config
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return checkpoint_path


def train(
    # Model config
    vocab_size: int = 50257,  # GPT-2 tokenizer size
    n_blocks: int = 6,
    dim: int = 384,
    n_heads: int = 6,
    max_seq_len: int = 512,
    # Training config
    batch_size: int = 32,
    grad_accum_steps: int = 4,
    learning_rate: float = 3e-4,
    num_epochs: int = 10,
    warmup_steps: int = 1000,
    # Data config
    dataset_name: str = "roneneldan/TinyStories",
    max_train_samples: Optional[int] = 100000,
    max_val_samples: Optional[int] = 10000,
    # Logging config
    checkpoint_dir: str = "./checkpoints",
    eval_every: int = 500,
    save_every: int = 1000,
):
    """Main training function"""

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = Transformer(vocab_size, n_blocks, dim, n_heads, max_seq_len)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    # Register gradient clipping hooks
    register_gradient_clipping(model, max_norm=1.0)

    # Learning rate scheduler (cosine with warmup)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (num_epochs * 1000 - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create dataloaders
    train_loader, tokenizer = get_dataloader(
        dataset_name, "train", max_seq_len, batch_size, max_samples=max_train_samples
    )
    val_loader, _ = get_dataloader(
        dataset_name, "validation", max_seq_len, batch_size, max_samples=max_val_samples
    )

    # Training loop
    global_step = 0
    checkpoint_dir = Path(checkpoint_dir)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Train step
            metrics = train_step(
                model, batch, optimizer, device, grad_accum_steps, global_step
            )

            epoch_losses.append(metrics["loss"])

            # Update scheduler
            if (global_step + 1) % grad_accum_steps == 0:
                scheduler.step()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "ppl": f"{metrics['perplexity']:.2f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            # Evaluate
            if global_step % eval_every == 0 and global_step > 0:
                val_metrics = evaluate(model, val_loader, device)
                print(
                    f"\nValidation - Loss: {val_metrics['val_loss']:.4f}, "
                    f"Perplexity: {val_metrics['val_perplexity']:.2f}"
                )

            # Save checkpoint
            if global_step % save_every == 0 and global_step > 0:
                checkpoint_path = save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    {"loss": metrics["loss"], "perplexity": metrics["perplexity"]},
                    checkpoint_dir,
                )
                print(f"\nSaved checkpoint: {checkpoint_path}")

            global_step += 1

        # End of epoch summary
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    # Final evaluation
    final_metrics = evaluate(model, val_loader, device)
    print(
        f"\nFinal Validation - Loss: {final_metrics['val_loss']:.4f}, "
        f"Perplexity: {final_metrics['val_perplexity']:.2f}"
    )

    # Save final model
    final_checkpoint = save_checkpoint(
        model,
        optimizer,
        num_epochs,
        global_step,
        final_metrics,
        checkpoint_dir / "final",
    )
    print(f"Saved final model: {final_checkpoint}")

    return model


if __name__ == "__main__":
    # Simple test with 10 sentences
    SENTENCES = [
        "The cat sat on the mat.",
        "Dogs love to play fetch.",
        "The sun rises in the east.",
        "Birds fly high in the sky.",
        "Fish swim in the ocean.",
        "Trees grow tall in the forest.",
        "Flowers bloom in spring.",
        "Rain falls from the clouds.",
        "Stars shine bright at night.",
        "The moon lights up the darkness.",
    ]
    
    # Create simple dataloaders from sentences  
    tokenizer = tiktoken.get_encoding("gpt2")
    # Repeat sentences to have more tokens
    train_sentences = SENTENCES * 10  # Repeat 10 times for more data
    train_dataset = TextDataset(train_sentences, seq_len=32, tokenizer=tokenizer)
    val_dataset = TextDataset(SENTENCES[-2:] * 5, seq_len=32, tokenizer=tokenizer)  # Last 2 for validation
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=False)
    
    # Small model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(50257, nblocks=2, dim=128, nheads=2, maxseqlen=32)
    model = model.to(device)
    
    # Simple training loop
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    register_gradient_clipping(model, max_norm=5.0)
    
    print(f"Training on {len(SENTENCES)} sentences...")
    for epoch in range(10):
        model.train()
        epoch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            metrics = train_step(model, batch, optimizer, device, 1, batch_idx)
            epoch_losses.append(metrics["loss"])
        
        if (epoch + 1) % 10 == 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch + 1}/100 - Loss: {avg_loss:.4f}")
