import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from nanolm.modules import Transformer, AttentionOutput
from nanolm.data import get_dataloader


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
    batch: torch.Tensor,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    step: int = 0,
) -> dict:
    """Single training step with gradient accumulation"""

    batch = batch.to(device)  # this should be done in dataloader
    input_seq, target_seq = batch[:, :-1], batch[:, 1:]

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
    checkpoint_path: Path,
    metrics: Optional[dict] = None,
):
    """Save model checkpoint"""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }

    if metrics:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> tuple[int, int]:
    """Load model and optimizer state from checkpoint.

    Returns:
        tuple: (epoch, step) from the checkpoint
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    metrics = checkpoint.get("metrics", {})

    print(f"Loaded checkpoint from epoch {epoch}, step {step}")
    if metrics:
        print(f"Checkpoint metrics: {metrics}")

    return epoch, step


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
    checkpoint_dir: Path = Path("./checkpoints"),
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
                checkpoint_path = (
                    checkpoint_dir / f"checkpoint_epoch{epoch}_step{global_step}.pt"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    checkpoint_path,
                    {"loss": metrics["loss"], "perplexity": metrics["perplexity"]},
                )

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
    final_checkpoint_path = checkpoint_dir / "final" / "checkpoint_final.pt"
    save_checkpoint(
        model,
        optimizer,
        num_epochs - 1,
        global_step,
        final_checkpoint_path,
        final_metrics,
    )

    return model


if __name__ == "__main__":
    # Example usage
    model = train(
        # Small model for testing
        n_blocks=4,
        dim=256,
        n_heads=4,
        max_seq_len=256,
        # Small batches for testing
        batch_size=8,
        grad_accum_steps=2,
        learning_rate=1e-3,
        num_epochs=2,
        # Limited data for testing
        max_train_samples=1000,
        max_val_samples=100,
        # No wandb by default
        eval_every=50,
        save_every=100,
    )
