#!/usr/bin/env python
"""Main training script with checkpoint detection."""

import torch
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path
import tiktoken

from nanolm.modules import Transformer
from nanolm.train import (
    register_gradient_clipping,
    train_step,
    save_checkpoint,
    load_checkpoint,
)
from nanolm.data import TextDataset
from nanolm.sample import sample, KvCache


# 10 simple sentences for training
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


def main():
    # Configuration
    CHECKPOINT_DIR = Path("./checkpoints")
    CHECKPOINT_PATH = CHECKPOINT_DIR / "simple_model.pt"
    NUM_EPOCHS = 100
    SEQ_LEN = 32
    BATCH_SIZE = 10
    LEARNING_RATE = 1e-3

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create tokenizer and datasets
    tokenizer = tiktoken.get_encoding("gpt2")

    # Repeat sentences to have more tokens
    train_sentences = SENTENCES * 10  # Repeat 10 times for more data
    train_dataset = TextDataset(train_sentences, seq_len=SEQ_LEN, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    # Create model
    model = Transformer(50257, nblocks=2, dim=128, nheads=2, maxseqlen=SEQ_LEN)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters ({num_params / 1e6:.2f}M)")

    do_training = True

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Check for existing checkpoint
    start_epoch = 0
    if CHECKPOINT_PATH.exists():
        print(f"\nFound existing checkpoint at {CHECKPOINT_PATH}")
        try:
            epoch, step = load_checkpoint(CHECKPOINT_PATH, model, optimizer, device)
            start_epoch = epoch + 1
            if start_epoch >= NUM_EPOCHS:
                print(f"Training already complete (epoch {epoch}/{NUM_EPOCHS})")
                do_training = False
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh training")
            start_epoch = 0
    else:
        print("\nNo checkpoint found, starting fresh training")

    if do_training:
        # Register gradient clipping
        register_gradient_clipping(model, max_norm=5.0)

        # Training loop
        print(f"\nTraining on {len(SENTENCES)} sentences for {NUM_EPOCHS} epochs...")
        print(f"Starting from epoch {start_epoch}")

        global_step = start_epoch * len(train_loader)

        for epoch in range(start_epoch, NUM_EPOCHS):
            model.train()
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                metrics = train_step(model, batch, optimizer, device, 1, global_step)
                epoch_losses.append(metrics["loss"])
                global_step += 1

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                save_checkpoint(model, optimizer, epoch, global_step, CHECKPOINT_PATH)

        # Save final checkpoint
        save_checkpoint(model, optimizer, NUM_EPOCHS - 1, global_step, CHECKPOINT_PATH)
        print("\nTraining complete!")
    else:
        print("\nSkipping training")

    # Simple generation test
    model.eval()

    # get the first two words of each
    test_text = [" ".join(x.split()[:2]) for x in SENTENCES]
    # Join all test texts into a single string for encoding
    test_text_str = "\n".join(test_text)
    tokens = [
        torch.tensor(tokenizer.encode(x), dtype=torch.long).to(device)
        for x in test_text
    ]

    eos_id = tokenizer.encode(".")[0]

    print(f"\nGenerating text from prompt: '{test_text_str}'")
    kvcache = KvCache(
        batchsize=len(test_text),
        maxlen=model.maxseqlen,
        hidden=model.dim,
        nblocks=model.nblocks,
        nheads=model.nheads,
    )
    generations = sample(
        num_steps=20,
        batch_size=4,
        input_ids=tokens,
        kvcache=kvcache,
        model=model,
        eos_id=eos_id,
        tokenizer=tokenizer,
    )
    generated_text = [tokenizer.decode(tokens) for tokens in generations]
    print("Generated:")
    for text in generated_text:
        print(text)


if __name__ == "__main__":
    main()
