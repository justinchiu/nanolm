import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity, record_function
import gc

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def format_memory(bytes_val):
    """Convert bytes to human readable format"""
    return bytes_val / 1024**2


def measure_memory_usage(
    model_id="meta-llama/Llama-3.1-8B", seq_length=20, batch_size=2
):
    """Measure memory usage using PyTorch profiler"""

    device = "cpu"

    # Clear cache and collect garbage
    gc.collect()

    print(f"Device: {device}")
    print(f"Model: {model_id}")
    print(f"Sequence length: {seq_length}")
    print(f"Batch size: {batch_size}")
    print("-" * 50)

    # Load model and tokenizer
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.train()  # Enable training mode for gradients

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Calculate model size
    num_params = sum(p.numel() for p in model.parameters())
    param_memory = num_params * 2 / 1024**2  # bfloat16 = 2 bytes
    print(f"Model parameters: {num_params:,}")
    print(f"Model parameter memory: {param_memory:.2f} MB")

    # Prepare input
    text = "The quick brown fox jumps over the lazy dog. " * (seq_length // 10)
    inputs = tokenizer(
        [text] * batch_size,
        max_length=seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Profile memory usage
    print("\n" + "=" * 50)
    print("PROFILING MEMORY USAGE")
    print("=" * 50)

    activities = [ProfilerActivity.CPU]

    with profile(
        activities=activities, profile_memory=True, record_shapes=True, with_stack=True
    ) as prof:
        # Clear gradients
        model.zero_grad()

        # Forward pass
        with record_function("forward_pass"):
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        # Backward pass
        with record_function("backward_pass"):
            loss.backward()

    # Print profiler results
    print("\nMemory usage by operation:")
    print("-" * 80)

    # Get key averages
    key_averages = prof.key_averages()

    # Separate forward and backward operations
    forward_ops = []
    backward_ops = []
    other_ops = []

    for evt in key_averages:
        if evt.self_cpu_memory_usage > 0:
            if "forward" in evt.key.lower() or "forward_pass" in evt.key:
                forward_ops.append(evt)
            elif (
                "backward" in evt.key.lower()
                or "backward_pass" in evt.key
                or "grad" in evt.key.lower()
            ):
                backward_ops.append(evt)
            else:
                # Check if it's part of forward or backward based on the stack
                if any("backward" in str(s) for s in (evt.stack or [])):
                    backward_ops.append(evt)
                elif any("forward" in str(s) for s in (evt.stack or [])):
                    forward_ops.append(evt)
                else:
                    other_ops.append(evt)

    # Calculate totals
    forward_memory = sum(evt.self_cpu_memory_usage for evt in forward_ops)
    backward_memory = sum(evt.self_cpu_memory_usage for evt in backward_ops)
    other_memory = sum(evt.self_cpu_memory_usage for evt in other_ops)
    total_memory = forward_memory + backward_memory + other_memory

    # Print top memory consuming operations
    print("\nTop memory consuming operations:")
    print("-" * 80)
    print(f"{'Operation':<50} {'Self Memory (MB)':>15} {'# Calls':>10}")
    print("-" * 80)

    # Sort all operations by memory usage
    all_ops = sorted(key_averages, key=lambda x: x.self_cpu_memory_usage, reverse=True)

    for evt in all_ops[:20]:
        if evt.self_cpu_memory_usage > 0:
            print(
                f"{evt.key[:50]:<50} {format_memory(evt.self_cpu_memory_usage):>15.2f} {evt.count:>10}"
            )

    # Print memory summary
    print("\n" + "=" * 50)
    print("MEMORY SUMMARY")
    print("=" * 50)
    print(f"Forward pass memory:  {format_memory(forward_memory):>10.2f} MB")
    print(f"Backward pass memory: {format_memory(backward_memory):>10.2f} MB")
    print(f"Other operations:     {format_memory(other_memory):>10.2f} MB")
    print(f"Total memory used:    {format_memory(total_memory):>10.2f} MB")

    # Calculate gradient memory
    grad_memory = sum(
        p.grad.element_size() * p.grad.nelement()
        for p in model.parameters()
        if p.grad is not None
    )
    print(f"\nGradient memory (calculated): {format_memory(grad_memory):.2f} MB")

    # Memory breakdown by category
    print("\n" + "=" * 50)
    print("DETAILED BREAKDOWN")
    print("=" * 50)

    # Group operations by type
    categories = {}
    for evt in key_averages:
        if evt.self_cpu_memory_usage > 0:
            # Categorize based on operation name
            if "linear" in evt.key.lower():
                category = "Linear layers"
            elif "attention" in evt.key.lower():
                category = "Attention"
            elif "norm" in evt.key.lower():
                category = "Normalization"
            elif "embedding" in evt.key.lower():
                category = "Embeddings"
            elif "softmax" in evt.key.lower():
                category = "Softmax"
            elif "gelu" in evt.key.lower() or "activation" in evt.key.lower():
                category = "Activations"
            elif "matmul" in evt.key.lower() or "mm" in evt.key.lower():
                category = "Matrix multiplication"
            elif "backward" in evt.key.lower():
                category = "Backward operations"
            else:
                category = "Other"

            if category not in categories:
                categories[category] = 0
            categories[category] += evt.self_cpu_memory_usage

    print(f"{'Category':<30} {'Memory (MB)':>15}")
    print("-" * 45)
    for category, memory in sorted(
        categories.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{category:<30} {format_memory(memory):>15.2f}")

    # Export trace for visualization (optional)
    trace_file = "memory_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace exported to {trace_file} (can be viewed in chrome://tracing)")

    return {
        "model_params": num_params,
        "model_memory": param_memory,
        "forward_memory": format_memory(forward_memory),
        "backward_memory": format_memory(backward_memory),
        "total_memory": format_memory(total_memory),
        "gradient_memory": format_memory(grad_memory),
    }


if __name__ == "__main__":
    # Test with different configurations
    results = measure_memory_usage(
        model_id="meta-llama/Llama-3.1-8B", seq_length=512, batch_size=1
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f} MB")
        else:
            print(f"{key}: {value:,}")
