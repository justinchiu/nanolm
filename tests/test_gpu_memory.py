import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


def get_gpu_memory_mb():
    """Get current GPU memory in MB"""
    return torch.cuda.memory_allocated() / 1024**2


def measure_gpu_memory(model_id="Qwen/Qwen2.5-1.5B", seq_length=512, batch_size=1):
    """Simple GPU memory measurement without profiler"""

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    # Clear everything
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"Model: {model_id}")
    print(f"Sequence length: {seq_length}")
    print(f"Batch size: {batch_size}")
    print("-" * 50)

    # Initial
    initial = get_gpu_memory_mb()
    print(f"Initial GPU memory: {initial:.2f} MB")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.train()

    after_model = get_gpu_memory_mb()
    print(f"After loading model: {after_model:.2f} MB")
    print(f"Model size: {after_model - initial:.2f} MB")

    # Tokenizer and inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    text = "The quick brown fox jumps over the lazy dog. " * (seq_length // 10)
    inputs = tokenizer(
        [text] * batch_size,
        max_length=seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    # Clear gradients
    model.zero_grad()
    torch.cuda.synchronize()

    before_forward = get_gpu_memory_mb()
    print(f"\nBefore forward pass: {before_forward:.2f} MB")

    # Forward pass
    print("Running forward pass...")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    torch.cuda.synchronize()

    after_forward = get_gpu_memory_mb()
    print(f"After forward pass: {after_forward:.2f} MB")
    print(f"Forward pass increase: {after_forward - before_forward:.2f} MB")

    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()
    torch.cuda.synchronize()

    after_backward = get_gpu_memory_mb()
    print(f"After backward pass: {after_backward:.2f} MB")
    print(f"Backward pass increase: {after_backward - after_forward:.2f} MB")

    # Peak memory
    peak = torch.cuda.max_memory_allocated() / 1024**2

    # Calculate gradient size
    grad_size = (
        sum(
            p.grad.element_size() * p.grad.nelement()
            for p in model.parameters()
            if p.grad is not None
        )
        / 1024**2
    )

    print("\n" + "=" * 50)
    print("MEMORY SUMMARY")
    print("=" * 50)
    print(f"Model parameters: {after_model - initial:.2f} MB")
    print(f"Forward activations: {after_forward - before_forward:.2f} MB")
    print(f"Backward gradients: {after_backward - after_forward:.2f} MB")
    print(f"Calculated gradient size: {grad_size:.2f} MB")
    print(f"Peak memory usage: {peak:.2f} MB")
    print(f"Current allocated: {after_backward:.2f} MB")

    # Test different sequence lengths
    if seq_length == 512:
        print("\n" + "=" * 50)
        print("TESTING DIFFERENT SEQUENCE LENGTHS")
        print("=" * 50)

        for test_seq_len in [128, 256, 512, 1024, 2048]:
            torch.cuda.empty_cache()
            gc.collect()

            test_inputs = tokenizer(
                ["test"] * batch_size,
                max_length=test_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to("cuda")

            before = get_gpu_memory_mb()
            with torch.no_grad():
                test_out = model(**test_inputs)
            torch.cuda.synchronize()
            after = get_gpu_memory_mb()

            print(
                f"Seq length {test_seq_len:4d}: {after - before:7.2f} MB activation memory"
            )

            del test_out, test_inputs
            torch.cuda.empty_cache()


def measure_inference_memory(
    model_id="Qwen/Qwen2.5-1.5B", seq_length=512, batch_size=1
):
    """Measure GPU memory for inference only (no gradients)"""

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    # Clear everything
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 50)
    print("INFERENCE MODE (NO GRADIENTS)")
    print("=" * 50)
    print(f"Model: {model_id}")
    print(f"Sequence length: {seq_length}")
    print(f"Batch size: {batch_size}")
    print("-" * 50)

    # Initial
    initial = get_gpu_memory_mb()
    print(f"Initial GPU memory: {initial:.2f} MB")

    # Load model in eval mode
    print("\nLoading model in eval mode...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()  # Set to eval mode

    after_model = get_gpu_memory_mb()
    print(f"After loading model: {after_model:.2f} MB")
    print(f"Model size: {after_model - initial:.2f} MB")

    # Tokenizer and inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    text = "The quick brown fox jumps over the lazy dog. " * (seq_length // 10)
    inputs = tokenizer(
        [text] * batch_size,
        max_length=seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    torch.cuda.synchronize()
    before_forward = get_gpu_memory_mb()
    print(f"\nBefore forward pass: {before_forward:.2f} MB")

    # Forward pass WITHOUT gradients
    print("Running forward pass (no grad)...")
    with torch.no_grad():
        model(**inputs)
    torch.cuda.synchronize()

    after_forward = get_gpu_memory_mb()
    print(f"After forward pass: {after_forward:.2f} MB")
    print(f"Forward pass increase: {after_forward - before_forward:.2f} MB")

    # Peak memory
    peak = torch.cuda.max_memory_allocated() / 1024**2

    print("\n" + "=" * 50)
    print("INFERENCE MEMORY SUMMARY")
    print("=" * 50)
    print(f"Model parameters: {after_model - initial:.2f} MB")
    print(f"Forward activations: {after_forward - before_forward:.2f} MB")
    print(f"Peak memory usage: {peak:.2f} MB")
    print(f"Current allocated: {after_forward:.2f} MB")

    # Compare with different sequence lengths
    print("\n" + "=" * 50)
    print("INFERENCE MEMORY VS SEQUENCE LENGTH")
    print("=" * 50)

    for test_seq_len in [128, 256, 512, 1024, 2048]:
        torch.cuda.empty_cache()
        gc.collect()

        test_inputs = tokenizer(
            ["test"] * batch_size,
            max_length=test_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        before = get_gpu_memory_mb()
        with torch.no_grad():
            test_out = model(**test_inputs)
        torch.cuda.synchronize()
        after = get_gpu_memory_mb()

        print(f"Seq length {test_seq_len:4d}: {after - before:7.2f} MB")

        del test_out, test_inputs
        torch.cuda.empty_cache()

    print(model)


if __name__ == "__main__":
    # First run training memory test
    print("=" * 70)
    print("TRAINING MODE TEST (WITH GRADIENTS)")
    print("=" * 70)
    measure_gpu_memory(model_id="Qwen/Qwen2.5-1.5B", seq_length=512, batch_size=1)

    # Clear GPU memory between tests
    torch.cuda.empty_cache()
    gc.collect()

    # Then run inference memory test
    print("\n" * 2)
    print("=" * 70)
    print("INFERENCE MODE TEST (NO GRADIENTS)")
    print("=" * 70)
    measure_inference_memory(model_id="Qwen/Qwen2.5-1.5B", seq_length=512, batch_size=1)
