import torch
from torch import nn
from nanolm.modules_dist import InputParallelLinear, OutputParallelLinear


def test_input_parallel_forward():
    torch.manual_seed(42)
    indim, outdim = 6, 8
    shards = 2

    # Create regular linear layer
    linear = nn.Linear(indim, outdim, bias=False)

    # Create input parallel version
    ilinear = InputParallelLinear(indim, outdim, shards)
    ilinear.set_weights(linear.weight.T)

    # Test forward pass
    x = torch.randn(4, indim)
    y_regular = linear(x)
    y_parallel = ilinear(x)

    assert torch.allclose(y_regular, y_parallel, atol=1e-6)


def test_output_parallel_forward():
    torch.manual_seed(42)
    indim, outdim = 6, 8
    shards = 2

    # Create regular linear layer
    linear = nn.Linear(indim, outdim, bias=False)

    # Create output parallel version
    olinear = OutputParallelLinear(indim, outdim, shards)
    olinear.set_weights(linear.weight.T)

    # Test forward pass
    x = torch.randn(4, indim)
    y_regular = linear(x)
    y_parallel = olinear(x)

    assert torch.allclose(y_regular, y_parallel, atol=1e-6)


def test_input_parallel_gradients():
    torch.manual_seed(42)
    indim, outdim = 6, 8
    shards = 2

    # Create regular linear layer
    linear = nn.Linear(indim, outdim, bias=False)

    # Create input parallel version
    ilinear = InputParallelLinear(indim, outdim, shards)
    ilinear.set_weights(linear.weight.T)

    # Test data
    x = torch.randn(4, indim, requires_grad=True)
    x_parallel = x.clone().detach().requires_grad_(True)

    # Forward pass
    y_regular = linear(x)
    y_parallel = ilinear(x_parallel)
    y_parallel

    # Backward pass
    loss_regular = y_regular.sum()
    loss_parallel = y_parallel.sum()

    loss_regular.backward()
    loss_parallel.backward()

    # Compare input gradients
    assert torch.allclose(x.grad, x_parallel.grad, atol=1e-6)  # pyright: ignore

    # Compare weight gradients
    expected_weight_grad = linear.weight.grad.T  # pyright: ignore
    actual_weight_grad = torch.cat([w.grad for w in ilinear.weights], dim=0)  # pyright: ignore
    assert torch.allclose(expected_weight_grad, actual_weight_grad, atol=1e-6)

    grads = ilinear.backward(x_parallel, torch.ones_like(y_parallel))
    assert torch.allclose(x_parallel.grad, grads[0])  # pyright: ignore
    for W, dW in zip(ilinear.weights, grads[1:]):
        assert torch.allclose(W.grad, dW)  # pyright: ignore


def test_output_parallel_gradients():
    torch.manual_seed(42)
    indim, outdim = 6, 8
    shards = 2

    # Create regular linear layer
    linear = nn.Linear(indim, outdim, bias=False)

    # Create output parallel version
    olinear = OutputParallelLinear(indim, outdim, shards)
    olinear.set_weights(linear.weight.T)

    # Test data
    x = torch.randn(4, indim, requires_grad=True)
    x_parallel = x.clone().detach().requires_grad_(True)

    # Forward pass
    y_regular = linear(x)
    y_parallel = olinear(x_parallel)
    y_parallel.retain_grad()

    # Backward pass
    loss_regular = y_regular.sum()
    loss_parallel = y_parallel.sum()

    loss_regular.backward()
    loss_parallel.backward()

    # Compare input gradients
    assert torch.allclose(x.grad, x_parallel.grad, atol=1e-6)  # pyright: ignore

    # Compare weight gradients
    expected_weight_grad = linear.weight.grad.T  # pyright: ignore
    actual_weight_grad = torch.cat([w.grad for w in olinear.weights], dim=1)  # pyright: ignore
    assert torch.allclose(expected_weight_grad, actual_weight_grad, atol=1e-6)

    grads = olinear.backward(x_parallel, torch.ones_like(y_parallel))
    assert torch.allclose(x_parallel.grad, grads[0])  # pyright: ignore
    for W, dW in zip(olinear.weights, grads[1:]):
        assert torch.allclose(W.grad, dW)  # pyright: ignore
