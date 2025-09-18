# CUDA Kernel Debugging Guide

This guide provides detailed methodology for debugging custom CUDA kernels when integrating them with PyTorch's autograd system. It's designed for developers who need to extend or modify the CUDA operations in this codebase.

## Core Debugging Methodology

When your CUDA implementation breaks, follow this systematic approach:

### Step 1: Establish PyTorch Baseline

**Always start here** - replace all custom CUDA operations with PyTorch equivalents:

```python
# In your model definition, temporarily replace:
# self.custom_op = CustomOp()  # CUDA version
self.custom_op = torch.nn.functional.some_op  # PyTorch equivalent
```

**Why this works:**
- PyTorch operations are thoroughly tested and reliable
- Eliminates CUDA-specific bugs from the equation
- Gives you a "known good" reference implementation

### Step 2: Incremental Replacement Testing

Once you have a working PyTorch baseline, replace operations one at a time:

```python
# Test 1: Replace embedding
# self.embedding = nn.Embedding(...)  # PyTorch
self.embedding = Embedding(...)       # Custom CUDA

# Test training - if it works, move to next operation
# If it breaks, you found your buggy kernel!
```

**Key principle:** Only one variable changes per test. This isolates exactly which operation is problematic.

### Step 3: Isolate and Fix

When you find a failing operation:

1. **Check tensor shapes** - CUDA kernels are sensitive to exact tensor dimensions
2. **Verify memory layout** - Use `.contiguous()` calls before CUDA operations
3. **Check kernel launch parameters** - Grid/block dimensions must match kernel expectations
4. **Add debug prints** - Log tensor shapes and values before/after operations
5. **Test backward pass** - CUDA bugs often manifest during gradient computation

## Common CUDA Kernel Bugs

### Memory Corruption Issues

**Symptoms:**
- `CUBLAS_STATUS_EXECUTION_FAILED`
- `illegal memory access`
- Random crashes or incorrect results
- Failures in downstream operations

**Causes:**
- Incorrect tensor indexing in kernels
- Missing `.contiguous()` calls on non-contiguous tensors
- Buffer overflows in kernel code
- Race conditions in parallel operations

**Debugging:**
```python
# Add before CUDA operations:
assert tensor.is_contiguous(), f"Tensor not contiguous: {tensor.shape}"
print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
```

### Broadcasting Problems

**Symptoms:**
- Operations work with some tensor shapes but fail with others
- Inconsistent results across batch sizes

**Common culprit:** Custom Add/Mul operations that don't handle PyTorch broadcasting rules.

**Fix:**
```python
# Instead of custom Add, use:
a_broadcast, b_broadcast = torch.broadcast_tensors(a, b)
out = torch.add(a_broadcast, b_broadcast)
```

### Backward Pass Issues

**Symptoms:**
- Forward pass works, but training fails during `loss.backward()`
- Gradients are `None` or incorrect

**Debugging:**
```python
# Check gradients after backward:
loss.backward()
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}")
    elif torch.isnan(param.grad).any():
        print(f"NaN gradients in {name}")
```

## Kernel-Specific Debugging

### Matrix Multiplication (MatMul)

**2D vs 3D variants:**
- **MatMul**: Expects 2D tensors `(M, N)` and `(N, K)` → `(M, K)`
- **BatchedMatMul**: Expects 3D tensors `(B, M, N)` and `(B, N, K)` → `(B, M, K)`

**Common issues:**
- Wrong dimension ordering (PyTorch uses row-major, some CUDA code assumes column-major)
- Incorrect grid/block size calculations for large matrices

### Layer Normalization

**Debugging tips:**
- Check that `mean` and `variance` tensors have correct shapes
- Verify epsilon value matches PyTorch's default (1e-5)
- Ensure proper broadcasting of gamma/beta parameters

### Attention Operations

**Common issues:**
- Incorrect transpose operations (`.transpose(-2, -1)` vs `.t()`)
- Causal masking implementation bugs
- Scale factor calculation errors (`head_size ** -0.5`)

## Testing Strategies

### Unit Tests for Individual Operations

Create minimal test scripts for each operation:

```python
# test_embedding.py
import torch
from custom_ops import Embedding

# Create test tensors
vocab_size, embed_dim = 100, 32
indices = torch.randint(0, vocab_size, (4, 10))  # batch_size=4, seq_len=10

# Test forward
embed = Embedding(vocab_size, embed_dim)
output = embed(indices)
assert output.shape == (4, 10, 32)

# Test backward
loss = output.sum()
loss.backward()
assert embed.weight.grad is not None

print("Embedding test passed!")
```

### Numerical Verification

Compare custom CUDA outputs with PyTorch equivalents:

```python
# Test numerical accuracy
torch_result = torch.nn.functional.layer_norm(x, (x.shape[-1],))
custom_result = custom_layer_norm(x)

max_diff = torch.abs(torch_result - custom_result).max()
print(f"Max difference: {max_diff}")

# Should be very small (< 1e-5 for float32)
assert max_diff < 1e-4, f"Numerical accuracy test failed: {max_diff}"
```

## Performance Debugging

### Profiling CUDA Kernels

Use PyTorch's profiler to identify bottlenecks:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Your training loop here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory Usage Analysis

Check for memory leaks or excessive usage:

```python
# Monitor GPU memory
print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
```

## Best Practices

### 1. Always Test Incrementally
Never implement multiple CUDA operations before testing. Add and test one at a time.

### 2. Use PyTorch as Reference
PyTorch operations are your ground truth. Custom CUDA should match PyTorch results exactly.

### 3. Check Tensor Properties
Always verify:
- Tensor shapes match expectations
- Data types are correct (float32)
- Device placement (CUDA)
- Memory contiguity

### 4. Start Simple, Then Optimize
Begin with naive but correct CUDA implementations. Optimize only after correctness is verified.

### 5. Document Your Changes
When you add or modify CUDA kernels, document:
- Expected input/output shapes
- Any assumptions about tensor layout
- Known limitations or edge cases

## Advanced Debugging Tools

### CUDA Device-Side Assertions

Enable device-side assertions for better error messages:

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python your_script.py
```

### Nsight Systems/Compute

For detailed profiling:
```bash
nsys profile --stats=true python train.py
ncu --set full python train.py  # Nsight Compute for kernel analysis
```

### Memory Sanitizers

Use CUDA's memory checking tools:
```bash
cuda-memcheck python your_script.py
```

## Troubleshooting Common Issues

### "No kernel image available" errors
- Check GPU architecture compatibility (sm_86 for RTX 30xx)
- Verify CUDA toolkit version matches PyTorch build

### Random crashes during training
- Add `torch.cuda.synchronize()` calls to isolate async errors
- Check for race conditions in custom backward passes

### Performance worse than PyTorch
- Profile both implementations
- Check memory transfer overhead
- Verify kernel launch parameters are optimal

## Getting Help

When debugging complex CUDA issues:

1. **Isolate the problem** using the methodology above
2. **Search PyTorch issues** for similar problems
3. **Check CUDA programming forums** for kernel-specific issues
4. **Use minimal reproducers** when asking for help

Remember: CUDA debugging is challenging, but systematic isolation makes even complex issues solvable. Start with PyTorch baselines, test incrementally, and verify numerical accuracy at each step.
