# Character-level Transformer: Training & Inference with Custom CUDA Kernels

A minimal character-level GPT implementation that demonstrates integrating custom CUDA kernels into PyTorch's autograd system. This repository contains both **training** and **inference** components with hybrid CUDA implementations - using custom CUDA operations where they work reliably, and PyTorch operations where needed for stability.

## 🏗️ Project Structure

```
naive.cu/
├── src/               # C++ source files and CUDA kernels
│   ├── training/      # Training C++ bindings and kernels
│   └── inference/     # Inference C++ bindings and kernels
├── custom_ops/        # Python wrappers for custom operations
│   ├── training/      # Training operation wrappers
│   └── inference/     # Inference operation wrappers
├── docs/              # Documentation and debugging guides
├── train.py           # Character-level transformer training
├── inference.py       # Transformer inference with dense/MoE
├── setup.py           # Combined build configuration
├── README.md          # This file
└── LICENSE
```

## 📚 Components

### Training Component
- **Character-level GPT training** with custom CUDA kernels
- **Two-stage approach**: PyTorch baseline → Custom CUDA implementation
- **Dataset**: "The Wonderful Wizard of Oz" (public domain)
- **Architecture**: Transformer with multi-head attention and feed-forward layers

### Inference Component
- **Transformer inference** with both dense and MoE architectures
- **KV caching** for efficient autoregressive generation
- **Mixture of Experts (MoE)**: 8 experts, top-2 routing
- **Optimized kernels**: GEMV operations for inference speedup

## 🔧 Setup & Installation

### Prerequisites
- **CUDA-compatible GPU** (RTX 30xx+ recommended)
- **Python 3.8+**
- **PyTorch 2.5+**
- **CUDA Toolkit 12.1+**

### Environment Setup

1. **Create virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   uv pip install torch torchvision torchaudio pybind11 requests
   ```

### Training Setup

1. **Build CUDA extensions:**
   ```bash
   python setup.py build_ext --inplace
   ```

2. **Run training:**
   ```bash
   python train.py
   ```

### Inference Setup

1. **Build CUDA extensions:**
   ```bash
   python setup.py build_ext --inplace
   ```

2. **Run inference:**
   ```bash
   python inference.py
   ```

### Quick Start (Both Components)

```bash
# Clone and setup
git clone <repository-url>
cd naive.cu

# Setup environment
uv venv && source .venv/bin/activate
uv pip install torch torchvision torchaudio pybind11 requests

# Build CUDA extensions
python setup.py build_ext --inplace

# Run training
python train.py

# Run inference
python inference.py
```

## 🎯 What You'll See When Running

### Training Output
```
Using device: cuda
Batch size: 16, Block size: 64, Embedding dim: 128

PyTorch baseline training...
PyTorch Model Parameters: 1,234,496
iter 0/1000 | loss 4.2345
iter 100/1000 | loss 2.3456
iter 200/1000 | loss 1.9876
...
PyTorch training time: 45.67 seconds

Custom CUDA training...
Custom Model Parameters: 1,234,496
iter 0/1000 | loss 4.2345
iter 100/1000 | loss 2.3456
...
Custom CUDA training time: 38.92 seconds
```

### Inference Output
```
=== Transformer Inference Setup (10x Larger Model) ===
Batch size: 1
Block size: 64
Embedding dimension: 768
Number of heads: 8
Number of layers: 12
Vocabulary size: 95
Device: cuda
Max new tokens: 200
Number of experts: 8
Top-k experts: 2

=== PyTorch Dense vs CUDA Dense ===
✓ SUCCESS: Dense implementations match exactly! Speedup: 1.45x

=== PyTorch MoE vs CUDA MoE ===
✓ SUCCESS: MoE implementations match within tolerance! Speedup: 1.32x
```

## ⚠️ Known Pain Points & Troubleshooting

### Training Setup Issues

1. **CUDA Kernel Compilation Failures**
   - **Symptom**: `nvcc fatal error` during `python setup.py build_ext --inplace`
   - **Common causes**: Incorrect CUDA toolkit version, missing GPU architecture flags
   - **Fix**: Ensure CUDA 12.1+ and add `-arch=sm_86` for RTX 30xx GPUs

2. **Memory Corruption in Custom Operations**
   - **Symptom**: Training works initially but crashes randomly or produces NaN gradients
   - **Cause**: Non-contiguous tensors passed to CUDA kernels
   - **Fix**: Always call `.contiguous()` before custom CUDA operations

3. **Gradient Flow Issues**
   - **Symptom**: `loss.backward()` fails or gradients are None
   - **Cause**: Incorrect backward pass implementation in custom CUDA operations
   - **Debug**: Replace custom ops with PyTorch equivalents incrementally to isolate

4. **Numerical Precision Differences**
   - **Symptom**: Identical loss curves expected but slight variations occur
   - **Normal**: Floating-point precision differences between PyTorch and custom CUDA
   - **Verification**: Check if differences are < 1e-4 (acceptable for float32)

### General Debugging Strategy

**Always establish a PyTorch baseline first:**
```python
# Replace custom CUDA operations with PyTorch equivalents temporarily
# self.custom_matmul = CustomMatMul()  # CUDA version
self.custom_matmul = torch.nn.functional.linear  # PyTorch equivalent
```

**Test incrementally:** Replace one operation at a time and verify training still works. This isolates exactly which custom operation is problematic.

**Common tensor issues:**
- Ensure tensors are contiguous: `assert x.is_contiguous()`
- Check tensor shapes: `print(f"Shape: {x.shape}, dtype: {x.dtype}")`
- Verify device placement: `assert x.device.type == 'cuda'`

### Inference-Specific Issues

1. **KV Cache Memory Issues**
   - **Symptom**: Memory usage grows unbounded during long generations
   - **Fix**: Implement proper cache size limits and periodic cleanup

2. **MoE Routing Instability**
   - **Symptom**: Different outputs on identical inputs due to floating-point precision
   - **Normal**: Expected with MoE routing - verify outputs are semantically similar

## 🏛️ Architecture Details

### Custom CUDA Operations

#### Training Operations (`custom_ops/`)
- **MatMul**: Matrix multiplication with batched support
- **Add/Mul**: Element-wise operations
- **LayerNorm**: Layer normalization
- **Softmax**: Attention softmax
- **Embedding**: Token/position embeddings
- **GELU**: Activation function

#### Inference Operations (`custom_ops_inf/`)
- **MatMul/GEMV**: Optimized for inference (single token generation)
- **TopK**: Expert selection for MoE routing
- **Element-wise**: Optimized for sequence processing
- **LayerNorm**: Fast inference implementation

### Hybrid Approach Philosophy

This codebase embraces **hybrid implementations** rather than pure CUDA:

✅ **Custom CUDA where reliable:**
- Matrix multiplications (well-established algorithms)
- Element-wise operations (simple, predictable)
- Standard normalization operations

⚠️ **PyTorch where complex:**
- Complex attention mechanisms (until fully debugged)
- Backward passes (gradient flow verification needed)
- Operations with dynamic shapes

**Why hybrid?** CUDA kernel debugging is extremely time-intensive. Using PyTorch for problematic operations allows you to accelerate the 80% of computation that works reliably while maintaining correctness.

## 🔬 Hyperparameters

### Training
- **Batch Size**: 16
- **Sequence Length**: 64
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Transformer Layers**: 4
- **Vocabulary Size**: ~80 (character-level)
- **Learning Rate**: 3e-4
- **Training Iterations**: 1000

### Inference
- **Batch Size**: 1 (autoregressive)
- **Sequence Length**: 64
- **Embedding Dimension**: 768 (10x larger than training)
- **Attention Heads**: 8
- **Transformer Layers**: 12
- **MoE Experts**: 8
- **Top-K Experts**: 2
- **Max New Tokens**: 200

## 🧪 Testing & Verification

### Numerical Accuracy Testing
```python
# Compare custom CUDA vs PyTorch outputs
torch_result = torch.nn.functional.layer_norm(x, (x.shape[-1],))
custom_result = custom_layer_norm(x)

max_diff = torch.abs(torch_result - custom_result).max()
assert max_diff < 1e-4, f"Accuracy test failed: {max_diff}"
```

### Performance Benchmarking
```python
# Profile CUDA kernels
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    # Your training/inference code here
    pass
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 📖 Advanced Documentation

### Training Debugging Guide
Located in `training/docs/debugging_guide.md` - comprehensive methodology for:
- Systematic CUDA kernel debugging
- Common bug patterns and fixes
- Testing strategies for numerical accuracy
- Performance profiling techniques

### Development Tips
- **Start simple**: Implement naive but correct CUDA kernels first
- **Test incrementally**: Add one custom operation at a time
- **Document assumptions**: Note tensor layout expectations and limitations
- **Use PyTorch as reference**: Always verify against known-good implementations

## 🎓 Educational Value

This repository serves as a **practical CUDA learning resource** with:

- **Real implementations**: Working custom CUDA kernels integrated with PyTorch
- **Debugging experience**: Comprehensive guide born from actual debugging sessions
- **Hybrid approach**: Practical balance between performance and maintainability
- **Progressive complexity**: From simple operations to complex transformer architectures

**Perfect for**: Students/researchers learning CUDA programming in the context of modern deep learning frameworks.

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Citation

If you use this code in your research or teaching, please consider citing:

```bibtex
@misc{cuda-transformer-naive,
  title={Character-level Transformer: Training \& Inference with Custom CUDA Kernels},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/naive.cu}
}
```

---

**Note**: This is a "hacky Karpathi-style repo" - educational code prioritizing clarity and learning over production optimization. Expect some rough edges and focus on understanding the concepts rather than perfect engineering practices.
