# Documentation

This directory contains detailed documentation for advanced usage and debugging of the CUDA kernel implementations.

## Available Guides

### [Debugging Guide](./debugging_guide.md)
Comprehensive methodology for debugging custom CUDA kernels when integrating them with PyTorch's autograd system. Includes:

- Systematic debugging approach for CUDA kernel issues
- Common bug patterns and their solutions
- Testing strategies and numerical verification
- Performance debugging techniques
- Best practices for CUDA kernel development

**Read this guide when:**
- Adding new CUDA operations to the codebase
- Debugging failing custom kernels
- Optimizing CUDA kernel performance
- Understanding CUDA memory management issues

## Usage

The debugging guide provides the methodology developed through extensive testing of this codebase. It includes specific examples from the transformer training implementation and general principles that apply to any CUDA kernel development project.
