#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

// GELU Forward: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_fwd_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Use erfc for numerical stability: erfc(-x/sqrt(2)) = 1 + erf(x/sqrt(2))
        float arg = val * 0.7071067811865476f;  // 1/sqrt(2)
        float erf_val = 1.0f - erfc(arg);
        out[idx] = 0.5f * val * (1.0f + erf_val);
    }
}

// GELU Backward: d/dx gelu(x) = 0.5 * (1 + erf(x/sqrt(2))) + 0.5 * x * d/dx(1 + erf(x/sqrt(2)))
__global__ void gelu_bwd_kernel(const float* grad_out, const float* x, float* grad_x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float arg = val * 0.7071067811865476f;  // 1/sqrt(2)
        float erf_val = 1.0f - erfc(arg);

        // Derivative: 0.5 * (1 + erf) + 0.5 * x * d/dx(1 + erf)
        // d/dx erf(x/sqrt(2)) = (2/sqrt(pi)) * exp(-(x/sqrt(2))^2) * (1/sqrt(2))
        float d_erf = (2.0f / sqrtf(M_PI)) * expf(-arg * arg) * (1.0f / sqrtf(2.0f));

        grad_x[idx] = grad_out[idx] * (0.5f * (1.0f + erf_val) + 0.5f * val * d_erf);
    }
}

// Host functions to launch kernels
void gelu_fwd_cuda(const float* x, float* out, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_fwd_kernel<<<blocks, threads>>>(x, out, size);
}

void gelu_bwd_cuda(const float* grad_out, const float* x, float* grad_x, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_bwd_kernel<<<blocks, threads>>>(grad_out, x, grad_x, size);
}
