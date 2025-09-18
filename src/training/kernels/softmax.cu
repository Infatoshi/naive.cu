#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Softmax forward pass
__global__ void softmax_fwd_kernel(const float* x, float* out, int batch_size, int seq_len, int n_embd) {
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int tid = threadIdx.x;

    if (batch < batch_size && seq < seq_len) {
        // Shared memory for max and sum reduction
        extern __shared__ float shared_mem[];
        float* max_val = shared_mem;
        float* sum_val = &shared_mem[blockDim.x];

        // First pass: find max for numerical stability
        float local_max = -INFINITY;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            local_max = fmaxf(local_max, x[idx]);
        }
        max_val[tid] = local_max;

        // Reduce max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                max_val[tid] = fmaxf(max_val[tid], max_val[tid + stride]);
            }
        }
        __syncthreads();
        float global_max = max_val[0];

        // Second pass: compute exp(x - max) and sum
        float local_sum = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            float exp_val = expf(x[idx] - global_max);
            out[idx] = exp_val;
            local_sum += exp_val;
        }
        sum_val[tid] = local_sum;

        // Reduce sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                sum_val[tid] += sum_val[tid + stride];
            }
        }
        __syncthreads();
        float global_sum = sum_val[0];

        // Third pass: normalize
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            out[idx] /= global_sum;
        }
    }
}

// Softmax backward pass
__global__ void softmax_bwd_kernel(const float* grad_out, const float* out,
                                 float* grad_x, int batch_size, int seq_len, int n_embd) {
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int tid = threadIdx.x;

    if (batch < batch_size && seq < seq_len) {
        // Shared memory for dot product reduction
        extern __shared__ float shared_dot[];

        // Compute sum of grad_out * out for each position
        float local_dot = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            local_dot += grad_out[idx] * out[idx];
        }
        shared_dot[tid] = local_dot;

        // Reduce dot product
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
        }
        __syncthreads();
        float sum_grad_out_softmax = shared_dot[0];

        // Compute gradient
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            grad_x[idx] = out[idx] * (grad_out[idx] - sum_grad_out_softmax);
        }
    }
}

// Host functions to launch kernels
void softmax_fwd_cuda(const float* x, float* out, int batch_size, int seq_len, int n_embd) {
    dim3 blocks(batch_size, seq_len);
    int threads = min(256, n_embd);
    size_t shared_mem = 2 * threads * sizeof(float);
    softmax_fwd_kernel<<<blocks, threads, shared_mem>>>(x, out, batch_size, seq_len, n_embd);
}

void softmax_bwd_cuda(const float* grad_out, const float* out, float* grad_x,
                     int batch_size, int seq_len, int n_embd) {
    dim3 blocks(batch_size, seq_len);
    int threads = min(256, n_embd);
    size_t shared_mem = threads * sizeof(float);
    softmax_bwd_kernel<<<blocks, threads, shared_mem>>>(grad_out, out, grad_x, batch_size, seq_len, n_embd);
}
