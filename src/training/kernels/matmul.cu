#include <cuda_runtime.h>
#include <torch/extension.h>

// Naive matrix multiplication forward pass
// C = A @ B where A is (M, K), B is (K, N), C is (M, N)
__global__ void matmul_fwd_kernel(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiplication backward pass
// grad_A = grad_C @ B.T
__global__ void matmul_bwd_A_kernel(const float* grad_C, const float* B, float* grad_A,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            sum += grad_C[row * N + n] * B[col * N + n];
        }
        grad_A[row * K + col] = sum;
    }
}

// grad_B = A.T @ grad_C
__global__ void matmul_bwd_B_kernel(const float* A, const float* grad_C, float* grad_B,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < K && col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) {
            sum += A[m * K + row] * grad_C[m * N + col];
        }
        grad_B[row * N + col] = sum;
    }
}

// Batched matrix multiplication forward pass
// C[b][i][j] = A[b][i][k] * B[b][k][j] where b is batch, i,j,k are matrix dims
__global__ void batched_matmul_fwd_kernel(const float* A, const float* B, float* C,
                                         int batch_size, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            int a_idx = batch * M * K + row * K + k;
            int b_idx = batch * K * N + k * N + col;
            sum += A[a_idx] * B[b_idx];
        }
        int c_idx = batch * M * N + row * N + col;
        C[c_idx] = sum;
    }
}

// Batched matrix multiplication backward pass for A
// grad_A[b][i][k] = sum_j grad_C[b][i][j] * B[b][k][j]
// (since we need grad_C @ B^T, and B^T[b][j][k] = B[b][k][j])
__global__ void batched_matmul_bwd_A_kernel(const float* grad_C, const float* B, float* grad_A,
                                           int batch_size, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < M && col < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            int grad_c_idx = batch * M * N + row * N + n;
            int b_idx = batch * K * N + col * N + n;  // B[b][col][n]
            sum += grad_C[grad_c_idx] * B[b_idx];
        }
        int grad_a_idx = batch * M * K + row * K + col;
        grad_A[grad_a_idx] = sum;
    }
}

// Batched matrix multiplication backward pass for B
// grad_B[b][k][j] = sum_i A[b][i][k] * grad_C[b][i][j]
// (since we need A^T @ grad_C, and A^T[b][k][i] = A[b][i][k])
__global__ void batched_matmul_bwd_B_kernel(const float* A, const float* grad_C, float* grad_B,
                                           int batch_size, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < K && col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) {
            int a_idx = batch * M * K + m * K + row;  // A[b][m][row]
            int grad_c_idx = batch * M * N + m * N + col;  // grad_C[b][m][col]
            sum += A[a_idx] * grad_C[grad_c_idx];
        }
        int grad_b_idx = batch * K * N + row * N + col;  // grad_B[b][row][col]
        grad_B[grad_b_idx] = sum;
    }
}

// Host functions to launch kernels
void matmul_fwd_cuda(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_fwd_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}

void matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                    float* grad_A, float* grad_B, int M, int N, int K) {
    // Compute grad_A = grad_C @ B.T
    dim3 threads_A(16, 16);
    dim3 blocks_A((K + threads_A.x - 1) / threads_A.x, (M + threads_A.y - 1) / threads_A.y);
    matmul_bwd_A_kernel<<<blocks_A, threads_A>>>(grad_C, B, grad_A, M, N, K);

    // Compute grad_B = A.T @ grad_C
    dim3 threads_B(16, 16);
    dim3 blocks_B((N + threads_B.x - 1) / threads_B.x, (K + threads_B.y - 1) / threads_B.y);
    matmul_bwd_B_kernel<<<blocks_B, threads_B>>>(A, grad_C, grad_B, M, N, K);
}

// Host functions for batched operations
void batched_matmul_fwd_cuda(const float* A, const float* B, float* C,
                            int batch_size, int M, int N, int K) {
    dim3 threads(8, 8);  // Reduced from 16x16 to 8x8 to reduce register pressure
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                batch_size);
    batched_matmul_fwd_kernel<<<blocks, threads>>>(A, B, C, batch_size, M, N, K);
}

void batched_matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                            float* grad_A, float* grad_B,
                            int batch_size, int M, int N, int K) {
    // Compute grad_A = grad_C @ B^T (batched)
    dim3 threads_A(8, 8);  // Reduced from 16x16
    dim3 blocks_A((K + threads_A.x - 1) / threads_A.x,
                  (M + threads_A.y - 1) / threads_A.y,
                  batch_size);
    batched_matmul_bwd_A_kernel<<<blocks_A, threads_A>>>(grad_C, B, grad_A, batch_size, M, N, K);

    // Compute grad_B = A^T @ grad_C (batched)
    dim3 threads_B(8, 8);  // Reduced from 16x16
    dim3 blocks_B((N + threads_B.x - 1) / threads_B.x,
                  (K + threads_B.y - 1) / threads_B.y,
                  batch_size);
    batched_matmul_bwd_B_kernel<<<blocks_B, threads_B>>>(A, grad_C, grad_B, batch_size, M, N, K);
}
