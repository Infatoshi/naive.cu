import torch
from torch.autograd import Function

class BatchedMatMulFunction(Function):
    @staticmethod
    def forward(ctx, A, B):
        # Save tensors for backward pass
        ctx.save_for_backward(A, B)

        # Create output tensor
        batch_size, M, K = A.shape
        _, _, N = B.shape
        C = torch.empty(batch_size, M, N, dtype=A.dtype, device=A.device)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.batched_matmul_fwd(A, B, C)

        return C

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors

        # Create gradient tensors
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(B)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.batched_matmul_bwd(A, B, grad_C, grad_A, grad_B)

        return grad_A, grad_B

class BatchedMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return BatchedMatMulFunction.apply(A, B)
