import torch
from torch.autograd import Function

class MatMulFunction(Function):
    @staticmethod
    def forward(ctx, A, B):
        # Save tensors for backward pass
        ctx.save_for_backward(A, B)

        # Create output tensor C = A @ B
        M, K = A.shape
        N = B.shape[1]
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.matmul_fwd(A, B, C)

        return C

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors

        # Create gradient tensors
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(B)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.matmul_bwd(A, B, grad_C, grad_A, grad_B)

        return grad_A, grad_B

class MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return MatMulFunction.apply(A, B)
