import torch
from torch.autograd import Function

class SoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # Save tensor for backward pass
        ctx.save_for_backward(x)

        # Create output tensor
        out = torch.empty_like(x)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.softmax_fwd(x, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors

        # Get the output from forward pass (this should be saved, but for simplicity we'll recompute)
        # In practice, you'd want to save the output tensor as well
        out = torch.empty_like(x)
        import custom_training_extension as cte
        cte.softmax_fwd(x, out)

        # Create gradient tensor
        grad_x = torch.empty_like(x)
        cte.softmax_bwd(grad_out, out, grad_x)

        return grad_x

class Softmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # For now, we assume the last dimension for softmax
        # In a full implementation, you'd handle different dimensions
        return SoftmaxFunction.apply(x)
