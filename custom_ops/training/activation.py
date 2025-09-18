import torch
from torch.autograd import Function

class GELUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # Save tensor for backward pass
        ctx.save_for_backward(x)

        # Create output tensor
        out = torch.empty_like(x)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.gelu_fwd(x, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors

        # Create gradient tensor
        grad_x = torch.empty_like(x)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.gelu_bwd(grad_out, x, grad_x)

        return grad_x

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GELUFunction.apply(x)
