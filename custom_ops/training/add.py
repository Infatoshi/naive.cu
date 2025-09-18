import torch
from torch.autograd import Function

class AddFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Handle broadcasting by expanding tensors to same shape
        # This follows PyTorch's broadcasting rules
        a_broadcast, b_broadcast = torch.broadcast_tensors(a, b)

        # Save original tensors for backward pass
        ctx.save_for_backward(a, b)
        ctx.a_broadcast_shape = a_broadcast.shape
        ctx.b_broadcast_shape = b_broadcast.shape

        # Create output tensor with broadcasted shape
        out = torch.empty_like(a_broadcast)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.add_fwd(a_broadcast, b_broadcast, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        a_broadcast_shape = ctx.a_broadcast_shape
        b_broadcast_shape = ctx.b_broadcast_shape

        # Create gradient tensors with broadcasted shapes first
        grad_a_broadcast = torch.empty(a_broadcast_shape, dtype=a.dtype, device=a.device)
        grad_b_broadcast = torch.empty(b_broadcast_shape, dtype=b.dtype, device=b.device)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.add_bwd(grad_out, grad_a_broadcast, grad_b_broadcast)

        # Reduce gradients back to original shapes if broadcasting was used
        if a_broadcast_shape != a.shape:
            # Sum over the broadcasted dimensions to get back to original shape
            grad_a = grad_a_broadcast.sum_to_size(a.shape)
        else:
            grad_a = grad_a_broadcast

        if b_broadcast_shape != b.shape:
            # Sum over the broadcasted dimensions to get back to original shape
            grad_b = grad_b_broadcast.sum_to_size(b.shape)
        else:
            grad_b = grad_b_broadcast

        return grad_a, grad_b

class Add(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return AddFunction.apply(a, b)
