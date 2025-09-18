import torch
from torch.autograd import Function

class EmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, weight, indices):
        # Save tensors for backward pass
        ctx.save_for_backward(weight, indices)

        # Create output tensor
        batch_size, seq_len = indices.shape
        n_embd = weight.shape[1]
        out = torch.empty(batch_size, seq_len, n_embd, dtype=weight.dtype, device=weight.device)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.embedding_fwd(weight, indices, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight, indices = ctx.saved_tensors

        # Create gradient tensor for weight
        grad_weight = torch.zeros_like(weight)

        indices_int32 = indices.to(torch.int32)

        # Import CUDA extension
        import custom_training_extension as cte
        cte.embedding_bwd(grad_out, indices_int32, grad_weight)

        return grad_weight, None

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize weight parameter
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, indices):
        # Convert indices to int32 for CUDA kernel compatibility
        indices_int32 = indices.to(torch.int32)
        return EmbeddingFunction.apply(self.weight, indices_int32)
