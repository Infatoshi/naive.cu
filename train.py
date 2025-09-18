import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import requests
import os
import time
import math

# Hyperparameters (matching requirements)
batch_size = 16
block_size = 64  # sequence length
n_embd = 128
n_head = 4
n_layer = 4
vocab_size = 65
learning_rate = 3e-4
max_iters = 1000
device = 'cuda'

print(f"Using device: {device}")
print(f"Batch size: {batch_size}, Block size: {block_size}, Embedding dim: {n_embd}")

# Data loading functions (inspired by nanogpt-train.py)
def download_file(url, filename):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(response.text)
            print(f"File downloaded and saved as {filename}")
        else:
            print("Failed to download the file")
    except Exception as e:
        print(f"An error occurred: {e}")

dataset_url = "https://www.gutenberg.org/files/55/55-0.txt"
dataset_filename = "wizard_of_oz.txt"

if not os.path.exists(dataset_filename):
    download_file(dataset_url, dataset_filename)
else:
    print(f"{dataset_filename} already exists, skipping download.")
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ================================
# Stage 1: PyTorch Baseline Model
# ================================

class PytorchHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class PytorchMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([PytorchHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class PytorchFeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

class PytorchBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = PytorchMultiHeadAttention(n_head, head_size)
        self.ffwd = PytorchFeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class PytorchTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[PytorchBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# ================================
# Stage 2: Custom CUDA Model
# ================================

from custom_ops.training import MatMul, BatchedMatMul, Add, Mul, GELU, Softmax, LayerNorm, Embedding

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Note: We use (out_features, in_features) shape and transpose during forward
        # to match how nn.Linear stores weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.matmul = MatMul()
        self.add = Add()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (batch, in_features) or (batch, seq, in_features)
        # weight: (out_features, in_features)
        # We need to compute x @ weight.T

        if x.dim() == 2:
            # For 2D input: (batch, in_features)
            # Ensure tensors are contiguous for CUDA kernel
            x_contiguous = x.contiguous()
            weight_t_contiguous = self.weight.t()
            out = self.matmul(x_contiguous, weight_t_contiguous)

            if self.bias is not None:
                out = self.add(out, self.bias.unsqueeze(0))
        else:
            # For 3D input: (batch, seq, in_features)
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.contiguous().view(-1, x.size(-1))  # Ensure contiguous before reshape

            # Ensure tensors are contiguous for CUDA kernel
            x_reshaped_contiguous = x_reshaped.contiguous()
            weight_t_contiguous = self.weight.t()

            out = self.matmul(x_reshaped_contiguous, weight_t_contiguous)

            if self.bias is not None:
                bias_expanded = self.bias.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                out = self.add(out.view(batch_size, seq_len, -1), bias_expanded).contiguous()
            else:
                out = out.view(batch_size, seq_len, -1).contiguous()

        return out

class CustomHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = CustomLinear(n_embd, head_size, bias=False)
        self.query = CustomLinear(n_embd, head_size, bias=False)
        self.value = CustomLinear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.matmul = MatMul()
        self.batched_matmul = BatchedMatMul()
        self.mul = Mul()
        self.add = Add()
        self.softmax = Softmax()

    def forward(self, x):
        B, T, C = x.shape

        # Ensure input is contiguous
        assert x.is_contiguous(), "Input tensor x must be contiguous"

        # Compute key, query, value projections
        k = self.key(x).contiguous()  # (B, T, head_size)
        q = self.query(x).contiguous()  # (B, T, head_size)
        v = self.value(x).contiguous()  # (B, T, head_size)

        # Attention computation
        k_t = k.transpose(-2, -1).contiguous()  # (B, head_size, T)
        wei = self.batched_matmul(q.contiguous(), k_t)  # (B, T, T)

        # Scale attention weights
        wei = wei * (C**-0.5)

        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.softmax(wei)

        # Apply attention to values
        out = self.batched_matmul(wei, v.contiguous())  # (B, T, head_size)

        return out

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CustomHead(head_size) for _ in range(num_heads)])
        # Keep PyTorch Linear for projection to avoid issues
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        # Concatenate attention outputs
        head_outputs = []
        for h in self.heads:
            head_outputs.append(h(x))
        out = torch.cat(head_outputs, dim=-1)

        # Project back to n_embd
        out = self.proj(out)
        return out

class CustomFeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Keep PyTorch Linear for feed-forward to avoid issues
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        # self.gelu = GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x

class CustomBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = CustomMultiHeadAttention(n_head, head_size)
        self.ffwd = CustomFeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        self.add = Add()

    def forward(self, x):
        # Self-attention with residual (Pre-LayerNorm like in GPT-2)
        x = self.ln1(x)
        y = self.sa(x)
        # Temporarily use PyTorch addition
        x = x + y

        # Feed-forward with residual
        x = self.ln2(x)
        y = self.ffwd(x)
        # Temporarily use PyTorch addition
        x = x + y

        return x

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Use custom components for pure CUDA implementation
        self.token_embedding = Embedding(vocab_size, n_embd)
        self.position_embedding = Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([CustomBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = CustomLinear(n_embd, vocab_size, bias=False)
        self.add = Add()

        # Initialize weights randomly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, CustomLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        # Ensure input is contiguous
        assert index.is_contiguous(), "Input tensor index must be contiguous"

        # Embedding
        tok_emb = self.token_embedding(index)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=device).unsqueeze(0))  # (1, T, n_embd)
        # Temporarily use PyTorch addition
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x.contiguous())

        # Final layer norm
        x = self.ln_f(x.contiguous())

        # Language model head
        logits = self.lm_head(x.contiguous())  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


# ================================
# Main Execution
# ================================

if __name__ == "__main__":
    # ================================
    # Stage 1: PyTorch Baseline Training
    # ================================
    print("\nPyTorch baseline training...")

    # Initialize PyTorch model
    pytorch_model = PytorchTransformer(vocab_size).to(device)
    pytorch_optimizer = torch.optim.AdamW(pytorch_model.parameters(), lr=learning_rate)

    print(f"PyTorch Model Parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")

    # Training loop
    pytorch_start_time = time.time()

    for iter in range(max_iters):
        xb, yb = get_batch('train')

        logits, loss = pytorch_model(xb, yb)

        if iter % 100 == 0:
            print(f"iter {iter}/{max_iters} | loss {loss.item():.4f}")

        pytorch_optimizer.zero_grad()
        loss.backward()
        pytorch_optimizer.step()

    pytorch_end_time = time.time()
    pytorch_training_time = pytorch_end_time - pytorch_start_time

    print(f"PyTorch training time: {pytorch_training_time:.2f} seconds")

    # ================================
    # Stage 2: Custom CUDA Implementation
    # ================================
    print("\nCustom CUDA training...")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize both models with same seed
    pytorch_model_verify = PytorchTransformer(vocab_size).to(device)
    custom_model = CustomTransformer(vocab_size).to(device)

    # Initialize optimizer for custom model
    custom_optimizer = torch.optim.AdamW(custom_model.parameters(), lr=learning_rate)

    print(f"Custom Model Parameters: {sum(p.numel() for p in custom_model.parameters()):,}")

    custom_start_time = time.time()

    for iter in range(max_iters):
        xb, yb = get_batch('train')

        logits, loss = custom_model(xb, yb)

        if iter % 100 == 0:
            print(f"iter {iter}/{max_iters} | loss {loss.item():.4f}")

        custom_optimizer.zero_grad()
        loss.backward()
        custom_optimizer.step()

    custom_end_time = time.time()
    custom_training_time = custom_end_time - custom_start_time

    print(f"Custom CUDA training time: {custom_training_time:.2f} seconds")
