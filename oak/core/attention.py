import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention Module

    Takes an input of dimensions (B, L, d_model) and returns an embedding of dimension (B, L, d_v).

    The attention mechanism is defined by the equation Z = softmax(Q @ K.transpose / sqrt(d_k)) @ V, where the
    matrices Q, K and V are computed from an input, X, according to X @ W_q, X @ W_k, and X @ W_v. The three weight
    matrices, W_q, W_k and W_v, are tunable parameters learned by the module during training.

    Args:
        d_model: model size (a free parameter defined by the Transformer architecture)
        d_k: dimension of the queries and keys; intuitively the number of the queries and keys
        d_v: dimension of the values; intuitively the number of values

    Let
        B: batch size
        L: sequence length; equal to the number of patches plus one for the class token, if used

    Parameters
        The AttentionHead module trains the following parameters:
        W_q: Query weight matrix of dimension (L x d_k)
        W_k: Key weight matrix of dimension (L x d_k)
        W_v: Value weight matrix of dimension (L x d_v)
    """

    def __init__(self, d_model, d_k, d_v, mask=False):
        super().__init__()
        self.use_mask = mask
        self.W_Q = nn.Parameter(torch.empty(d_model, d_k))
        self.W_K = nn.Parameter(torch.empty(d_model, d_k))
        self.W_V = nn.Parameter(torch.empty(d_model, d_v))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

    def forward(self, X, store_values=False):
        B, L, d_model = X.shape
        Q = X @ self.W_Q  # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
        K = X @ self.W_K  # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
        V = X @ self.W_V  # (B, L, d_model) @ (d_model, d_v) -> (B, L, d_v)

        dot_product = Q @ K.transpose(-2, -1) * self.d_k ** -0.5
        if self.use_mask:
            mask = torch.tril(torch.ones(L, L)).to(self.W_Q.device)
            dot_product = dot_product.masked_fill(mask == 0, float('-inf'))
        A = F.softmax(dot_product, dim=-1)  # -> (B, L, L)
        Z = A @ V  # (B, L, L) @ (B, L, d_model) -> (B, L, d_model)

        if store_values:
            self.Q = Q
            self.K = K
            self.V = V
            self.A = A

        return Z
