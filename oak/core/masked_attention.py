import torch
import torch.nn.functional as F

from oak import Attention


class MaskedAttention(Attention):
    def __init__(self, d_model, d_k, d_v):
        super().__init__(d_model=d_model, d_k=d_k, d_v=d_v)

    def forward(self, X, store_values=False):
        # X should have dimension (B, L, d_model)
        B, L, d_model = X.shape

        Q = X @ self.W_Q  # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
        K = X @ self.W_K  # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
        V = X @ self.W_V  # (B, L, d_model) @ (d_model, d_v) -> (B, L, d_v)

        mask = torch.tril(torch.ones(L, L)).to(self.W_Q.device)
        dot_product = (Q @ K.transpose(-2, -1) * self.d_k ** -0.5).masked_fill(mask == 0, float('-inf'))
        A = F.softmax(dot_product, dim=-1)  # -> (B, L, L)
        Z = A @ V  # (B, L, L) @ (B, L, d_model) -> (B, L, d_model)

        if store_values:
            self.Q = Q
            self.K = K
            self.V = V
            self.A = A

        return Z
