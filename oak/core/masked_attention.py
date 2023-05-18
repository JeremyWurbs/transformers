import torch
import torch.nn.functional as F

from oak import Attention


class MaskedAttention(Attention):
    def __init__(self, d_model, d_k, d_v):
        super().__init__(d_model=d_model, d_k=d_k, d_v=d_v)
        self.register_buffer('mask', torch.Tensor(torch.empty(0)))

    def forward(self, X, store_values=False):
        # X should have dimension (B, L, d_model)
        B, L, d_model = X.shape

        if self.mask.numel() < 1:
            mask = torch.tril(torch.ones(L, L))
            self.mask = mask.masked_fill(mask == 0, float('-inf')).unsqueeze(dim=0).repeat(B, 1, 1).type_as(self.mask)

        Q = X @ self.W_Q  # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
        K = X @ self.W_K  # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
        V = X @ self.W_V  # (B, L, d_model) @ (d_model, d_v) -> (B, L, d_v)

        dot_product = (Q @ K.transpose(-2, -1) * self.d_k ** -0.5).masked_fill(self.mask == 0, float('-inf'))
        A = F.softmax(dot_product, dim=-1)  # -> (B, L, L)
        Z = A @ V  # (B, L, L) @ (B, L, d_model) -> (B, L, d_model)

        if store_values:
            self.Q = Q
            self.K = K
            self.V = V
            self.A = A

        return Z

