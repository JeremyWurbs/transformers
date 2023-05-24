import torch.nn as nn

from oak import MultiHeadAttention, MLP


class CrossAttentionBlock(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout=0., mask=False):
        super().__init__()

        self.mhsa = MultiHeadAttention(h=h, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, mask=mask)
        self.mhca = MultiHeadAttention(h=h, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, mask=False)
        self.mlp = MLP(input_dim=d_model, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, y):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.mhca(self.ln2(x), y)
        x = x + self.mlp(self.ln3(x))
        return x
