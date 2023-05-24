import torch.nn as nn

from oak import MultiHeadAttention, MLP


class EncoderBlock(nn.Module):
    """Encoder Block

    This module is a single encoder block, as defined in Vaswani et al. 2017.
    """
    def __init__(self, h, d_model, d_k, d_v, dropout=0., mask=False):
        super().__init__()

        self.mha = MultiHeadAttention(h=h, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, mask=mask)
        self.mlp = MLP(input_dim=d_model, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
