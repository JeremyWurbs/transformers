import torch
import torch.nn as nn

from oak import Attention, MaskedAttention


class MultiHeadSelfAttention(nn.Module):
    """Multi-head Self Attention Module

    Takes an input of dimensions (B, L, d_model) and returns an embedding of dimension (B, L, d_v).

    Multi-head self attention (MHSA) breaks the input embeddings into h pieces, passing each piece into h different
    attention head modules, before concatenating the output of each attention head back into a single output
    embedding. That is, each individual attention head still receives input from each patch, but only a reduced
    subspace (d_model/h) of the original d_model embedding. Each head subsequently trains their each independent
    weight matrices with reduced dimensions, W_Q: (d_model/h, d_k), W_K: (d_model/h, d_k) and W_V: (d_model/h, d_v).
    That is,

        Single Attention: (B, L, d_model) -> Attention -> (B, L, d_v)

        Multi-head Self Attention (h=2):
            X (B, L, d_model) -> {X_1, X_2} each of size (B, L, d_model/2)
            Z = concat(AttentionHead1(X_1) -> (B, L, d_v), AttentionHead2(X_2) -> (B, L, d_v)) -> (B, L, 2*d_v)

    Note that the output of the MHSA module is (B, L, h * d_v). In order to output the same dimensions as the input,
    it is common to set d_v = d_model/h, such that the result will be (B, L, d_model) -> MHSA -> (B, L, d_model),
    enabling the output of one MHSA block to directly feed into another with the same hyperparameters.

    Args
        h: number of attention heads
        d_model: model size (a free parameter defined by the Transformer architecture)
        d_k: dimension of the queries and keys; intuitively the number of the queries and keys
        d_v: dimension of the values; intuitively the number of values
        dropout: probability of each individual output being zeroed during training
    """

    def __init__(self, h, d_model, d_k, d_v, dropout=0., mask=False):
        super().__init__()

        assert d_model % h == 0, 'd_model must be divisible by h'

        self.h = h
        self.d_model = d_model
        self.d_head = d_model // h
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        if mask is True:
            self.heads = nn.ModuleList([MaskedAttention(d_model=self.d_head, d_k=d_k, d_v=d_v) for _ in range(h)])
        else:
            self.heads = nn.ModuleList([Attention(d_model=self.d_head, d_k=d_k, d_v=d_v) for _ in range(h)])
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, d_model = x.shape
        x = x.view(B, L, self.h, self.d_model // self.h)  # (B, L, d_model) -> (B, L, h, d_head)
        Z = torch.cat([self.heads[i](x[:, :, i, :].squeeze(2)) for i in range(self.h)], dim=-1)
        Z = self.dropout(self.linear(Z))

        return Z
