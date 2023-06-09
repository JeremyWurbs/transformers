import torch
import torch.nn as nn

from oak import Attention


class MultiHeadAttention(nn.Module):
    """Multi-head Attention Module

    Takes an input of dimensions (B, L, d_model) and returns an embedding of dimension (B, L, d_model).

    Multi-head attention (MHA) breaks the input embeddings into h pieces, passing each piece into h different attention
    head modules, before concatenating the output of each attention head back into a single output embedding. That is,
    each individual attention head still receives input from each word (or patch), but only a reduced subspace
    (d_model/h) of the original d_model embedding. Each head subsequently trains their own, independent weight matrices
    with reduced dimensions, W_Q: (d_model/h, d_k), W_K: (d_model/h, d_k) and W_V: (d_model/h, d_v). That is,

        Single Attention: (B, L, d_model) -> Attention -> (B, L, d_v)

        Multi-head (Self) Attention (h=2):
            X (B, L, d_model) -> {X_1, X_2} each of size (B, L, d_model/2)
            Z = concat(Attention1(X_1) -> (B, L, d_v), Attention2(X_2) -> (B, L, d_v)) -> (B, L, 2*d_v)

    Note that the last dimension of the MHA output is then h * d_v, and NOT just d_v, as in simple attention. In order
    to output the same dimensions as the input, as well as maintain the dimensionality of the model throughout, it is 
    common to set d_v = d_model/h, such that the result will be (B, L, d_model) -> MHA -> (B, L, d_model), enabling 
    the output of one MHA block to directly feed into another with the same d_model hyperparameter. 
    
    In either case, following Vaswani et al. 2017, we actually add a linear layer after the attention head 
    computations, which will convert (B, L, h*d_v) to (B, L, d_model), allowing you to use different, non-matching 
    d_model, h and d_v hyperparameters if you wish.

    Cross-Attention:

    If a second input, Y, is given then cross-attention will be computed. Refer to `attention.py` for details.

    Masked Attention:

    If the argument `mask` is set to True, masked attention will be computed. Refer to `attention.py` for details.

    Args
        h: number of attention heads
        d_model: model size (a free parameter defined by the Transformer architecture)
        d_k: dimension of the queries and keys; intuitively the number of the queries and keys
        d_v: dimension of the values; intuitively the number of values
        dropout: probability of each individual output being zeroed during training
        mask: if set to True, masked attention will be computed

    Let
        B: batch size
        L: sequence length; equal to the number of words or patches (plus one for the class token, if used)
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

        self.heads = nn.ModuleList([Attention(d_model=self.d_model, d_k=d_k, d_v=d_v, mask=mask) for _ in range(h)])
        self.linear = nn.Linear(h*d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y=None):
        if Y is None:  # Self-Attention
            Z = torch.cat([self.heads[i](X) for i in range(self.h)], dim=-1)  # -> (B, L, h*d_v)
        else:  # Cross-Attention
            Z = torch.cat([self.heads[i](X, Y) for i in range(self.h)], dim=-1)  # -> (B, L, h*d_v)

        Z = self.dropout(self.linear(Z))  # (B, L, h*d_v) -> (B, L, d_model)
        return Z
