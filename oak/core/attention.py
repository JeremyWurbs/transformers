import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention Module

    Takes an input of dimensions (B, L, d_model) and returns an embedding of dimension (B, L, d_v).

    Self-Attention:
    
    The attention mechanism is defined by the equation Z = softmax(Q @ K.transpose / sqrt(d_k)) @ V, where the
    matrices Q, K and V are computed from an input, X, according to X @ W_q, X @ W_k, and X @ W_v. The three weight
    matrices, W_q, W_k and W_v, are tunable parameters learned by the module during training. 
    
    Cross-Attention:
    
    If a second input, Y, is provided then cross-attention will be computed. Cross attention uses the same 
    equations but computes K and V using the second input, Y, where Y has dimensions (B, L_Y, d_model). I.e. Y may 
    have a different L, but must have the same batch size and embedding dimension (d_model) as X. In either case, 
    the output embedding will have the same dimension (B, L, d_v) as before, where L comes from the first input, X.
    
    Masked Attention:
    
    If the argument `mask` is set to True, masked-attention will be computed. Masked attention assumes there is an 
    ordering of the L elements of X, such that earlier elements should not be allowed access to view or use later 
    elements. E.g. if the elements of X are words in a sentence, and the network is supposed to learn to predict 
    later words, then those words should not be given during training. If masked attention is used, the output 
    will have the same dimensions, (B, L, d_model), but here the output[:, 0, :] values will only be computed 
    using X[:, 0, :], output[:, 1, :] will be computed only using X[:, 0:2, :] (note 0:2 here means 0 and 1), 
    output[:, 2, :] using X[:, 0:3, :], etc etc.
    
    Cross-attention is never used with a mask in either Vaswani et al. 2017 (the original transformer paper) or 
    Dosovitskiy et al. 2020 (the original visual transformer paper), and so it is not used in this repository
    either. Still, technically it could be done, assuming L_X=L_Y, in which case output[:, 0, :] would only be
    computed using X[:, 0, :] and Y[:, 0, :], etc etc. 

    Args:
        d_model: model size (a free parameter defined by the Transformer architecture)
        d_k: dimension of the queries and keys; intuitively the number of the queries and keys
        d_v: dimension of the values; intuitively the number of values
        mask: if set to True, masked attention will be computed

    Let
        B: batch size
        L: sequence length; equal to the number of words or patches (plus one for the class token, if used)

    Parameters
        The AttentionHead module trains the following parameters:
        W_q: Query weight matrix of dimension (d_model, d_k)
        W_k: Key weight matrix of dimension (d_model, d_k)
        W_v: Value weight matrix of dimension (d_model, d_v)
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

    def forward(self, X, Y=None, store_values=False):
        B, L, d_model = X.shape
        Y = X if Y is None else Y  # i.e. L = L_Y if Y is None
        Q = X @ self.W_Q  # (B, L, d_model) @ (d_model, d_k) -> (B, L, d_k)
        K = Y @ self.W_K  # (B, L, d_model) @ (d_model, d_k) -> (B, L_Y, d_k)
        V = Y @ self.W_V  # (B, L, d_model) @ (d_model, d_v) -> (B, L_Y, d_v)

        dot_product = Q @ K.transpose(-2, -1) * self.d_k ** -0.5
        if self.use_mask:
            mask = torch.tril(torch.ones(L, L)).to(self.W_Q.device)
            dot_product = dot_product.masked_fill(mask == 0, float('-inf'))
        A = F.softmax(dot_product, dim=-1)  # -> (B, L, L_Y)
        Z = A @ V  # (B, L, L_Y) @ (B, L_Y, d_model) -> (B, L, d_model)

        if store_values:
            self.Q = Q
            self.K = K
            self.V = V
            self.A = A

        return Z
