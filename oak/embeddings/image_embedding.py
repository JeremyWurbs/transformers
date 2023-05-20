import torch
import torch.nn as nn


class ImageEmbedding(nn.Module):
    """Vision Transformer Patch Embedding Module

    Takes an input of dimensions (B, C, H, W) and returns an embedding of dimension:
        (B, N+1, d_model) if classification is True
        (B, N, d_model) if classification is False

    For vision transformers, it is expected that an HxW image is broken into PxP non-overlapping patches, which
    are equivalent to individual "words" in a standard transformer architecture. Each of these patches are then
    embedded into a space with dimension d_model, which is a free parameter defined by the transformer.

    Positional encoding:

    Following the original paper (Dosovitskiy et al. 2020), instead of using fixed position encodings from the
    original transformer architecture, learnable 1D positional encodings are used.

    Classification Token:

    Assuming the vision transformer is being used for classification, the embedding module prepends a class token to
    the patch embedding output, as described in the original paper (Dosovitskiy et al. 2020).

    The main idea for using a (randomly initialized) class token is that while it contains no useful information at
    the start, as this embedding token moves through the network it is able to pick up information from the other
    patches. Note that at the end of the transformer, the MLP classification head will use ONLY this embedding for
    its classification, meaning that the classification will be unbiased towards any image patch.

    Args
        C: number of channels in the original image
        H: height of the original image
        W: width of the original image
        P: patch size; i.e. each patch will have dimension [P x P]
        d_model: model size (a free parameter defined by the Transformer architecture)
        bias: whether to use bias weights for the embedding projection
        classification: whether the model is being used for classification
        positional_encoding: whether to use positional encodings

    Further, let
        B: batch size
        N: number of patches (i.e. H*W / P**2)
        L: sequence length; equal to the number of patches plus one for the class token, if used

    Parameters
        The embedding module trains the following parameters:
        W_emb & B_emb: The embedding weights and biases to transform each image patch from length P**2 to d_model
        pos_enc: 1D positional encodings added onto the embedded patch matrix
        cls_token: Input class token appended to the embedded patch matrix
    """

    def __init__(self, C, H, W, P, d_model, bias=True, classification=True, positional_encoding=True):
        super().__init__()
        assert H % P == 0, f'Image height, H ({H}), must be evenly divisible by the patch size, P ({P}).'
        assert W % P == 0, f'Image width, W ({W}), must be evenly divisible by the patch size, P ({P}).'
        self.C = C
        self.H = H
        self.W = W
        self.P = P
        self.d_model = d_model
        self.bias = bias
        self.classification = classification
        self.positional_encoding = positional_encoding

        self.W_emb = nn.Parameter(torch.empty(P**2, d_model))
        self.B_emb = nn.Parameter(torch.empty(self.N, 1)) if bias else None
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model)) if classification else None
        self.pos_enc = nn.Parameter(torch.empty(self.L, d_model)) if positional_encoding else None

        self.init_param()

    def init_param(self):
        nn.init.xavier_uniform_(self.W_emb)
        if self.B_emb is not None:
            nn.init.xavier_uniform_(self.B_emb)
        if self.cls_token is not None:
            nn.init.xavier_uniform_(self.cls_token)
        if self.pos_enc is not None:
            nn.init.xavier_uniform_(self.pos_enc)

    @property
    def N(self):
        """Number of patches"""
        return self.H * self.W // self.P ** 2

    @property
    def L(self):
        """Sequence length"""
        return self.N + 1 if self.classification else self.N

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
        elif len(x.shape) == 3:
            B = 1
        else:
            raise AssertionError('The input should have shape [B, C, H, W] or [C, H ,W]')

        patches = x.unfold(-2, self.P, self.P).unfold(-2, self.P, self.P)  # (B, C, H, W) -> (B, C, H/P, W/P, P, P)
        patches = patches.reshape(-1, self.N, self.P**2)  # -> (B, N, P**2)

        if self.bias:
            patches = patches @ self.W_emb + self.B_emb  # (B, N, P**2) @ (P**2, d_model) + (N, 1) -> (B, N, d_model)
        else:
            patches = patches @ self.W_emb  # (B, N, P**2) @ (P**2, d_model) -> (B, N, d_model)

        if self.classification:
            patches = torch.concat((self.cls_token.expand([B, -1, -1]), patches), dim=1)

        if self.positional_encoding:
            patches += self.pos_enc

        return patches