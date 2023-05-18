import torch
import torch.nn as nn

from oak import ShakespeareTokenizer


class TextEmbedding(nn.Module):
    """Text Embedding Module

    Takes a tokenized input of dimensions (B, L) or (B, L, d_emb) and returns an embedding of dimension (B, L, d_model).

    For language transformers, it is expected that the input is a tokenized string of text, where each token is either
    an integer or a (float) vector of dimension d_emb. As different token embeddings may output different size tokens,
    the TextEmbedding module will learn a mapping of these tokens from size d_emb to d_model, where d_model is the model
    size expected by the transformer.

    Positional encoding:

    Following Vaswani et al. (2017), we use a fixed positional encoding scheme that is added onto the embeddings.

    """

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

    def __init__(self, d_model, d_emb=1, bias=True, seq_len=None, positional_encoding=True):
        if positional_encoding is True:
            assert seq_len is not None, 'Sequence length must be given to use positional encoding'

        super().__init__()
        self.d_model = d_model

        self.W_emb = nn.Parameter(torch.empty(d_emb, d_model))
        self.B_emb = nn.Parameter(torch.empty(d_emb, 1)) if bias else None
        self.pos_enc = nn.Parameter(torch.empty(seq_len, d_model)) if positional_encoding else None

        self.init_param()

        # TODO: add positional encoding

    def init_param(self):
        nn.init.xavier_uniform_(self.W_emb)
        if self.B_emb is not None:
            nn.init.xavier_uniform_(self.B_emb)
        if self.pos_enc is not None:
            nn.init.xavier_uniform_(self.pos_enc)

    def forward(self, x):
        # x has shape (B, L) or (B, L, d_emb), where each row is a tokenized sample
        if len(x.shape) == 3:
            B, L, d_emb = x.shape
        else:
            B, L = x.shape
            d_emb = 1
        x = x.view((B, L, d_emb)).type(self.W_emb.dtype) @ self.W_emb  # -> (B, L, d_model)
        if self.B_emb is not None:
            x += self.B_emb

        # TODO: add positional encoding
        return x
