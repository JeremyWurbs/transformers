import torch
import torch.nn as nn


class TextEmbedding(nn.Module):
    """Text Embedding Module

    Takes a tokenized input of dimensions (B, L) and returns an embedding of dimension (B, L, d_model).

    For language transformers, it is expected that the input is a tokenized string of text, where each token is an
    integer from 0 to vocab_size-1. The TextEmbedding module will learn a mapping of these tokens to a higher
    dimensional space with dimension d_model, where d_model is the model size expected by the transformer.

    Positional encoding:

    Unlike Vaswani et al. (2017), which uses a fixed positional encoding scheme that is added onto the embeddings, we
    use a learnable weight embedding for the positional encodings.

    Args
        d_model: model size (a free parameter defined by the Transformer architecture)
        bias: whether to use bias weights for the embedding projection
        positional_encoding: whether to use positional encodings

        Further, let
            B: batch size
            L: sequence length; equal to the number of words in the input sequence

        Parameters
            The embedding module trains the following parameters:
            W_emb & B_emb: The embedding weights and biases to transform each image patch from length P**2 to d_model
            pos_enc: 1D positional encodings added onto the output embedded
    """

    def __init__(self, d_model, bias=True, seq_len=None, positional_encoding=True):
        if positional_encoding is True:
            assert seq_len is not None, 'Sequence length must be given to use positional encoding'

        super().__init__()
        self.d_model = d_model

        self.W_emb = nn.Parameter(torch.empty(1, d_model))
        self.B_emb = nn.Parameter(torch.empty(1, 1)) if bias else None
        self.pos_enc = nn.Parameter(torch.empty(seq_len, d_model)) if positional_encoding else None

        self.init_param()

    def init_param(self):
        nn.init.xavier_uniform_(self.W_emb)
        if self.B_emb is not None:
            nn.init.xavier_uniform_(self.B_emb)
        if self.pos_enc is not None:
            nn.init.xavier_uniform_(self.pos_enc)

    def forward(self, x):
        B, L = x.shape

        x = x.view(B, L, 1).type_as(self.W_emb) @ self.W_emb  # (B, L) -> (B, L, d_model)

        if self.B_emb is not None:
            x += self.B_emb

        if self.pos_enc is not None:
            x += torch.arange(L).type_as(self.pos_enc) @ self.pos_enc

        return x
