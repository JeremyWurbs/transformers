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
        vocab_size: number of words in the tokenizer dictionary
        d_model: model size (a free parameter defined by the Transformer architecture)
        positional_encoding: whether to use positional encodings
        seq_len: sequence length, L; required only if positional_encoding is set to True
    """

    def __init__(self, vocab_size, d_model, seq_len=None, positional_encoding=True):
        if positional_encoding is True:
            assert seq_len is not None, 'Sequence length must be given to use positional encoding'

        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        self.token_enc = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Embedding(seq_len, d_model) if positional_encoding else None

    def forward(self, tokens):
        B, L = tokens.shape
        embedding = self.token_enc(tokens)
        if self.pos_enc is not None:
            pos_emb = self.pos_enc(torch.arange(L, device=tokens.device))
            embedding += pos_emb
        return embedding
