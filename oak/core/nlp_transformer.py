import torch
import torch.nn as nn

from oak import TextEmbedding, SelfAttentionBlock, MLP


class NLPTransformer(nn.Module):
    def __init__(self, seq_len, num_blocks, vocab_size, h, d_emb, d_model, batch_size, d_k=None, d_v=None, dropout=0., mlp_size=None):
        super().__init__()

        assert d_model % h == 0, 'd_model must be divisible by h'

        if d_v is None:
            d_v = d_model // h
        if d_k is None:
            d_k = d_model // h

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.h = h
        self.d_emb = d_emb
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        self.embedding = TextEmbedding(d_model=d_model, seq_len=seq_len)
        self.blocks = nn.Sequential(*[SelfAttentionBlock(h=h, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, mask=True) for _ in range(num_blocks)])
        self.mlp = MLP(input_dim=seq_len * d_model, output_dim=vocab_size, hidden_dim=mlp_size, dropout=dropout)

        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.masked_fill(mask == 0, float('-inf')).repeat(batch_size, 1).unsqueeze(dim=2).repeat(1, 1, self.d_model)
        self.register_buffer('mask', mask)

    def forward(self, x):
        # x has dimensions (B, L, d_emb)
        assert len(x.shape) == 3, 'Input must have dimensions (B, L, d_emb)'
        B, *_ = x.shape

        E = self.embedding(x).repeat_interleave(self.seq_len, dim=0)
        Z = self.blocks(E)
        logits = self.mlp(Z.view(B * self.seq_len, -1))

        # TODO: add final softmax?

        return logits
