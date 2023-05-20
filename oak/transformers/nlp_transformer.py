import torch
import torch.nn as nn
import torch.nn.functional as F

from oak import TextEmbedding, SelfAttentionBlock, MLP


class NLPTransformer(nn.Module):
    def __init__(self, seq_len, num_blocks, vocab_size, h, d_model, d_k=None, d_v=None, dropout=0., mlp_size=None):
        super().__init__()

        assert d_model % h == 0, 'd_model must be divisible by h'

        if d_v is None:
            d_v = d_model // h
        if d_k is None:
            d_k = d_model // h

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        self.embedding = TextEmbedding(d_model=d_model, seq_len=seq_len)
        self.blocks = nn.Sequential(*[SelfAttentionBlock(h=h, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout, mask=True) for _ in range(num_blocks)])
        self.mlp = MLP(input_dim=d_model, output_dim=vocab_size, hidden_dim=mlp_size, dropout=dropout)

    def forward(self, x):
        B, L = x.shape

        E = self.embedding(x)
        Z = self.blocks(E)
        logits = self.mlp(Z.view(B*self.seq_len, self.d_model))

        return logits

    def generate(self, context, max_new_tokens):
        context = context.to(self.embedding.W_emb.device)

        B, L = context.shape
        for _ in range(max_new_tokens):
            x = context[:, -self.seq_len:]  # crop the input context to fit within our sequence length
            logits = self.forward(x).view(B, L, self.vocab_size)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_pred = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_pred), dim=1)

        return context

    @property
    def num_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
