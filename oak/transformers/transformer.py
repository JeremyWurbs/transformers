import torch
import torch.nn as nn
import torch.nn.functional as F

from oak import TextEmbedding, EncoderBlock, DecoderBlock, MLP


class Transformer(nn.Module):
    def __init__(self, seq_len, num_blocks, h, d_model, vocab_size, BOS, EOS, PAD, src_vocab_size=None, d_k=None, d_v=None, dropout=0., mlp_size=None, **_):
        super().__init__()

        assert d_model % h == 0, 'd_model must be divisible by h'

        if d_v is None:
            d_v = d_model // h
        if d_k is None:
            d_k = d_model // h

        self.seq_len = seq_len
        self.tgt_vocab_size = vocab_size
        self.src_vocab_size = src_vocab_size if src_vocab_size is not None else vocab_size
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.BOS = BOS
        self.EOS = EOS
        self.PAD = PAD

        self.src_embedding = TextEmbedding(vocab_size=self.src_vocab_size, d_model=d_model, seq_len=seq_len)
        self.tgt_embedding = TextEmbedding(vocab_size=self.tgt_vocab_size, d_model=d_model, seq_len=seq_len, padding_idx=PAD)
        self.encoder = nn.Sequential(*[EncoderBlock(h=h, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout) for _ in range(num_blocks)])
        self.decoder = nn.ModuleList([DecoderBlock(h=h, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout) for _ in range(num_blocks)])
        self.mlp = MLP(input_dim=d_model, output_dim=self.tgt_vocab_size, hidden_dim=mlp_size, dropout=dropout)

    def forward(self, input):
        x, y = input
        if len(x.shape) == 1:
            x = x.view(1, -1)
        if len(y.shape) == 1:
            y = y.view(1, -1)
        B, L = x.shape

        src_emb = self.src_embedding(y)
        src_context = self.encoder(src_emb)

        x = self.tgt_embedding(x)
        for decoder_block in self.decoder:
            x = decoder_block(x, src_context)

        logits = self.mlp(x.view(B * L, self.d_model))

        return logits

    def translate(self, src, max_new_tokens=2048):
        with torch.no_grad():
            self.eval()
            src = src.to(self.src_embedding.token_enc.weight.device)
            B, L_Y = src.shape
            output_emb = self.BOS * torch.ones((B, 1), dtype=torch.int).to(self.src_embedding.token_enc.weight.device)

            while True:
                B, L = output_emb.shape
                x = output_emb[:, -self.seq_len:]
                logits = self.forward((x, src)).view(B, -1, self.tgt_vocab_size)[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_pred = torch.multinomial(probs, num_samples=1)

                if B == 1 and next_pred == self.EOS:
                    break
                else:
                    output_emb = torch.cat((output_emb, next_pred), dim=1)

                if output_emb.shape[1] >= max_new_tokens:
                    break

            return output_emb[:, 1:]

    @property
    def num_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
