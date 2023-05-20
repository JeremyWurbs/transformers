from warnings import warn
import torch.nn as nn

from oak import MLP, ImageEmbedding, SelfAttentionBlock


class VisionTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_blocks, P, h, d_model, d_k=None, d_v=None, dropout=0., mlp_size=None):
        super().__init__()

        assert len(input_dim) == 3, f'Number of input_dims must be three, received {input_dim}'

        C, H, W = input_dim
        if C != 1 and C != 3:
            warn(f'Number of detected channels is {C}, which is unusual. Make sure the input and input_dims are both [C, H, W].')

        assert d_model % h == 0, 'd_model must be divisible by h'

        if d_v is None:
            d_v = d_model // h
        if d_k is None:
            d_k = d_v

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.P = P
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

        self.embedding = ImageEmbedding(C=C, H=H, W=W, P=P, d_model=d_model, bias=True, positional_encoding=True)
        self.blocks = nn.Sequential(*[SelfAttentionBlock(h=h, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout) for _ in range(num_blocks)])
        self.mlp = MLP(input_dim=d_model, output_dim=num_classes, hidden_dim=mlp_size, dropout=dropout)

    def forward(self, x):
        # x has dimensions (B, C, H, W)
        E = self.embedding(x)  # (B, L, d_k)
        Z = self.blocks(E)  # (B, L, d_v)
        logits = self.mlp(Z[:, 0, :].squeeze(dim=1))  # Only use the token class embedding (i.e. the first embedding)
        return logits

    @property
    def num_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
