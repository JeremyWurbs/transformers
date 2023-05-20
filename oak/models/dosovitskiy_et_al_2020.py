"""Dosovitskiy, Alexey, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale."""

from oak import VisionTransformer, LightningModel


class ViTBase(LightningModel):
    """ViT Base as defined in Dosovitskiy et al. 2020"""
    def __init__(self, input_dim, num_classes, P):
        super().__init__(base_model=VisionTransformer(input_dim, num_classes,
                                                      P=P, h=12, num_blocks=12, d_model=768, dropout=0., mlp_size=3072),
                         num_classes=num_classes)


class ViTLarge(LightningModel):
    """ViT Large as defined in Dosovitskiy et al. 2020"""
    def __init__(self, input_dim, num_classes, P):
        super().__init__(base_model=VisionTransformer(input_dim, num_classes,
                                                      P=P, h=16, num_blocks=16, d_model=1024, dropout=0., mlp_size=4096),
                         num_classes=num_classes)


class ViTHuge(LightningModel):
    """ViT Base as defined in Dosovitskiy et al. 2020"""
    def __init__(self, input_dim, num_classes, P):
        super().__init__(base_model=VisionTransformer(input_dim, num_classes,
                                                      P=P, h=16, num_blocks=32, d_model=1280, dropout=0., mlp_size=5120),
                         num_classes=num_classes)
