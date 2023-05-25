import torch
from pytorch_lightning import Trainer

from oak import Visualizer, LightningModel
from oak.data import MNIST
from oak.transformers import VisionTransformer

#torch.set_float32_matmul_precision('medium')

param = {
    'input_dim': [1, 28, 28],
    'num_classes': 10,
    'num_blocks': 2,
    'P': 14,
    'h': 2,
    'd_model': 64
}

dm = MNIST(batch_size=512, num_workers=48)
model = VisionTransformer(**param)
model = LightningModel(model)

# Pretrained PCAs
dm.setup()
visualizer = Visualizer(model, layers=['model.embedding', 'model.blocks.0.mha.heads.0', 'model.blocks.0.mha.heads.1', 'model.blocks.1.mha.heads.0', 'model.blocks.1.mha.heads.1'])
visualizer.collect_features(dm.test_dataloader())
visualizer.PCA_scatter(layers=['input', 'model.blocks.0.mha.heads.0', 'model.blocks.0.mha.heads.1', 'model.blocks.1.mha.heads.0', 'model.blocks.1.mha.heads.1', 'output'], k=3, embed_index=0)

# Train
trainer = Trainer(max_epochs=10, check_val_every_n_epoch=1)
trainer.fit(model, dm)
trainer.test(model, dm)

# Trained PCAs
visualizer.collect_features(dm.test_dataloader())
visualizer.PCA_scatter(layers=['input', 'model.blocks.0.mha.heads.0', 'model.blocks.0.mha.heads.1', 'model.blocks.1.mha.heads.0', 'model.blocks.1.mha.heads.1', 'output'], k=3, embed_index=0)
