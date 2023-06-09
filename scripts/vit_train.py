import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from oak import LightningModel
from oak.data import MNIST
from oak.transformers import VisionTransformer

#torch.set_float32_matmul_precision('medium')

param = {
    'input_dim': [1, 28, 28],
    'num_classes': 10,
    'num_blocks': 4,
    'P': 28,
    'h': 8,
    'd_model': 64
}

dm = MNIST()
model = VisionTransformer(**param)
model = LightningModel(model)

logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"{os.path.join('experiments', param.__str__())}")
trainer = Trainer(max_epochs=20, logger=logger, val_check_interval=0.25)
trainer.fit(model, dm)
trainer.test(model, dm)
