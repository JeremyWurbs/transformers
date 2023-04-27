import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from oak import MNIST, VisionTransformer, LightningModel


param = {
    'input_dim': [1, 28, 28],
    'num_classes': 10,
    'blocks': 4,
    'P': 28,
    'h': 8,
    'd_model': 64
}

dm = MNIST()
model = VisionTransformer(**param)
model = LightningModel(model)

logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"{os.path.join('experiments', param.__str__())}")
trainer = Trainer(max_epochs=3, logger=logger, val_check_interval=0.05)
trainer.fit(model, dm)
trainer.test(model, dm)
