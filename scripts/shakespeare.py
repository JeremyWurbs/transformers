import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from oak import Shakespeare, NLPTransformer, LightningModel

param = {
    'num_blocks': 4,
    'seq_len': 8,
    'h': 8,
    'd_emb': 1,
    'd_model': 64,
    'batch_size': 4
}

dm = Shakespeare(batch_size=param['batch_size'], block_size=param['seq_len'])
model = NLPTransformer(vocab_size=dm.vocab_size, **param)
model = LightningModel(model, num_classes=dm.vocab_size)

logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"{os.path.join('experiments', param.__str__())}")
trainer = Trainer(max_epochs=3, logger=logger, val_check_interval=0.05)
trainer.fit(model, dm)
trainer.test(model, dm)
