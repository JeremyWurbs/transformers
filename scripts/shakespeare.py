import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from oak import Shakespeare, NLPTransformer, LightningModel

torch.set_float32_matmul_precision('medium')

param = {
    'num_blocks': 4,
    'seq_len': 8,
    'h': 4,
    'd_model': 32,
    'dropout': 0.2,
}

dm = Shakespeare(batch_size=32, block_size=param['seq_len'], num_workers=32)
model = NLPTransformer(vocab_size=dm.vocab_size, **param)
model = LightningModel(model, num_classes=dm.vocab_size)

logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"{os.path.join('experiments', param.__str__())}")
trainer = Trainer(max_epochs=1, logger=logger, val_check_interval=1.0, strategy='ddp_find_unused_parameters_true', devices=[1])
trainer.fit(model, dm)
trainer.test(model, dm)

sample_text = dm.train_data[1000]
generated_text = model.model.generate(sample_text[0].view(1, -1).to('cuda:1'), max_new_tokens=500)
print(f'{dm.tokenizer.decode(generated_text[0, :].cpu().detach().numpy())}')
