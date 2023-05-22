import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from oak import Shakespeare, NLPTransformer, LightningModel, ShakespeareTokenizer

#torch.set_float32_matmul_precision('medium')

param = {
    'num_blocks': 6,
    'seq_len': 256,
    'h': 6,
    'd_model': 384,
    'dropout': 0.2,
    'batch_size': 64,
    'lr': 3e-4,
}

dm = Shakespeare(batch_size=param['batch_size'], block_size=param['seq_len'], num_workers=48, tokenizer=ShakespeareTokenizer())
model = NLPTransformer(vocab_size=dm.vocab_size, **param)

gen_id = [0]
def validation_hook():
    sample_text = dm.test_data[0]
    generated_text = model.model.generate(sample_text[0].view(1, -1).to('cuda:1'), max_new_tokens=250)
    generated_text = dm.tokenizer.decode(generated_text[0, :].cpu().detach().numpy())[0]
    print(f"{generated_text}")
    with open('generated_text.txt', 'a') as f:
        f.write(f"Sample {gen_id[0]}:\n\n{generated_text}\n\n\n\n\n\n\n\n")
    gen_id[0] += 1


model = LightningModel(model, num_classes=dm.vocab_size, lr=param['lr'], validation_hook=validation_hook)
logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"{os.path.join('experiments', param.__str__())}")
trainer = Trainer(max_epochs=1, logger=logger, val_check_interval=.2, devices=[1])
trainer.fit(model, dm)
trainer.test(model, dm)
