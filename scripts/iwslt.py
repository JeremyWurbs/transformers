import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from oak import IWSLT, Transformer, CrossAttentionLightningModel, Tokenizer

torch.set_float32_matmul_precision('medium')

param = {
    'num_blocks': 6,
    'seq_len': 256,
    'h': 6,
    'd_model': 384,
    'dropout': 0.2,
    'batch_size': 32,
    'lr': 1e-3,
}

tokenizer = Tokenizer(multilingual=True)
dm = IWSLT(batch_size=param['batch_size'], block_size=param['seq_len'], num_workers=48, tokenizer=tokenizer,
           num_train=1e10, num_val=1e10, num_test=1e10)
model = Transformer(vocab_size=dm.vocab_size, PAD=tokenizer.PAD, BOS=tokenizer.BOS, EOS=tokenizer.EOS, **param)

gen_id = [0]
def validation_hook(max_new_tokens=1000):
    src_sentence, tgt_sentence, _ = dm.test[5000]
    translated_text = model.model.translate(src_sentence.view(1, -1), max_new_tokens=max_new_tokens)
    translated_text = dm.tgt_tokenizer.decode(translated_text[0, :].cpu().detach().numpy())[0]
    print(f"Sample {gen_id[0]}:\n\nSource Text: {dm.src_tokenizer.decode(src_sentence.cpu().detach().numpy())[0]}, \nTarget Text: {dm.tgt_tokenizer.decode(tgt_sentence.cpu().detach().numpy())[0]}, \nTranslated Text: {translated_text}\n\n\n\n\n\n\n\n")
    with open('generated_text.txt', 'a') as f:
        f.write(f"Sample {gen_id[0]}:\n\nSource Text: {dm.src_tokenizer.decode(src_sentence.cpu().detach().numpy())[0]}, \nTarget Text: {dm.tgt_tokenizer.decode(tgt_sentence.cpu().detach().numpy())[0]}, \nTranslated Text: {translated_text}\n\n\n\n\n\n\n\n")
    gen_id[0] += 1


model = CrossAttentionLightningModel(model, num_classes=dm.vocab_size, lr=param['lr'], validation_hook=validation_hook)
logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"{os.path.join('experiments', param.__str__())}")
trainer = Trainer(max_epochs=10, logger=logger, val_check_interval=.2, devices=[1])
trainer.fit(model, dm)
trainer.test(model, dm)

validation_hook()
