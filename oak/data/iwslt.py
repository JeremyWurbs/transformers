import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm

from oak import Tokenizer


class IWSLTDataset(Dataset):
    def __init__(self, src_data, tgt_data, block_size):
        assert len(src_data) == len(tgt_data)
        super().__init__()
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.block_size = block_size

    def __len__(self):
        return len(self.tgt_data)-self.block_size

    def __getitem__(self, idx):
        src = torch.tensor(self.src_data[idx][:self.block_size])
        tgt = torch.tensor(self.tgt_data[idx][:self.block_size])
        return src, tgt


class IWSLTDataModule(pl.LightningDataModule):
    def __init__(self, src_lang='de', tgt_lang='en', block_size=64, batch_size=32, num_workers=None, src_tokenizer=None, tgt_tokenizer=None):
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer if src_tokenizer is not None else Tokenizer(multilingual=True)
        self.tgt_tokenizer = tgt_tokenizer if src_tokenizer is not None else Tokenizer(multilingual=True)
        self.dataset = load_dataset('iwslt2017', f'iwslt2017-{src_lang}-{tgt_lang}')
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(os.cpu_count() // 2, 1)

        train_src_text = [src_tokenizer(sample[src_lang]) for sample in tqdm(self.dataset['train']['translation'], desc='Preparing source training data')]
        train_tgt_text = [tgt_tokenizer(sample[tgt_lang]) for sample in tqdm(self.dataset['train']['translation'], desc='Preparing target training data')]
        val_src_text = [src_tokenizer(sample[src_lang]) for sample in tqdm(self.dataset['validation']['translation'], desc='Preparing source validation data')]
        val_tgt_text = [tgt_tokenizer(sample[tgt_lang]) for sample in tqdm(self.dataset['validation']['translation'], desc='Preparing target validation data')]
        test_src_text = [src_tokenizer(sample[src_lang]) for sample in tqdm(self.dataset['test']['translation'], desc='Preparing source testing data')]
        test_tgt_text = [tgt_tokenizer(sample[tgt_lang]) for sample in tqdm(self.dataset['test']['translation'], desc='Preparing target testing data')]

        self.train = IWSLTDataset(train_src_text, train_tgt_text, block_size=block_size)
        self.val = IWSLTDataset(val_src_text, val_tgt_text, block_size=block_size)
        self.test = IWSLTDataset(test_src_text, test_tgt_text, block_size=block_size)

    @property
    def num_train(self):
        return self.dataset['train'].num_rows

    @property
    def num_val(self):
        return self.dataset['validation'].num_rows

    @property
    def num_test(self):
        return self.dataset['test'].num_rows

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
