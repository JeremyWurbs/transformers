import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm

from oak import Tokenizer


class IWSLTDataset(Dataset):
    def __init__(self, src_data, tgt_data, block_size, PAD):
        assert len(src_data) == len(tgt_data)
        super().__init__()
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.block_size = block_size

        # Special tokens
        self.PAD = torch.tensor(PAD, dtype=torch.int)  # Padding

    def __len__(self):
        return len(self.tgt_data)

    def __getitem__(self, idx):
        src_len = min(len(self.src_data[idx]), self.block_size)
        tgt_len = min(len(self.tgt_data[idx])-1, self.block_size)

        src = self.PAD * torch.ones((self.block_size,), dtype=torch.int)
        tgt = self.PAD * torch.ones((self.block_size,), dtype=torch.int)
        labels = self.PAD * torch.ones((self.block_size,), dtype=torch.long)

        src[:src_len] = torch.tensor(self.src_data[idx][:src_len], dtype=torch.int)
        tgt[:tgt_len] = torch.tensor(self.tgt_data[idx][:tgt_len], dtype=torch.int)
        labels[:tgt_len] = torch.tensor(self.tgt_data[idx][1:tgt_len+1], dtype=torch.long)

        return src, tgt, labels


class IWSLTDataModule(pl.LightningDataModule):
    def __init__(self, src_lang='de', tgt_lang='en', block_size=64, batch_size=32, num_workers=None, tokenizer=None, src_tokenizer=None, num_train=None, num_val=None, num_test=None):
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tgt_tokenizer = tokenizer if tokenizer is not None else Tokenizer(multilingual=True)
        self.src_tokenizer = src_tokenizer if src_tokenizer is not None else self.tgt_tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(os.cpu_count() // 2, 1)

        self.dataset = load_dataset('iwslt2017', f'iwslt2017-{src_lang}-{tgt_lang}')
        self._num_train = self.dataset['train'].num_rows if num_train is None else min(self.dataset['train'].num_rows, num_train)
        self._num_val = self.dataset['validation'].num_rows if num_val is None else min(self.dataset['validation'].num_rows, num_val)
        self._num_test = self.dataset['test'].num_rows if num_test is None else min(self.dataset['test'].num_rows, num_test)

        train_src_text = [self.src_tokenizer(sample[src_lang])[0] for sample in tqdm(self.dataset['train']['translation'][:self.num_train], desc='Preparing source training data')]
        train_tgt_text = [[self.tgt_tokenizer.BOS] + self.tgt_tokenizer(sample[tgt_lang])[0] + [self.tgt_tokenizer.EOS] for sample in tqdm(self.dataset['train']['translation'][:self.num_train], desc='Preparing target training data')]
        val_src_text = [self.src_tokenizer(sample[src_lang])[0] for sample in tqdm(self.dataset['validation']['translation'][:self.num_val], desc='Preparing source validation data')]
        val_tgt_text = [[self.tgt_tokenizer.BOS] + self.tgt_tokenizer(sample[tgt_lang])[0]+[self.tgt_tokenizer.EOS] for sample in tqdm(self.dataset['validation']['translation'][:self.num_val], desc='Preparing target validation data')]
        test_src_text = [self.src_tokenizer(sample[src_lang])[0] for sample in tqdm(self.dataset['test']['translation'][:self.num_test], desc='Preparing source testing data')]
        test_tgt_text = [[self.tgt_tokenizer.BOS] + self.tgt_tokenizer(sample[tgt_lang])[0]+[self.tgt_tokenizer.EOS] for sample in tqdm(self.dataset['test']['translation'][:self.num_test], desc='Preparing target testing data')]

        self.train = IWSLTDataset(train_src_text, train_tgt_text, block_size=block_size, PAD=self.tgt_tokenizer.PAD)
        self.val = IWSLTDataset(val_src_text, val_tgt_text, block_size=block_size, PAD=self.tgt_tokenizer.PAD)
        self.test = IWSLTDataset(test_src_text, test_tgt_text, block_size=block_size, PAD=self.tgt_tokenizer.PAD)

    @property
    def vocab_size(self):
        return self.tgt_tokenizer.vocab_size

    @property
    def src_vocab_size(self):
        return self.src_tokenizer.vocab_size

    @property
    def num_train(self):
        return self._num_train

    @property
    def num_val(self):
        return self._num_val

    @property
    def num_test(self):
        return self._num_test

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
