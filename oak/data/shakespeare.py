import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from oak import ShakespeareTokenizer


class ShakespeareDataset(Dataset):
    def __init__(self, data, block_size=8):
        assert len(data) >= block_size
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data)-self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size])
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1])
        return x, y


class ShakespeareDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer=None, block_size=8, batch_size=1, num_workers=1, train_percent=0.8, val_percent=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer if tokenizer is not None else ShakespeareTokenizer()

        assert train_percent + val_percent <= 1.

        with open(os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'shakespeare.txt'), 'r', encoding='utf-8') as f:
            text = self.tokenizer(f.read())[0]

        m = int(train_percent * len(text))
        n = int(val_percent * len(text))
        self.train_data = ShakespeareDataset(data=text[:m], block_size=block_size)
        self.val_data = ShakespeareDataset(data=text[m:m+n], block_size=block_size)
        self.test_data = ShakespeareDataset(data=text[m+n:], block_size=block_size)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def encode(self, str):
        return self.tokenizer.encode(str)

    def decode(self, idx):
        return self.tokenizer.decode(idx)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)
