import os
from typing import Optional

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # download data
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=False)

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        self.train, self.val = random_split(MNIST(os.getcwd(), train=True, transform=transform), [55000, 5000])
        self.test = MNIST(os.getcwd(), train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
