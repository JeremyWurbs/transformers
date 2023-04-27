import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl


class LightningModel(pl.LightningModule):
    """Lightning wrapper for torch.nn.Modules"""

    def __init__(self, base_model: nn.Module, num_classes=10):
        super().__init__()
        self.model = base_model
        self.accuracy_metrics = {step_type: torchmetrics.Accuracy(task='multiclass', num_classes=num_classes) for step_type in ['train', 'val', 'test']}

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def _step(self, batch, step_type):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log(f'{step_type}_loss_step', loss)
        self.log(f'{step_type}_acc_step', self.accuracy_metrics[step_type](logits, y))
        return loss

    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, 'train')

    def validation_step(self, val_batch, batch_idx):
        return self._step(val_batch, 'val')

    def test_step(self, test_batch, batch_idx):
        return self._step(test_batch, 'test')

    def training_epoch_end(self, _):
        self.log('train_acc_epoch', self.accuracy_metrics['train'].compute())

    def validation_epoch_end(self, _):
        self.log('val_acc_epoch', self.accuracy_metrics['val'].compute())

    def test_epoch_end(self, _):
        self.log('test_acc_epoch', self.accuracy_metrics['test'].compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
