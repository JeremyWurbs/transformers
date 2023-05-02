import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from oak import Visualizer


class LightningModel(pl.LightningModule):
    """Lightning wrapper for torch.nn.Modules"""

    def __init__(self, base_model: nn.Module, num_classes=10, use_visualizer=False, visualizer_args=None):
        super().__init__()
        self.use_visualizer = use_visualizer
        if use_visualizer:
            self.model = Visualizer(base_model, **visualizer_args)
            self.val_PCA = list()
        else:
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
        self.log(f'{step_type}_loss_step', loss, sync_dist=True)
        self.log(f'{step_type}_acc_step', self.accuracy_metrics[step_type].to(x.device)(logits, y), sync_dist=True)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, 'train')

    def validation_step(self, val_batch, batch_idx):
        if self.use_visualizer:
            if batch_idx == 0:
                self.model.reset_features()
            self.model.step(batch=val_batch)

        return self._step(val_batch, 'val')

    def test_step(self, test_batch, batch_idx):
        return self._step(test_batch, 'test')

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.accuracy_metrics['train'].compute(), sync_dist=True)

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy_metrics['val'].compute(), sync_dist=True)
        if self.use_visualizer:
            PCA = self.model.PCA()
            self.val_PCA.append(PCA)

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.accuracy_metrics['test'].compute(), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
