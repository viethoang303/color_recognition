from argparse import ArgumentParser

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import F1Score

from model import AlexNet


class VehicleClassifier(pl.LightningModule):
    def __init__(self, n_classes=15, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = AlexNet()
        self.lr = learning_rate
        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()
        # self.test_acc = torchmetrics.Accuracy()

        self.f1_score = F1Score(num_classes = n_classes)


    def forward(self, x):
        out = self.backbone(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.backbone(x.float())
        loss = F.cross_entropy(y_hat, y)

        # class_pred = torch.max(y_hat.detach(), dim=1)[1]
        acc = self.f1_score(y_hat, y)
        self.log('train_loss ', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.backbone(x.float())
        loss = F.cross_entropy(y_hat, y)
        acc = self.f1_score(y_hat, y)

        self.log('val_loss', loss)
        # self.log('val_accuracy', acc)

        return {'loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.backbone(x.float())
        loss = F.cross_entropy(y_hat, y)
        acc = self.f1_score(y_hat, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return {'loss': loss, 'accuracy': acc}
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--learning_rate', type=float, default=0.0001, required=False
        )
        return parser
