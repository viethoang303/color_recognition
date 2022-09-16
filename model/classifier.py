from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
import torchmetrics
from torchmetrics import F1Score, ConfusionMatrix
import torchvision
import clip

from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt

class VehicleClassifier(pl.LightningModule):
    def __init__(self, n_classes=14, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.lr = learning_rate
        self.f1_score = F1Score(num_classes = n_classes)

        # self.backbone = torchvision.models.efficientnet_v2_s()
        # self.backbone.load_state_dict(torch.load('pretrained_checkpoint_model/efficientnet_v2_s-dd5fe13b.pth'))
        # self.model, preprocess = clip.load(clip.load("ViT-L/14", device="cuda"))
        # self.model.float()

        # self.model = torchvision.models.vit_b_16(pretrained=False)
        # self.model.load_state_dict(torch.load('vit_b_16_imagenet.pt'))
        # modules = list(self.model.children())[:-1]
        # self.model = nn.Sequential(*list(modules))

        self.model = torchvision.models.efficientnet_v2_s(num_classes=n_classes)
        # self.model.load_state_dict(torch.load("pretrained_checkpoint_model/efficientnet_v2_s-dd5fe13b.pth"))
        # modules = list(self.model.children())[:-1]
        # self.model = nn.Sequential(*list(modules))
        # self.gelu = nn.GeLU(approximate="none")   
        # self.fc1 = nn.Linear(1280, n_classes, bias=True)
        # self.model = torchvision.models.resnet50(num_classes=n_classes)


    def forward(self, x):
        # if self.current_epoch < 15:
        #     with torch.no_grad():
        #         out = self.model(x)
        # else:
        out = self.model(x)
        
        # out = out.squeeze(dim=2).squeeze(dim=2)
        # out = self.fc1(out)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x.float())
        loss = F.cross_entropy(y_hat, y)

        # class_pred = torch.max(y_hat.detach(), dim=1)[1]
        acc = self.f1_score(y_hat, y)
        self.log('train_loss ', loss)
        self.log('train_f1_score', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x.float())
        loss = F.cross_entropy(y_hat, y)
        acc = self.f1_score(y_hat, y)

        self.log('val_loss', loss)
        self.log('val_f1_score', acc)

        return {'loss': loss, 'val_f1_score': acc}
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x.float())
        loss = F.cross_entropy(y_hat, y)
        acc = self.f1_score(y_hat, y)

        
        self.log('test_loss', loss)
        self.log('test_f1_score', acc)

        return {'loss': loss, 'test_f1_score': acc, 'y_hat': y_hat, 'y': y} 

    # def test_epoch_end(self, outputs):
    #     preds = torch.cat([tmp['y_hat'] for tmp in outputs])
    #     targets = torch.cat([tmp['y'] for tmp in outputs])

    #     confusion_matrix = torchmetrics.functional.classification.multiclass_confusion_matrix(preds.cpu(), targets.cpu(), self.n_classes)
    #     self.log('confusion_matrix', confusion_matrix)
        # df_cm = pd.DataFrame(confusion_matrix.numpy(), index = range(self.n_classes), columns=range(self.n_classes))
        # plt.figure(figsize = (10,7))
        # fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        # plt.close(fig_)
        
        # self.logger.experiment.add_figure("Confusion matrix", fig_)
        
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.00001)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=0)
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr)   
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, last_epoch=15)
        return optimizer#{"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_f1_score"}#torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=0.9, momentum=0.9, eps=0.001)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--learning_rate', type=float, default=1e-3, required=False
        )
        return parser

if __name__ == "__main__":
    model = VehicleClassifier(n_classes=14)
    print(model)