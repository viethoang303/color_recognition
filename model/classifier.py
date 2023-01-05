from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, LambdaLR, ReduceLROnPlateau, StepLR
from torch.optim.lr_scheduler import _LRScheduler

import pytorch_lightning as pl
import torchmetrics
from torchmetrics import F1Score, ConfusionMatrix, Accuracy
import torchvision
# import clip

from collections import OrderedDict
from .attention import CBAM, BAM
import numpy as np
import math 

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, inital_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.inital_lr = inital_lr
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return self.inital_lr * float(step) / float(max(1, self.warmup_steps))
        return self.inital_lr * max(0.0,1.0 - float(step)/self.t_total)    #max(0.0, float(self.t_total - step) / float(self.t_total))
    

class VehicleClassifier(pl.LightningModule):
    def __init__(self, n_classes=14, learning_rate=1e-3):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.lr = learning_rate
        self.f1_score = F1Score(num_classes = n_classes, task="multiclass")
        self.acc = Accuracy(num_classes=n_classes, task="multiclass") 
        
        self.model = torchvision.models.efficientnet_v2_s(num_classes = n_classes)


    def forward(self, x):
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x.float())

        opt = self.optimizers()
        opt.zero_grad()
        loss = F.cross_entropy(y_hat, y)
        self.manual_backward(loss)
        opt.step()
        
        scheduler = self.lr_schedulers()
        scheduler.step()

        acc = self.acc(y_hat, y)
        f1_score = self.f1_score(y_hat, y)

        # class_pred = torch.max(y_hat.detach(), dim=1)[1]
        self.log('train_loss ', loss)
        self.log('train_f1_score', acc)
        self.log('train_acc', f1_score)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x.float())
        loss = F.cross_entropy(y_hat, y)
        acc = self.acc(y_hat, y)
        f1_score = self.f1_score(y_hat, y)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_f1_score', f1_score)

        return {'loss': loss, 'val_f1_score': f1_score, 'val_acc': acc}

    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x.float())
        loss = F.cross_entropy(y_hat, y)
        acc = self.acc(y_hat, y)
        f1_score = self.f1_score(y_hat, y)

        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_f1_score', f1_score)
    
        return {'loss': loss, 'test_f1_score': f1_score, 'test_acc': acc ,'y_hat': y_hat, 'y': y} 
 
    def configure_optimizers(self):
        # torch.optim.Adam(self.parameters(), weight_decay=1e-5)
        optimizer = torch.optim.RMSprop(self.parameters(), weight_decay=0.9, momentum=0.9)   
        lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100, t_total=16000, inital_lr=self.lr)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_f1_score"}
    

if __name__ == "__main__":
    model = VehicleClassifier(n_classes=14)
    print(model)
