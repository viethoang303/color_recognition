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

# class WarmupLinearSchedule(LambdaLR):
#     """ Linear warmup and then linear decay.
#         Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
#         Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
#     """
#     def __init__(self, optimizer, warmup_steps, t_total, inital_lr, last_epoch=-1):
#         self.warmup_steps = warmup_steps
#         self.t_total = t_total
#         self.inital_lr = inital_lr
#         super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

#     def lr_lambda(self, step):
#         if step < self.warmup_steps:
#             return self.inital_lr * float(step) / float(max(1, self.warmup_steps))
#         return self.inital_lr * max(0.0,1.0 - float(step)/self.t_total)    #max(0.0, float(self.t_total - step) / float(self.t_total))
    
class WarmupLR(_LRScheduler):
    def __init__(self, scheduler, init_lr=1e-3, num_warmup=1, warmup_strategy='linear'):
        if warmup_strategy not in ['linear', 'cos', 'constant']:
            raise ValueError("Expect warmup_strategy to be one of ['linear', 'cos', 'constant'] but got {}".format(warmup_strategy))
        self._scheduler = scheduler
        self._init_lr = init_lr
        self._num_warmup = num_warmup
        self._step_count = 0
        # Define the strategy to warm up learning rate 
        self._warmup_strategy = warmup_strategy
        if warmup_strategy == 'cos':
            self._warmup_func = self._warmup_cos
        elif warmup_strategy == 'linear':
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const
        # save initial learning rate of each param group
        # only useful when each param groups having different learning rate
        self._format_param()

    def __getattr__(self, name):
        return getattr(self._scheduler, name)
    
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        wrapper_state_dict = {key: value for key, value in self.__dict__.items() if (key != 'optimizer' and key !='_scheduler')}
        wrapped_state_dict = {key: value for key, value in self._scheduler.__dict__.items() if key != 'optimizer'} 
        return {'wrapped': wrapped_state_dict, 'wrapper': wrapper_state_dict}
    
    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict['wrapper'])
        self._scheduler.__dict__.update(state_dict['wrapped'])


    def _format_param(self):
        # learning rate of each param group will increase
        # from the min_lr to initial_lr
        for group in self._scheduler.optimizer.param_groups:
            group['warmup_max_lr'] = group['lr']
            group['warmup_initial_lr'] = min(self._init_lr, group['lr'])

    def _warmup_cos(self, start, end, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end)/2.0*cos_out
    
    def _warmup_const(self, start, end, pct):
        return start if pct < 0.9999 else end 

    def _warmup_linear(self, start, end, pct):
        return (end - start) * pct + start 

    def get_lr(self):
        lrs = []
        step_num = self._step_count
        # warm up learning rate 
        if step_num <= self._num_warmup:
            for group in self._scheduler.optimizer.param_groups:
                computed_lr = self._warmup_func(group['warmup_initial_lr'], 
                                                group['warmup_max_lr'],
                                                step_num/self._num_warmup)
                lrs.append(computed_lr)
        else:
            lrs = self._scheduler.get_lr()
        return lrs

    def step(self, *args):
        if self._step_count <= self._num_warmup:
            values = self.get_lr()
            for param_group, lr in zip(self._scheduler.optimizer.param_groups, values):
                param_group['lr'] = lr
            self._step_count += 1 
        else:
            self._scheduler.step(*args)

class VehicleClassifier(pl.LightningModule):
    def __init__(self, n_classes=14, learning_rate=1e-3):
        super().__init__()
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
        loss = F.cross_entropy(y_hat, y)
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
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.00001)
        # 
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-5)#torch.optim.RMSprop(self.parameters(), lr=self.lr)   
        lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=20, t_total=100, inital_lr=self.lr)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_f1_score"}
    

if __name__ == "__main__":
    model = VehicleClassifier(n_classes=14)
    print(model)
