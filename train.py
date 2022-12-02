from argparse import ArgumentParser

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.loggers import WandbLogger


from data_loader import VehicleDataModule, get_map_classification
from model import VehicleClassifier


import torch
import torchvision

# wandb_logger = WandbLogger(project="MLOps Basics")


def cli_main():
    pl.seed_everything(1234)
    
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_classes', default=14, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--data_path', default='color_data', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    # parser = VehicleClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------

    data = VehicleDataModule(args.data_path, batch_size=args.batch_size, num_workers=args.num_workers)
    data.setup()
    class_map = get_map_classification(args.data_path+'/train')
    class_map = {value: key for key, value in class_map.items()}

    model = VehicleClassifier(n_classes=data.n_classes, learning_rate=args.learning_rate)
    

    checkpoint_callback = ModelCheckpoint(
        save_top_k = 1, 
        monitor = 'val_loss',
        mode = 'min',
        dirpath = 'checkpoints',
        filename = "best-checkpoint"#{epoch:02d}-{val_f1_score:.4f}"
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(
        args, 
        # logger = wandb_logger,
        profiler=AdvancedProfiler(),
        callbacks = [
            checkpoint_callback,
            lr_monitor
        ],
        max_epochs=1, 
        # accelerator='gpu', 
        # devices=[args.device]
    )
    
    trainer.fit(model, data)

    #Testing
    result = trainer.test(model=model, dataloaders=data)#.test_dataloader(), ckpt_path = 'checkpoints/best-epoch=32-val_f1_score=0.7196.ckpt')
    print(result)

if __name__ == '__main__':
    cli_main()
