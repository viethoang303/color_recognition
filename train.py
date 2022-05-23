from argparse import ArgumentParser

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler

from data_loader import VehicleDataModule, get_map_classification
from model import VehicleClassifier

def cli_main():
    pl.seed_everything(1234)
    
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--n_classes', default=15, type=int)
    parser.add_argument('--data_path', default='vehicle', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VehicleClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------

    data = VehicleDataModule(args.data_path, batch_size=args.batch_size, num_workers=args.num_workers)
    data.setup()
    class_map = get_map_classification('vehicle/train')
    class_map = {value: key for key, value in class_map.items()}

    model = VehicleClassifier(n_classes=data.n_classes)

    trainer = pl.Trainer.from_argparse_args(
        args, 
        # profiler=AdvancedProfiler(),
        # callbacks = [
        #     ModelCheckpoint(),
        # ],
        max_epochs=100, 
        accelerator='gpu', 
        devices=[2]
    )
    
    # trainer.fit(model, data)

    #Testing#
    result = trainer.test(model=model, dataloaders=data.test_dataloader(), ckpt_path = 'lightning_logs/version_0/checkpoints/epoch=99-step=11400.ckpt')
    print(result)

if __name__ == '__main__':
    cli_main()

    

