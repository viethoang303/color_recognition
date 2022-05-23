import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

import cv2
# import PIL
from typing import Optional
import os
import glob

def get_fns_lbs(base_dir, map_classification):
    fns = [] # list of all images filenames
    lbs = [] # list of all labels
    cnt_classes = {}
    for idx, fn in enumerate(glob.glob(os.path.join(base_dir, "**"), recursive=True)):
        if os.path.isdir(fn) or os.path.getsize(fn) == 0:  # folder/ zero-byte files
            continue
        dir_name = os.path.dirname(fn)
        label = os.path.basename(dir_name)
        if label in map_classification:
            lbs.append(map_classification[label])
            fns.append(fn)
            if label not in cnt_classes:
                cnt_classes[label] = 0
            cnt_classes[label] += 1
    
    return fns, lbs, cnt_classes
        

class MyDataset(Dataset):
    def __init__(self, data_dir, transform = None,map_classification=None):
        filenames, labels, cnt_classes = get_fns_lbs(data_dir, map_classification=map_classification)
        
        assert len(filenames) == len(labels) # Number of files != number of labels
        print("#data: {} data_dir: {} cnt_class: {}".format(len(filenames), data_dir, cnt_classes))
        self.fns = filenames
        self.lbs = labels
        self.transform = transform
        self.n_classes = len(map_classification)
        

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        # TODO: replace Image by opencv
        # image = Image.open(self.fns[idx])
        image = cv2.imread(self.fns[idx])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # image.shape: HWC (100, 100, 3)
        tmp = image[0, 0]
        if isinstance(tmp, int) or len(tmp) != 3:  # not rgb image
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        image = cv2.resize(image, ((224, 224)), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)  # CHW
        return image, self.lbs[idx], self.fns[idx]
    

class VehicleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'path/to/data', batch_size: int = 32, num_workers: int = 8):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        map_classification = get_map_classification(os.path.join(self.data_dir, 'train'))
        self.train_set = MyDataset(os.path.join(self.data_dir, "train"), transform=None,
                              map_classification=map_classification)

        self.n_classes = self.train_set.n_classes
        self.val_set = MyDataset(os.path.join(self.data_dir, "val"), transform=None,
                                  map_classification=map_classification)
        self.test_set = MyDataset(os.path.join(self.data_dir, "test"), transform=None,
                                  map_classification=map_classification)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=False)


def get_map_classification(base_dir):
    classes = sorted(os.listdir(base_dir))
    map_classification = {}
    for idc, class_name in enumerate(classes):
        map_classification[class_name] = idc
    return map_classification


if __name__ == "__main__":
    datamodule = VehicleDataModule('vehicle', batch_size=1)
    datamodule.setup()
    trainloader = datamodule.train_dataloader()
    print(trainloader.__len__())