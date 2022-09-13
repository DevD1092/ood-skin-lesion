import pandas as pd
import numpy as np
import torch
import torchvision
import PIL

from torch.utils.data import Dataset
from torchvision import transforms
from fastai import *
from fastai.vision import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ISICDataset(Dataset):
    def __init__(self, image_folder, df, transform, size=256, is_train=True, test_mode=False):
        self.tfms = transform
        self.p = image_folder
        self.paths = list(df.image)
        self.labels = list(df.label)
        self.size = size
        self.is_train = is_train
        self.test_mode = test_mode
        if not test_mode:
            val_index = np.linspace(0, len(self.paths), len(self.paths) // 5, endpoint=False, dtype=np.int)
            train_index = np.setdiff1d(np.arange(len(self.paths)), val_index)
            np.save('isic_val_index.npy',val_index)
            np.save('isic_train_index.npy',train_index)
            if is_train:
                self.paths = np.array(self.paths)[train_index]
                self.labels = np.array(df.label)[train_index]
                print('training samples number is: ', len(self.paths))
            else:
                self.paths = np.array(self.paths)[val_index]
                self.labels = np.array(df.label)[val_index]
                print('validation samples number is: ', len(self.paths))

    def __getitem__(self, idx):
        # Convert image to tensor and pre-process using transform
        img = PIL.Image.open(self.p+'/'+self.paths[idx]+'.jpg').convert('RGB')
        img = self.tfms(img)

        # Convert caption to tensor of word ids.
        target = torch.tensor(self.labels[idx], dtype=torch.int64)

        # return pre-processed image and caption tensor
        if self.test_mode == False:
            return img, target
        else:
            return img, target, idx

    def __len__(self):
        return len(self.paths)