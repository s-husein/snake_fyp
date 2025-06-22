import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from paths import *
import random
import numpy as np


class DepthImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.label_to_idx = {label: idx for idx, label in enumerate(self.data['direction'].unique())}
        self.data['label_idx'] = self.data['direction'].map(self.label_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['depth_image'])
        image = Image.open(img_name).convert('L')  # grayscale
        label = self.data.iloc[idx]['label_idx']
        if self.transform:
            image = self.transform(image)
        return image, label
    
    @property
    def classes(self):
        return self.data['direction'].unique()
    
    def __getrawimage__(self):
        idx = random.randint(0, self.__len__()-1)
        img_path = os.path.join(self.image_dir, self.data.iloc[idx]['depth_image'])
        return np.array(Image.open(img_path))
    
    def __report__(self):
        return f'''\nDataset size: {self.__len__()}\n
                {self.data['direction'].value_counts()}'''



    

class DepthImageDataModule:
    def __init__(self, train_trans, val_trans, test_trans,
                 train_batch_size = 32, test_batch_size = 32,
                 val_batch_size = 32):
        num_workers = os.cpu_count()

        if num_workers >= 12:
            num_workers = 12

        self.train_ds = DepthImageDataset(os.path.join(SPLIT_DIR, 'train.csv'), IMAGE_DIR, train_trans)
        self.val_ds = DepthImageDataset(os.path.join(SPLIT_DIR, 'val.csv'), IMAGE_DIR, val_trans)
        self.test_ds = DepthImageDataset(os.path.join(SPLIT_DIR, 'test.csv'), IMAGE_DIR, test_trans)

        self.train_loader = DataLoader(self.train_ds, batch_size=train_batch_size, shuffle=True, 
                                       num_workers=num_workers, pin_memory=True)
        
        self.val_loader = DataLoader(self.val_ds, batch_size=val_batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
        
        self.test_loader = DataLoader(self.test_ds, batch_size=test_batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)

