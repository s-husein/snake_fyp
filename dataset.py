import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from paths import *
import random
import numpy as np
import torch


class DepthImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, sequence_len = 5, transform=None, sequence_step=3):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.seq_len = sequence_len
        self.seq_step = sequence_step

        self.label_to_idx = {label: idx for idx, label in enumerate(self.data['direction'].unique())}
        self.data['label_idx'] = self.data['direction'].map(self.label_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = []
        for i in range(self.seq_len):
            data_idx = idx - (self.seq_len - 1 - i) * self.seq_step
            if data_idx < 0:
                data_idx = 0  # Clamp to start of dataset
            img_path = os.path.join(self.image_dir, self.data.iloc[data_idx]['depth_image'])
            image = Image.open(img_path).convert('L')  # Load grayscale image
            if self.transform:
                image = self.transform(image)
            seq.append(image)

        # Stack sequence into a single tensor of shape [seq_len, C, H, W]
        sequence_tensor = torch.stack(seq, dim=0)

        # Get label for the final frame in the sequence (idx)
        label = self.data.iloc[idx]['label_idx']
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(-1)

        return sequence_tensor, label_tensor
        
    
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
                 val_batch_size = 32, sequence_len = 10, sequence_step=3):
        num_workers = os.cpu_count()

        if num_workers >= 12:
            num_workers = 12

        self.train_ds = DepthImageDataset('data.csv', IMAGE_DIR, sequence_len, train_trans, sequence_step)
        self.val_ds = DepthImageDataset('data.csv', IMAGE_DIR, sequence_len, val_trans, sequence_step)
        self.test_ds = DepthImageDataset('data.csv', IMAGE_DIR, sequence_len, test_trans, sequence_step)

        self.train_loader = DataLoader(self.train_ds, batch_size=train_batch_size, shuffle=True, 
                                       num_workers=num_workers, pin_memory=True)
        
        self.val_loader = DataLoader(self.val_ds, batch_size=val_batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=True)
        
        self.test_loader = DataLoader(self.test_ds, batch_size=test_batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True)

