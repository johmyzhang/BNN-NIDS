import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class NetworkFeatureDataset(Dataset):
    def __init__(self, csv, transform=None, target_transform=None):
        self.df = pd.read_csv(csv)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx, :26].values.flatten()
        label = int(self.df.iloc[idx, 26])

        normalized = data / 255.0
        tensor = torch.FloatTensor(normalized).unsqueeze(0)

        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return tensor, label