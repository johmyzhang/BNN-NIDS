import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PacketImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        target = self.img_labels.iloc[idx, 1]
        image = Image.open(img_path)
        img_2d = image.convert('L')
        img = np.array(img_2d)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target