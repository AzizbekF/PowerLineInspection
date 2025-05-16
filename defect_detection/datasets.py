from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torch

image_column = 'image_path'
label_column = 'status'

class InsulatorDataset(Dataset):
    def __init__(self, df, root_dir="", transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.df.loc[idx, image_column]
        label = self.df.loc[idx, label_column]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


class RustDataset(Dataset):
    def __init__(self, df, root_dir="", transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.df.loc[idx, image_column]
        label = self.df.loc[idx, label_column]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


class VariGripDataset(Dataset):
    def __init__(self, df, root_dir="", transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.df.loc[idx, image_column]
        label = self.df.loc[idx, label_column]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)