import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class DefectDataset(Dataset):
    """
    Expects CSV with at least columns:
       image_path, status
    (status: 0 = good, 1 = defect)
    """
    def __init__(self, csv_file, root_dir, transform=None):
        df = pd.read_csv(csv_file)

        self.img_paths = df["image_path"].values
        self.labels = df["status"].astype(int).tolist()

        self.root = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.img_paths[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label