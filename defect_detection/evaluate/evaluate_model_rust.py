import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model_code = 1
save_model_path = f'rust_detector_resnet_{model_code}.pt'
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# === Config ===
csv_path = '/dataset/image_labels_with_code.csv'
image_column = 'image_path'
label_column = 'status'
image_root_dir = "/data/InsPLAD-fault/defect_supervised"
batch_size = 32
num_epochs = 50
patience = 5
learning_rate = 1e-4
weight_decay = 1e-5
model_code = 1
save_model_path = f'rust_detector_resnet_{model_code}.pt'
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# === Dataset ===
class RustDataset(Dataset):
    def __init__(self, df, root_dir="", transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'status']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# === Load CSV and split ===
df = pd.read_csv(csv_path)
df = df[df['category_code'] == model_code]
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[label_column], random_state=42)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Dataloaders ===
train_loader = DataLoader(RustDataset(train_df, root_dir=image_root_dir, transform=transform), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(RustDataset(val_df,   root_dir=image_root_dir, transform=val_transform),   batch_size=batch_size)


# === Load the model ===
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(save_model_path, map_location=device))
model.to(device)
model.eval()

# === Prepare test dataset and loader ===
test_loader = DataLoader(RustDataset(test_df, root_dir=image_root_dir, transform=val_transform), batch_size=batch_size)

# === Inference ===
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# === Metrics ===
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4, target_names=["Good", "Rusty"]))