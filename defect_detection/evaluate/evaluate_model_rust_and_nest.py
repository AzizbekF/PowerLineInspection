import torch.nn as nn
import torch
import pandas as pd
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import VariGripDataset, label_column
from sklearn.metrics import classification_report, confusion_matrix

csv_path = '/dataset/image_labels_with_code.csv'  # CSV must have columns: image_path, status (0 or 1)
image_root_dir = "/data/InsPLAD-fault/defect_supervised"
batch_size = 16
num_epochs = 50
patience = 6
learning_rate = 1e-4
weight_decay = 1e-5
save_model_path = '../../models/efficientnet_b3_varigrip.pt'
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# === Load CSV and split ===
df = pd.read_csv(csv_path)
df = df[df["category_code"] == 3]
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[label_column], random_state=42)

# === Transforms ===
train_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Dataloaders ===
train_loader = DataLoader(VariGripDataset(train_df, root_dir=image_root_dir, transform=train_transform),
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(VariGripDataset(val_df, root_dir=image_root_dir, transform=val_transform),
                        batch_size=batch_size)

model_path = "../../models/efficientnet_b3_varigrip3.pt"
# === Load model
model = models.efficientnet_b3(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Prepare test set
test_loader = DataLoader(VariGripDataset(test_df, root_dir=image_root_dir, transform=val_transform), batch_size=batch_size)

# === Run inference
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === Print evaluation
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Good", "Rusty", "Bird-Nest"], digits=4))