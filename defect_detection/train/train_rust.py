import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import RustDataset, label_column
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

# === Config ===
csv_path = '/dataset/image_labels_with_code.csv'  # CSV must have columns: image_path, status (0 or 1)
image_root_dir = "/data/InsPLAD-fault/defect_supervised"
batch_size = 32
num_epochs = 50
patience = 5
learning_rate = 1e-4
weight_decay = 1e-5
model_code = 4
save_model_path = f'rust_detector_resnet_{model_code}.pt'
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# === Dataset ===


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

# === Load and modify model ===
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
model.to(device)

# === Balanced loss ===
n_rust = (train_df['status'] == 1).sum()
n_good = (train_df['status'] == 0).sum()
pos_weight = torch.tensor([n_good / n_rust], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# === Early Stopping Vars ===
best_val_recall = 0
early_stopping_counter = 0

# === Training ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    tp = fn = fp = tn = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # === Validation ===
    model.eval()
    val_tp = val_fn = val_fp = val_tn = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float()

            val_tp += ((preds == 1) & (labels == 1)).sum().item()
            val_fn += ((preds == 0) & (labels == 1)).sum().item()
            val_fp += ((preds == 1) & (labels == 0)).sum().item()
            val_tn += ((preds == 0) & (labels == 0)).sum().item()

    val_precision = val_tp / (val_tp + val_fp + 1e-6)
    val_recall = val_tp / (val_tp + val_fn + 1e-6)
    val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-6)

    print(f"Epoch {epoch+1}/{num_epochs} | Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    # === Early stopping based on recall ===
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        early_stopping_counter = 0
        torch.save(model.state_dict(), save_model_path)
        print(f"  🔥 New best recall! Model saved.")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"  ⛔ Early stopping triggered (no recall improvement in {patience} epochs).")
            break

print(f"Training complete. Best validation recall: {best_val_recall:.4f}")
