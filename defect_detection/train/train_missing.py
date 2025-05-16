import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import InsulatorDataset, label_column
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

# === Config ===
csv_path = '/dataset/image_labels_with_code.csv'  # CSV must have columns: image_path, status (0 or 1)
image_root_dir = "/data/InsPLAD-fault/defect_supervised"
batch_size = 16
num_epochs = 50
patience = 6
learning_rate = 1e-4
weight_decay = 1e-5
save_model_path = '../../models/efficientnet_b3_missing_cup.pt'
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# === Load CSV and split ===
df = pd.read_csv(csv_path)
df = df[df["category_code"] == 0]
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[label_column], random_state=42)

# === Transforms (EfficientNetB3 expects 300x300 input) ===
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
train_loader = DataLoader(InsulatorDataset(train_df, root_dir=image_root_dir, transform=train_transform),
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(InsulatorDataset(val_df, root_dir=image_root_dir, transform=val_transform),
                        batch_size=batch_size)

# === Model: EfficientNet-B3 ===
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

# Freeze all except classifier
for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.classifier.named_parameters():
    param.requires_grad = True

model.to(device)

# === Loss and Optimizer ===
# Dataset is balanced, so no pos_weight needed
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# === Early Stopping Vars ===
best_val_recall = 0
early_stopping_counter = 0

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

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

    print(
        f"Epoch {epoch + 1}/{num_epochs} | Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

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

print(f"✅ Training complete. Best validation recall: {best_val_recall:.4f}")
