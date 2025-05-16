import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import VariGripDataset, label_column
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
save_model_path = '../../models/efficientnet_b3_varigrip3.pt'
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

# === Model ===
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.classifier.named_parameters():
    param.requires_grad = True
model.to(device)

# === Class weighting (to handle imbalance) ===
class_counts = train_df[label_column].value_counts().sort_index().values  # [good, rusty, bird-nest]
class_weights = torch.tensor([class_counts.max() / c for c in class_counts], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# === Early Stopping Vars ===
best_val_f1 = 0
early_stopping_counter = 0

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    total = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

    # === Validation ===
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import classification_report, f1_score

    f1 = f1_score(val_labels, val_preds, average="macro")
    print(f"Epoch {epoch + 1}/{num_epochs} | Val Macro F1: {f1:.4f} | Train Acc: {correct / total:.4f}")

    # === Early stopping ===
    if f1 > best_val_f1:
        best_val_f1 = f1
        early_stopping_counter = 0
        torch.save(model.state_dict(), save_model_path)
        print(f"  🔥 New best F1! Model saved.")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"  ⛔ Early stopping triggered.")
            break

print(f"✅ Training complete. Best validation macro F1: {best_val_f1:.4f}")
