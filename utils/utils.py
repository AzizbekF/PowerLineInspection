import random
import torch
from collections import Counter

def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def calc_pos_weight(labels):
    """pos_weight for BCEWithLogitsLoss = #neg / #pos"""
    counter = Counter(labels)
    neg, pos = counter[0], counter[1]
    return torch.tensor(neg / max(pos, 1.))

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        outputs = model(imgs)
        val_loss += criterion(outputs, labels).item() * imgs.size(0)
        preds = (outputs.sigmoid() > 0.5).int()
        correct += (preds == labels.int()).sum().item()
        total += imgs.size(0)
    return val_loss / total, correct / total

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = (outputs.sigmoid() > 0.5).int()
        correct += (preds == labels.int()).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total