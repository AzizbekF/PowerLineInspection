import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import cv2
import copy  # For saving the best model

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def load_data_from_csv(csv_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV from: {csv_path}")
        print(f"CSV columns: {df.columns.tolist()}")
        # Ensure 'image_path' and 'status' columns exist
        if 'image_path' not in df.columns or 'status' not in df.columns:
            raise ValueError("CSV must contain 'image_path' and 'status' columns.")
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()


def split_data(df, test_size=0.1, random_state=42, stratify_col='status'):
    """Splits DataFrame into training and validation sets."""
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[stratify_col] if stratify_col in df.columns else None
        )
        print(f"Data split: {len(train_df)} training samples, {len(val_df)} validation samples.")
        if stratify_col in df.columns:
            print(f"Stratified by column: '{stratify_col}'")
            print(f"Training status distribution:\n{train_df[stratify_col].value_counts(normalize=True)}")
            print(f"Validation status distribution:\n{val_df[stratify_col].value_counts(normalize=True)}")
        return train_df, val_df
    except Exception as e:
        print(f"Error splitting data: {e}")
        exit()


class DefectDataset(Dataset):
    """Custom Dataset for defect detection."""

    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        # Ensure 'image_path' and 'status' columns exist (already checked in load_data_from_csv)
        self.image_paths = dataframe['image_path'].values
        self.labels = dataframe['status'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        # Construct full image path.
        # Assumes image_path in CSV is relative to image_dir
        full_img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(full_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {full_img_path}. Please check IMAGE_DIR and paths in CSV.")
            # Return a placeholder or raise an error, depending on desired handling
            # For now, let's raise an error to stop execution if an image is missing.
            raise FileNotFoundError(f"Image not found: {full_img_path}")
        except Exception as e:
            print(f"Error opening image {full_img_path}: {e}")
            raise

        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # For BCEWithLogitsLoss

        if self.transform:
            image = self.transform(image)

        return image, label.unsqueeze(0)  # Reshape label to [1] for BCEWithLogitsLoss


def get_data_loaders(train_df, val_df, image_dir, train_transform, val_transform, batch_size, num_workers=4):
    """Creates DataLoaders for training and validation."""
    train_dataset = DefectDataset(train_df, image_dir, transform=train_transform)
    val_dataset = DefectDataset(val_df, image_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    print(f"DataLoaders created. Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    return train_loader, val_loader


# --- 2. Model Related Functions ---

def get_pretrained_resnet(num_classes=1, pretrained=True, freeze_base=False):
    """
    Loads a pretrained ResNet model and modifies its final layer.
    Args:
        num_classes (int): Number of output classes. For binary (defect/good), this is 1 if using BCEWithLogitsLoss.
        pretrained (bool): Whether to load pretrained weights.
        freeze_base (bool): Whether to freeze the convolutional base layers.
    Returns:
        torch.nn.Module: The ResNet model.
    """
    # Using resnet18 as an example, can be changed to resnet34, resnet50 etc.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    if freeze_base and pretrained:
        print("Freezing base ResNet layers.")
        for param in model.parameters():
            param.requires_grad = False

    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"ResNet18 loaded. Final layer replaced for {num_classes} output features.")
    if freeze_base:
        print("Only the final layer will be trained initially.")
    else:
        print("All layers will be trained (or fine-tuned if pretrained).")

    return model.to(DEVICE)


def save_best_model(model_state, filepath):
    """Saves the model state dictionary."""
    torch.save(model_state, filepath)
    print(f"Best model saved to {filepath}")


# --- 3. Training and Evaluation Functions ---

def compute_metrics(logits, labels, thr=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    y_true = labels.cpu().numpy().astype(int)
    y_pred = (probs >= thr).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    return prec, rec, f1

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    tp = fp = fn = tn = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss

        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item() * images.size(0)

        # Calculate accuracy

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).byte()        # » 1 if defect else 0
        lbls  = labels.byte()

        total_samples += lbls.size(0)

        correct_predictions += (preds == lbls).sum().item()

        tp += (preds &  lbls).sum().item()
        fp += (preds & ~lbls).sum().item()
        fn += (~preds &  lbls).sum().item()
        tn += (~preds & ~lbls).sum().item()

        if (batch_idx + 1) % 10 == 0:  # Print progress every 10 batches
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # ── epoch metrics
    eps = 1e-8                              # avoid div‑by‑zero
    epoch_loss = running_loss / total_samples
    accuracy   = correct_predictions / total_samples
    precision  = tp / (tp + fp + eps)
    recall     = tp / (tp + fn + eps)
    f1         = 2 * precision * recall / (precision + recall + eps)

    return epoch_loss, accuracy, precision, recall, f1


def evaluate_model(model, val_loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    tp = fp = fn = tn = 0

    with torch.no_grad():  # Disable gradient calculations
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).byte()  # » 1 if defect else 0
            lbls = labels.byte()

            total_samples += labels.size(0)

            correct_predictions += (preds == lbls).sum().item()

            tp += (preds & lbls).sum().item()
            fp += (preds & ~lbls).sum().item()
            fn += (~preds & lbls).sum().item()
            tn += (~preds & ~lbls).sum().item()

    # ── epoch metrics
    eps = 1e-8                              # avoid div‑by‑zero
    epoch_loss = running_loss / total_samples
    accuracy   = correct_predictions / total_samples
    precision  = tp / (tp + fp + eps)
    recall     = tp / (tp + fn + eps)
    f1         = 2 * precision * recall / (precision + recall + eps)

    return epoch_loss, accuracy, precision, recall, f1


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs, early_stopping_patience, best_model_path):
    """Main training loop with early stopping."""
    best_val_recall = float(0)
    epochs_no_improve = 0
    best_model_state = None

    print(f"\nStarting training on {device} for up to {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        train_loss, train_acc, precision, recall, f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1} Training: Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

        print(f"Epoch done  Loss {train_loss:.3f}  Acc {train_acc:.3f}  "
              f"Prec {precision:.3f}  Rec {recall:.3f}  F1 {f1:.3f}")

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1} Validation: Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, recall: {val_recall:.4f}, f1: {val_f1:.4f}")

        # Early stopping and saving best model
        if val_recall >= best_val_recall:
            best_val_recall = val_recall
            epochs_no_improve = 0
            # Save a copy of the model state that achieved this best validation loss
            best_model_state = copy.deepcopy(model.state_dict())
            save_best_model(best_model_state, best_model_path)  # Save immediately
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            print(f"Best validation loss: {best_val_recall:.4f}")
            break  # Exit training loop

    if best_model_state is None and os.path.exists(best_model_path):
        print(
            f"No improvement from initial state, but a model might exist at {best_model_path} from a previous run or first epoch.")
    elif best_model_state is None:
        print("Training completed, but no model was saved as validation loss never improved.")

    print("Training finished.")
    return model  # Return the last state of the model (or best if loaded)



def predict_single(img, model, device):
    img_tfms = transforms.Compose([
        transforms.Resize((224, 224)),  # <— same size you used
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # <— same stats you used
                             [0.229, 0.224, 0.225]),
    ])

    """Return probability and binary label (0=good, 1=defect) for one image."""
    model.eval()
    x   = img_tfms(img).unsqueeze(0).to(device)        # shape 1×3×H×W

    with torch.no_grad():
        logit = model(x)                               # shape [1, 1]
        prob  = torch.sigmoid(logit).item()            # 0 – 1
        label = int(prob > 0.5)                        # threshold

    return prob, label


def crop_boxes_from_image(img_bgr, boxes_xyxy, pad=0):
    """
    Crop regions from an OpenCV BGR image given bounding boxes in absolute
    XYXY pixel coordinates.

    Args:
        img_bgr (np.ndarray): HxWx3 BGR image loaded by cv2.
        boxes_xyxy (np.ndarray | list): N×4 array [[x1,y1,x2,y2], ...].
        pad (int): Optional padding (pixels) added equally on all sides.

    Returns:
        List[np.ndarray] : list of cropped BGR images.
    """
    h, w = img_bgr.shape[:2]
    crops = []
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        # add optional padding & clip to image bounds
        x1p, y1p = max(x1 - pad, 0), max(y1 - pad, 0)
        x2p, y2p = min(x2 + pad, w - 1), min(y2 + pad, h - 1)
        crop = img_bgr[y1p: y2p, x1p: x2p].copy()
        crops.append(crop)
    return crops

def crop_object(img, box):
    x1, y1, x2, y2 = map(int, box.tolist())
    crop = img.crop((x1, y1, x2, y2))
    return crop

@torch.inference_mode()
def classify_crops(crops_bgr, model, device=DEVICE, prob_thr=0.5):
    """
    Simple loop that converts OpenCV crops to model tensor, runs classifier,
    and returns predicted labels (or probabilities).
    """
    preds = []
    preprocess = lambda im: (
            torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)  # HWC → CHW
            .float()
            / 255.0
    )
    for crop in crops_bgr:
        tensor = preprocess(crop).unsqueeze(0).to(device)
        logits = model(tensor)  # shape [1, C] or [1]
        probs = torch.sigmoid(logits) if logits.shape[-1] == 1 else torch.softmax(logits, 1)
        preds.append(probs.squeeze().cpu())
    return preds
