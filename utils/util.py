import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models as torchvision_models
from PIL import Image
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import cv2
import copy  # For saving the best model
import numpy as np

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
from torchvision import transforms
from PIL import Image
import torch

# === Define same transform as validation ===
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform_efficientnet = transforms.Compose([
    transforms.Resize((300, 300)),  # 👈 must match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image_for_missing_part_with_gradcam(
    image, model, target_layer_name, device='cpu', threshold=0.5
):
    model.eval()
    img_tensor = val_transform_efficientnet(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # --- 1. Hook for activations and gradients ---
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # --- 2. Register hooks ---
    target_layer = dict([*model.named_modules()])[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # --- 3. Forward pass ---
    output = model(img_tensor)
    prob = torch.sigmoid(output).item()
    pred = 1 if prob > threshold else 0

    # --- 4. Backward pass ---
    model.zero_grad()
    output.backward(torch.ones_like(output))

    # --- 5. Get activations and gradients ---
    acts = activations['value'][0]  # shape: [C, H, W]
    grads = gradients['value'][0]   # shape: [C, H, W]

    # --- 6. Compute weights & Grad-CAM map ---
    weights = grads.mean(dim=(1, 2))  # [C]
    cam = torch.zeros_like(acts[0])
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # --- 7. Resize CAM ---
    cam_resized = cv2.resize(cam, image.size)  # image.size = (W, H)

    # --- (Optional) Generate overlay for visualization only ---
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    image_np = np.array(image).astype(np.float32)
    heatmap = heatmap.astype(np.float32)

    overlay = 0.3 * image_np + 0.7 * heatmap
    overlay = np.uint8(np.clip(overlay, 0, 255))

    # --- 8. Cleanup hooks ---
    forward_handle.remove()
    backward_handle.remove()

    # --- 9. Return ---
    return prob, pred, overlay, cam_resized

def predict_with_gradcam_bird_nest(image, model, target_layer_name, device='cpu'):
    model.eval()
    input_tensor = val_transform_efficientnet(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # Store activations and gradients
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # Register hooks
    target_layer = dict([*model.named_modules()])[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)                  # shape: [1, num_classes]
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()

    # Backward pass for the predicted class
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, pred_class] = 1
    output.backward(gradient=one_hot)

    # Get hooked data
    acts = activations['value'][0]   # shape: [C, H, W]
    grads = gradients['value'][0]    # shape: [C, H, W]

    # Compute CAM
    weights = grads.mean(dim=(1, 2))  # shape: [C]
    cam = torch.zeros_like(acts[0])
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # Resize CAM and apply selective overlay
    cam_resized = cv2.resize(cam, image.size)  # image.size = (W, H)
    focus_mask = cam_resized > 0.4  # Only highlight strong activation areas

    # Generate heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert to float32 for blending
    image_np = np.array(image).astype(np.float32)
    heatmap = heatmap.astype(np.float32)

    # Create a copy of the original image
    overlay = image_np.copy()

    # Only blend pixels where mask is True
    overlay[focus_mask] = (
        0.3 * image_np[focus_mask] + 0.7 * heatmap[focus_mask]
    )

    # Convert back to uint8
    overlay = np.uint8(np.clip(overlay, 0, 255))

    # Cleanup
    forward_handle.remove()
    backward_handle.remove()

    return confidence, pred_class, overlay, cam_resized

def predict_image_with_gradcam(image, model, device='cpu', threshold=0.5, target_layer_name='layer4'):
    model.eval()

    # Transform and prepare input
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # Hook containers
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # Register hooks
    target_layer = dict([*model.named_modules()])[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor).squeeze()  # raw logit
    prob = torch.sigmoid(output).item()
    pred = 1 if prob > threshold else 0

    # Backward for Grad-CAM
    model.zero_grad()
    output.backward()

    # Extract data
    acts = activations['value'][0]  # shape: [C, H, W]
    grads = gradients['value'][0]   # shape: [C, H, W]

    # Compute Grad-CAM
    weights = grads.mean(dim=(1, 2))  # shape: [C]
    cam = torch.zeros_like(acts[0])
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # Resize and overlay
    # Resize CAM and apply selective overlay
    cam_resized = cv2.resize(cam, image.size)  # image.size = (W, H)
    focus_mask = cam_resized > 0.4  # Only highlight strong activation areas

    # Generate heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert to float32 for blending
    image_np = np.array(image).astype(np.float32)
    heatmap = heatmap.astype(np.float32)

    # Create a copy of the original image
    overlay = image_np.copy()

    # Only blend pixels where mask is True
    overlay[focus_mask] = (
        0.3 * image_np[focus_mask] + 0.7 * heatmap[focus_mask]
    )

    # Convert back to uint8
    overlay = np.uint8(np.clip(overlay, 0, 255))

    # Clean up
    forward_handle.remove()
    backward_handle.remove()

    return prob, pred, overlay, cam_resized


def predict_image_for_missing_part(image, model, device='cpu', threshold=0.5):
    model.eval()
    img = val_transform_efficientnet(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).squeeze()
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > threshold else 0

    return prob, pred

def predict_image(image, model, device='cpu', threshold=0.5):
    model.eval()

    # Load and preprocess image
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor).squeeze()
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > threshold else 0

    return prob, pred


def predict_single_image_bird_nest(image, model, device='cpu'):
    model.eval()
    input_tensor = val_transform_efficientnet(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)               # raw logits, shape: [1, 3]
        probs = F.softmax(output, dim=1)           # convert to probabilities
        pred = torch.argmax(probs, dim=1)          # predicted class index
        confidence = probs[0, pred.item()].item()  # probability of that class

    return confidence, pred.item()

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
    model = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

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
