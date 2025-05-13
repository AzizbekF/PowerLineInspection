import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from ulrtalytics import RTDETR
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import time
import random
from collections import Counter


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Multi-class State Classifier model
class PowerlineStateClassifier(nn.Module):
    def __init__(self, num_states=11, num_categories=5, backbone="efficientnet_b0", use_category_embedding=True):
        super().__init__()
        self.use_category_embedding = use_category_embedding

        # Use a pretrained model as feature extractor
        self.backbone_name = backbone
        if backbone == "resnet50":
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='IMAGENET1K_V1')
            feature_dim = 2048
        elif backbone == "efficientnet_b0":
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', weights='IMAGENET1K_V1')
            feature_dim = 1280
        elif backbone == "mobilenet_v2":
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='IMAGENET1K_V1')
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Category embedding (optional)
        if use_category_embedding:
            self.category_embedding = nn.Embedding(num_categories, 128)  # embedding dimension
            self.fc_combined = nn.Linear(feature_dim + 128, 512)
            self.classifier = nn.Linear(512, num_states)
        else:
            # Direct classifier from visual features
            self.classifier = nn.Linear(feature_dim, num_states)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x, category_ids=None):
        # Extract visual features
        features = self.backbone(x)
        features = torch.flatten(features, 1)

        if self.use_category_embedding and category_ids is not None:
            # Get category embeddings
            category_features = self.category_embedding(category_ids)

            # Combine visual and category features
            combined = torch.cat([features, category_features], dim=1)
            combined = self.fc_combined(combined)
            combined = self.relu(combined)
            combined = self.dropout(combined)

            # Final classification
            logits = self.classifier(combined)
        else:
            # Direct classification from visual features
            logits = self.classifier(features)

        return logits


# Dataset class for powerline component states
class PowerlineStateDataset(Dataset, root_path):
    def __init__(self, df, transform=None, category_to_idx=None, status_to_idx=None):
        """
        Args:
            df: DataFrame with columns: category, category_code, image_path, status_name, status_code
            transform: Optional image transformations
            category_to_idx: Dictionary mapping category codes to indices
            status_to_idx: Dictionary mapping status codes to indices
        """
        self.df = df
        self.transform = transform
        self.root_path = root_path

        # Create mappings if not provided
        if category_to_idx is None:
            self.category_to_idx = {code: idx for idx, code in enumerate(self.df['category_code'].unique())}
        else:
            self.category_to_idx = category_to_idx

        if status_to_idx is None:
            self.status_to_idx = {code: idx for idx, code in enumerate(self.df['status_code'].unique())}
        else:
            self.status_to_idx = status_to_idx

        self.idx_to_category = {v: k for k, v in self.category_to_idx.items()}
        self.idx_to_status = {v: k for k, v in self.status_to_idx.items()}

        # Create human-readable mappings
        self.category_code_to_name = dict(zip(self.df['category_code'], self.df['category']))
        self.status_code_to_name = dict(zip(self.df['status_code'], self.df['status_name']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            # Load and transform image
            image_path = row['image_path']
            image_path = os.path.join(self.root_path, image_path)
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            # Get category and status indices
            category_code = row['category_code']
            status_code = row['status_code']

            category_idx = self.category_to_idx[category_code]
            status_idx = self.status_to_idx[status_code]

            return image, torch.tensor(category_idx), torch.tensor(status_idx)

        except Exception as e:
            print(f"Error loading image {row['image_path']}: {e}")
            # Return a placeholder in case of error
            return torch.zeros((3, 224, 224)), torch.tensor(0), torch.tensor(0)

    def get_class_weights(self):
        """
        Calculate class weights inversely proportional to class frequencies
        """
        status_counts = Counter(self.df['status_code'].map(self.status_to_idx))
        total = len(self.df)
        weights = {status: total / count for status, count in status_counts.items()}

        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {status: weight / weight_sum * len(weights) for status, weight in weights.items()}

        return weights


# Create balanced sampler for training
def create_weighted_sampler(dataset):
    """
    Create a sampler that balances classes
    """
    class_weights = dataset.get_class_weights()

    # Assign weight to each sample based on its class
    sample_weights = []
    for idx in range(len(dataset)):
        _, _, status_idx = dataset[idx]
        sample_weights.append(class_weights[status_idx.item()])

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


# Process RT-DETR detection with state classifier
def process_detection(detection, image, state_model, transform, category_to_idx, idx_to_status,
                      status_code_to_name, device='cuda'):
    """
    Process a single RT-DETR detection with the state classifier
    """
    bbox = detection["bbox"]
    class_name = detection["class"]
    category_code = detection.get("category_code", class_name)  # Use class name if code not provided

    # Check if category exists in mapping
    if category_code not in category_to_idx:
        print(f"Warning: Unknown category {category_code}, skipping")
        return None

    # Get category index
    category_idx = category_to_idx[category_code]
    category_tensor = torch.tensor([category_idx]).to(device)

    # Crop object from image
    x1, y1, x2, y2 = bbox
    object_crop = image.crop((x1, y1, x2, y2))

    # Preprocess the cropped image
    processed_crop = transform(object_crop).unsqueeze(0).to(device)

    # Get state prediction
    with torch.no_grad():
        outputs = state_model(processed_crop, category_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(outputs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    # Get human-readable state
    status_code = idx_to_status[pred_idx]
    status_name = status_code_to_name.get(status_code, f"Status {pred_idx}")

    # Determine if it's a defect based on the status name
    is_defect = "defect" in status_name.lower() or "rust" in status_name.lower() or \
                "missing" in status_name.lower() or "extra" in status_name.lower() or \
                "cracked" in status_name.lower() or "bent" in status_name.lower() or \
                "bird" in status_name.lower()

    # Return result
    return {
        "bbox": bbox,
        "class": class_name,
        "category_code": category_code,
        "state": status_name,
        "state_code": status_code,
        "confidence": confidence,
        "is_defect": is_defect
    }


# RT-DETR model class
class RTDETRModel:
    """
    Implementation of the RT-DETR model for powerline component detection
    This is where you would implement your actual RT-DETR model
    """

    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device

        # Load your RT-DETR model here
        # This is a placeholder - in a real implementation, you would:
        self.model = RTDETR(model_path)
        self.model.to(device)
        # 1. Load your trained RT-DETR model
        # 2. Move it to the appropriate device
        # 3. Set it to evaluation mode
        print(f"Initialized RT-DETR model from {model_path}")

    def __call__(self, image):
        """
        Process an image and return detections

        Args:
            image: PIL Image to process

        Returns:
            List of detections, each with:
            - bbox: (x1, y1, x2, y2) coordinates
            - class: Class name
            - score: Detection confidence
        """
        # This is where you would implement your RT-DETR inference
        # For demonstration, we'll create dummy detections
        # Replace this with your actual RT-DETR implementation

        detections = self.model.predict(image)

        return detections


# Real implementation of training and testing
def train_and_test_model():
    """
    Complete implementation for training and testing powerline defect detection system
    """
    # Set random seed for reproducibility
    set_seed(42)

    # Define parameters
    csv_path = "./dataset/labels_with_status_code.csv"  # Path to your CSV file
    rtdetr_model_path = "models/object/best.pt"  # Path to your RT-DETR model
    output_dir = "output/powerline_defect"  # Output directory
    test_split = 0.2  # 20% of data for testing
    val_split = 0.1  # 10% of remaining data for validation
    num_epochs = 25  # Number of training epochs
    batch_size = 32  # Batch size
    learning_rate = 0.0005  # Learning rate
    backbone = "efficientnet_b0"  # Backbone architecture

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Create sample data if the CSV doesn't exist
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found. Creating sample data.")

        # Create directories
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Create sample DataFrame
        df = pd.DataFrame(sample_data)

        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Created sample CSV at {csv_path}")


    # Step 2: Load and prepare data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Check if all image files exist
    missing_images = []
    for idx, row in df.iterrows():
        if not os.path.exists(row['image_path']):
            missing_images.append(row['image_path'])

    if missing_images:
        print(f"Warning: {len(missing_images)} image files are missing:")
        for path in missing_images[:5]:
            print(f"  - {path}")
        if len(missing_images) > 5:
            print(f"  - ... and {len(missing_images) - 5} more")

    # Show dataset statistics
    print("\nCategory distribution:")
    print(df['category'].value_counts())

    print("\nStatus distribution:")
    print(df['status_name'].value_counts())

    # Create mappings
    category_to_idx = {code: idx for idx, code in enumerate(df['category_code'].unique())}
    status_to_idx = {code: idx for idx, code in enumerate(df['status_code'].unique())}

    num_categories = len(category_to_idx)
    num_states = len(status_to_idx)

    print(f"\nNumber of categories: {num_categories}")
    print(f"Number of states: {num_states}")

    # Step 3: Split data
    try:
        # Try to stratify by both category and status if enough samples
        train_val_df, test_df = train_test_split(
            df, test_size=test_split, random_state=42,
            stratify=df[['category_code', 'status_code']]
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=val_split / (1 - test_split), random_state=42,
            stratify=train_val_df[['category_code', 'status_code']]
        )
    except ValueError:
        # Fallback to stratify by just status_code if not enough samples
        print("Warning: Not enough samples for stratification by both category and status.")
        print("Falling back to stratification by status only.")

        train_val_df, test_df = train_test_split(
            df, test_size=test_split, random_state=42,
            stratify=df['status_code']
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=val_split / (1 - test_split), random_state=42,
            stratify=train_val_df['status_code']
        )

    print(f"\nSplit data into:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Step 4: Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 5: Create datasets and data loaders
    train_dataset = PowerlineStateDataset(
        train_df, transform=train_transform,
        category_to_idx=category_to_idx,
        status_to_idx=status_to_idx
    )

    val_dataset = PowerlineStateDataset(
        val_df, transform=val_transform,
        category_to_idx=category_to_idx,
        status_to_idx=status_to_idx
    )

    test_dataset = PowerlineStateDataset(
        test_df, transform=val_transform,
        category_to_idx=category_to_idx,
        status_to_idx=status_to_idx
    )

    # Create balanced sampler for training
    train_sampler = create_weighted_sampler(train_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=0, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # Step 6: Create and train the model
    print(f"\nInitializing model with backbone: {backbone}")
    state_model = PowerlineStateClassifier(
        num_states=num_states,
        num_categories=num_categories,
        backbone=backbone,
        use_category_embedding=True
    )
    state_model = state_model.to(device)

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(state_model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        state_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, category_ids, status_ids in train_loader:
            images = images.to(device)
            category_ids = category_ids.to(device)
            status_ids = status_ids.to(device)

            # Forward pass
            outputs = state_model(images, category_ids)
            loss = criterion(outputs, status_ids)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += status_ids.size(0)
            train_correct += (predicted == status_ids).sum().item()

        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total

        # Validation phase
        state_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, category_ids, status_ids in val_loader:
                images = images.to(device)
                category_ids = category_ids.to(device)
                status_ids = status_ids.to(device)

                outputs = state_model(images, category_ids)
                loss = criterion(outputs, status_ids)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += status_ids.size(0)
                val_correct += (predicted == status_ids).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Calculate epoch time
        epoch_time = time.time() - start_time

        print(f'Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.1f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print('-' * 50)

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'category_to_idx': category_to_idx,
                'status_to_idx': status_to_idx,
                'backbone': backbone,
                'num_states': num_states,
                'num_categories': num_categories,
            }, best_model_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

    # Step 7: Load the best model for evaluation
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    state_model.load_state_dict(checkpoint['model_state_dict'])
    state_model.eval()

    # Step 8: Evaluate on test set
    print("\nEvaluating on test set...")

    # Initialize variables
    all_preds = []
    all_labels = []
    all_categories = []

    # Evaluate on test set
    with torch.no_grad():
        for images, category_ids, status_ids in test_loader:
            images = images.to(device)
            category_ids = category_ids.to(device)
            status_ids = status_ids.to(device)

            outputs = state_model(images, category_ids)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(status_ids.cpu().numpy())
            all_categories.extend(category_ids.cpu().numpy())

    # Convert indices to readable names
    idx_to_status = {v: k for k, v in status_to_idx.items()}
    idx_to_category = {v: k for k, v in category_to_idx.items()}

    status_code_to_name = dict(zip(df['status_code'], df['status_name']))
    category_code_to_name = dict(zip(df['category_code'], df['category']))

    # Overall performance
    print("\nOverall test performance:")
    test_acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    print(f"Test accuracy: {test_acc:.2f}%")

    # Generate classification report
    status_names = [status_code_to_name.get(idx_to_status[i], f"Status {i}")
                    for i in range(num_states)]

    print("\nClassification report:")
    clf_report = classification_report(
        all_labels, all_preds,
        target_names=status_names,
        output_dict=True
    )

    # Save classification report as CSV
    clf_report_df = pd.DataFrame(clf_report).transpose()
    clf_report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=status_names, yticklabels=status_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Per-category performance
    print("\nPer-category performance:")
    for cat_idx in sorted(set(all_categories)):
        cat_code = idx_to_category[cat_idx]
        cat_name = category_code_to_name.get(cat_code, f"Category {cat_idx}")

        cat_indices = [i for i, c in enumerate(all_categories) if c == cat_idx]
        if not cat_indices:
            continue

        cat_preds = [all_preds[i] for i in cat_indices]
        cat_labels = [all_labels[i] for i in cat_indices]

        cat_acc = (np.array(cat_preds) == np.array(cat_labels)).mean() * 100
        print(f"{cat_name}: {cat_acc:.2f}% accuracy")

    # Step 9: Visualize sample predictions
    print("\nGenerating sample prediction visualizations...")

    # Function to denormalize images
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ])

    # Get a batch of test data
    dataiter = iter(test_loader)
    images, categories, labels = next(dataiter)

    # Get predictions
    images_cuda = images.to(device)
    categories_cuda = categories.to(device)
    with torch.no_grad():
        outputs = state_model(images_cuda, categories_cuda)
        _, preds = torch.max(outputs, 1)

    # Convert to numpy for visualization
    images = images.cpu().numpy()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    categories = categories.cpu().numpy()

    # Plot sample predictions
    num_samples = min(5, len(images))

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        # Denormalize and convert to image
        img = images[i].transpose(1, 2, 0)  # CHW -> HWC
        img = denormalize(torch.from_numpy(img.copy())).numpy()
        img = np.clip(img, 0, 1)

        # Get labels
        cat_code = idx_to_category[categories[i]]
        cat_name = category_code_to_name.get(cat_code, f"Category {categories[i]}")

        true_code = idx_to_status[labels[i]]
        true_name = status_code_to_name.get(true_code, f"Status {labels[i]}")

        pred_code = idx_to_status[preds[i]]
        pred_name = status_code_to_name.get(pred_code, f"Status {preds[i]}")

        # Set title color based on prediction
        title_color = 'green' if preds[i] == labels[i] else 'red'

        # Display image
        axes[i].imshow(img)
        axes[i].set_title(f"{cat_name}\nTrue: {true_name}\nPred: {pred_name}",
                          color=title_color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    plt.close()

    # Step 10: Test with RT-DETR integration
    print("\nTesting integration with RT-DETR...")

    # Initialize RT-DETR model
    rtdetr_model = RTDETRModel(rtdetr_model_path, device)

    # Create a visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Test on a few selected test images
    test_image_paths = test_df['image_path'].unique()[:3]  # Limit to 3 images for demonstration

    for img_path in test_image_paths:
        print(f"\nProcessing {os.path.basename(img_path)}")

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            img_np = np.array(image)

            # Detect objects with RT-DETR
            detections = rtdetr_model(image)
            print(f"Detected {len(detections)} objects")

            # Process each detection
            results = []

            for detection in detections:
                result = process_detection(
                    detection, image, state_model, val_transform,
                    category_to_idx, idx_to_status, status_code_to_name, device
                )

                if result:
                    results.append(result)
                    print(f"  - {result['class']}: {result['state']} (Confidence: {result['confidence']:.2f})")

            # Create visualization
            if results:
                output_img = img_np.copy()

                for result in results:
                    bbox = result["bbox"]
                    category = result["class"]
                    state = result["state"]
                    confidence = result["confidence"]
                    is_defect = result["is_defect"]

                    # Determine color based on state
                    color = (0, 255, 0) if not is_defect else (255, 0, 0)  # Green for normal, red for defect

                    # Draw bounding box
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

                    # Prepare label text
                    label = f"{category}: {state} ({confidence:.2f})"

                    # Draw label background
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(
                        output_img,
                        (x1, y1 - text_size[1] - 10),
                        (x1 + text_size[0] + 10, y1),
                        color,
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        output_img,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                # Save visualization
                base_name = os.path.basename(img_path)
                output_path = os.path.join(vis_dir, f"result_{base_name}")
                cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
                print(f"Saved visualization to {output_path}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Step 11: Save model metadata
    metadata = {
        'num_categories': num_categories,
        'num_states': num_states,
        'backbone': backbone,
        'final_test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'training_epochs': num_epochs,
        'category_count': len(category_to_idx),
        'status_count': len(status_to_idx)
    }

    pd.DataFrame([metadata]).to_csv(os.path.join(output_dir, 'model_metadata.csv'), index=False)

    print("\nSaved model and results to:", output_dir)
    print(f"Final test accuracy: {test_acc:.2f}%")
    print("Training and testing completed successfully!")

    return state_model, test_acc


if __name__ == "__main__":
    train_and_test_model()
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# Constants
ASSET_CATEGORIES = [
    'Glass Insulator', 'Lightning Rod Suspension',
    'Polymer Insulator Upper Shackle', 'Vari-grip', 'Yoke Suspension'
]


# 1. Define the Defect Classifier Model
class DefectClassifier(nn.Module):
    def __init__(self, num_classes=2, backbone="resnet50"):
        super().__init__()
        # Use a pretrained model as feature extractor
        self.backbone_name = backbone
        if backbone == "resnet50":
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            feature_dim = 2048
        elif backbone == "efficientnet_b0":
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=True)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add asset-specific classification heads
        self.asset_classifiers = nn.ModuleDict({
            asset: nn.Linear(feature_dim, num_classes) for asset in ASSET_CATEGORIES
        })

        # Add a default classifier for unknown asset types
        self.default_classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, asset_type=None):
        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1)

        if asset_type is None or isinstance(asset_type, list):
            # Batch processing with potentially different asset types
            outputs = []
            if asset_type is None:
                # If no asset type provided, use default classifier for all
                return self.default_classifier(features)

            # Process each sample with its corresponding asset classifier
            for i, asset in enumerate(asset_type):
                if asset in self.asset_classifiers:
                    classifier = self.asset_classifiers[asset]
                else:
                    classifier = self.default_classifier

                # Get the prediction for this sample
                output = classifier(features[i:i + 1])
                outputs.append(output)

            return torch.cat(outputs, dim=0)
        else:
            # Single asset type for the whole batch
            if asset_type in self.asset_classifiers:
                return self.asset_classifiers[asset_type](features)
            else:
                return self.default_classifier(features)


# 2. Define the Anomaly Detection Model
class AnomalyDetector(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode the input
        latent = self.encoder(x)
        # Decode the latent representation
        reconstructed = self.decoder(latent)
        return reconstructed


# 3. Dataset and Data Loading
class PowerlineDefectDataset(Dataset):
    def __init__(self, image_paths, asset_types, labels=None, transform=None):
        self.image_paths = image_paths
        self.asset_types = asset_types
        self.labels = labels  # None for inference
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            asset_type = self.asset_types[idx]

            if self.transform:
                image = self.transform(image)

            if self.labels is not None:
                label = self.labels[idx]
                return image, asset_type, label
            else:
                return image, asset_type
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a placeholder in case of error
            if self.labels is not None:
                return torch.zeros((3, 224, 224)), ASSET_CATEGORIES[0], 0
            else:
                return torch.zeros((3, 224, 224)), ASSET_CATEGORIES[0]


def create_balanced_sampler(dataset):
    # Count samples per class
    class_sample_count = {}
    for _, _, label in dataset:
        asset_label = (_, label)
        if asset_label not in class_sample_count:
            class_sample_count[asset_label] = 0
        class_sample_count[asset_label] += 1

    # Calculate weights
    weights = []
    for i in range(len(dataset)):
        _, asset_type, label = dataset[i]
        asset_label = (asset_type, label)
        weight = 1.0 / class_sample_count[asset_label]
        weights.append(weight)

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler


# 4. Training functions
def train_defect_classifier(train_loader, val_loader, model, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, asset_types, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, asset_types)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, asset_types, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images, asset_types)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.2f}%')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_defect_model.pth')

    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    return model, history


def train_anomaly_detector(train_loader, val_loader, model, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer (MSE for reconstruction)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for images, _, _ in train_loader:  # We only need normal images for training
            images = images.to(device)

            # Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, _, _ in val_loader:
                images = images.to(device)

                reconstructed = model(images)
                loss = criterion(reconstructed, images)

                val_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.8f}, '
              f'Val Loss: {val_loss:.8f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_anomaly_model.pth')

    print(f'Best validation loss: {best_val_loss:.8f}')
    return model, history


# 5. Inference and evaluation functions
def evaluate_defect_classifier(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_asset_types = []

    with torch.no_grad():
        for images, asset_types, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, asset_types)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_asset_types.extend(asset_types)

    # Overall metrics
    print("Overall Performance:")
    print(classification_report(all_labels, all_preds))

    # Per-asset metrics
    for asset in set(all_asset_types):
        asset_indices = [i for i, a in enumerate(all_asset_types) if a == asset]
        asset_preds = [all_preds[i] for i in asset_indices]
        asset_labels = [all_labels[i] for i in asset_indices]

        print(f"\nPerformance for {asset}:")
        print(classification_report(asset_labels, asset_preds))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')

    return all_preds, all_labels, all_asset_types


def evaluate_anomaly_detector(model, normal_loader, anomaly_loader, threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Compute reconstruction errors for normal samples
    normal_errors = []
    with torch.no_grad():
        for images, _, _ in normal_loader:
            images = images.to(device)
            reconstructed = model(images)

            # Calculate per-sample MSE
            for i in range(images.size(0)):
                error = torch.mean((images[i] - reconstructed[i]) ** 2).item()
                normal_errors.append(error)

    # Compute reconstruction errors for anomaly samples
    anomaly_errors = []
    with torch.no_grad():
        for images, _, _ in anomaly_loader:
            images = images.to(device)
            reconstructed = model(images)

            # Calculate per-sample MSE
            for i in range(images.size(0)):
                error = torch.mean((images[i] - reconstructed[i]) ** 2).item()
                anomaly_errors.append(error)

    # If threshold not provided, compute it from normal distribution
    if threshold is None:
        # Compute threshold as mean + 3*std of normal errors
        mean_error = np.mean(normal_errors)
        std_error = np.std(normal_errors)
        threshold = mean_error + 3 * std_error

    # Evaluate with the threshold
    normal_preds = [1 if e > threshold else 0 for e in normal_errors]  # 1 for anomaly
    anomaly_preds = [1 if e > threshold else 0 for e in anomaly_errors]

    # Calculate metrics
    true_positives = sum(anomaly_preds)
    false_negatives = len(anomaly_preds) - true_positives
    true_negatives = len(normal_preds) - sum(normal_preds)
    false_positives = sum(normal_preds)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Threshold: {threshold:.8f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot histogram of errors
    plt.figure(figsize=(10, 6))
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal')
    plt.hist(anomaly_errors, bins=50, alpha=0.5, label='Anomaly')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.8f})')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig('error_distribution.png')

    return threshold, normal_errors, anomaly_errors


# 6. End-to-End Pipeline
class PowerlineInspectionSystem:
    def __init__(self, detector_model, defect_model=None, anomaly_model=None, anomaly_threshold=None):
        """
        Initialize the powerline inspection system.

        Args:
            detector_model: Your RT-DETR model for object detection
            defect_model: Trained defect classification model
            anomaly_model: Trained anomaly detection model
            anomaly_threshold: Threshold for anomaly detection
        """
        self.detector_model = detector_model
        self.defect_model = defect_model
        self.anomaly_model = anomaly_model
        self.anomaly_threshold = anomaly_threshold

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def inspect_image(self, image_path):
        """
        Perform full inspection on an image.

        Args:
            image_path: Path to the input image

        Returns:
            List of detected objects with their defect/anomaly information
        """
        # Load the image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            # Assume PIL Image
            image = image_path

        # Step 1: Detect objects using RT-DETR
        detections = self.detector_model(image)

        results = []
        for detection in detections:
            # Extract bounding box and class
            bbox = detection["bbox"]
            asset_class = detection["class"]

            # Crop the detected object
            x1, y1, x2, y2 = bbox
            object_crop = image.crop((x1, y1, x2, y2))

            # Preprocess the cropped image
            processed_crop = self.transform(object_crop).unsqueeze(0)

            # Initialize result dictionary
            result = {
                "bbox": bbox,
                "class": asset_class,
                "defect_status": None,
                "defect_confidence": None,
                "anomaly_score": None,
                "is_anomaly": None
            }

            # Step 2: Perform defect classification if model is available
            if self.defect_model is not None:
                device = next(self.defect_model.parameters()).device
                processed_crop = processed_crop.to(device)

                with torch.no_grad():
                    defect_output = self.defect_model(processed_crop, asset_class)
                    defect_probs = torch.softmax(defect_output, dim=1)
                    defect_pred = torch.argmax(defect_output, dim=1).item()
                    defect_conf = defect_probs[0][defect_pred].item()

                result["defect_status"] = "Fault" if defect_pred == 1 else "Normal"
                result["defect_confidence"] = defect_conf

            # Step 3: Perform anomaly detection if model is available
            if self.anomaly_model is not None:
                device = next(self.anomaly_model.parameters()).device
                processed_crop = processed_crop.to(device)

                with torch.no_grad():
                    reconstructed = self.anomaly_model(processed_crop)
                    recon_error = torch.mean((processed_crop - reconstructed) ** 2).item()

                result["anomaly_score"] = recon_error
                result["is_anomaly"] = recon_error > self.anomaly_threshold if self.anomaly_threshold else None

            results.append(result)

        return results

    def visualize_results(self, image_path, results):
        """
        Visualize the inspection results on the input image.

        Args:
            image_path: Path to the input image
            results: Inspection results from inspect_image()

        Returns:
            PIL Image with visualization
        """
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        # Load the image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path

        # Convert to numpy for OpenCV operations
        img_np = np.array(image)

        # Define colors for different classes and statuses
        colors = {
            "Normal": (0, 255, 0),  # Green
            "Fault": (255, 0, 0),  # Red
            "Anomaly": (0, 0, 255)  # Blue
        }

        # Draw bounding boxes and labels
        for result in results:
            bbox = result["bbox"]
            asset_class = result["class"]
            defect_status = result["defect_status"]
            defect_conf = result.get("defect_confidence", 0)
            is_anomaly = result.get("is_anomaly", False)

            # Determine color based on defect status or anomaly
            if is_anomaly:
                color = colors["Anomaly"]
                status = "Anomaly"
            elif defect_status == "Fault":
                color = colors["Fault"]
                status = "Fault"
            else:
                color = colors["Normal"]
                status = "Normal"

            # Convert PIL bbox to OpenCV format
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Prepare label text
            label = f"{asset_class}: {status}"
            if defect_conf:
                label += f" ({defect_conf:.2f})"

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                img_np,
                (int(x1), int(y1) - text_size[1] - 10),
                (int(x1) + text_size[0], int(y1)),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                img_np,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Convert back to PIL Image
        return Image.fromarray(img_np)


# 7. Example Usage
def example_usage():
    """
    Example of how to use the powerline inspection system.
    """
    print("Setting up the powerline inspection system...")

    # 1. Load your pre-trained object detector
    # This is a placeholder for your RT-DETR model
    class RTDETRDetector:
        def __init__(self, model_path):
            # Load your actual model here
            self.model_path = model_path
            print(f"Loaded object detector from {model_path}")

        def __call__(self, image):
            # This is just a placeholder - in reality you'd use your RT-DETR model
            print("Detecting objects in image...")
            # Return dummy detections for demonstration
            return [
                {"bbox": (100, 100, 300, 300), "class": "Glass Insulator"},
                {"bbox": (400, 200, 600, 400), "class": "Yoke Suspension"}
            ]

    # 2. Create and load the defect classifier
    defect_model = DefectClassifier(num_classes=2)
    try:
        defect_model.load_state_dict(torch.load('best_defect_model.pth'))
        print("Loaded defect classifier from best_defect_model.pth")
    except:
        print("Could not load defect model weights - using untrained model")

    # 3. Create and load the anomaly detector
    anomaly_model = AnomalyDetector()
    try:
        anomaly_model.load_state_dict(torch.load('best_anomaly_model.pth'))
        print("Loaded anomaly detector from best_anomaly_model.pth")
        # You would normally load or compute the threshold from validation
        anomaly_threshold = 0.01  # Example threshold
    except:
        print("Could not load anomaly model weights - using untrained model")
        anomaly_threshold = None

    # 4. Create the inspection system
    detector = RTDETRDetector(model_path="path/to/your/rtdetr/model.pth")
    inspection_system = PowerlineInspectionSystem(
        detector_model=detector,
        defect_model=defect_model,
        anomaly_model=anomaly_model,
        anomaly_threshold=anomaly_threshold
    )

    # 5. Inspect an image
    image_path = "path/to/your/test/image.jpg"
    print(f"Inspecting image: {image_path}")

    # You would load a real image here
    # For this example, we'll create a dummy black image
    dummy_image = Image.new('RGB', (800, 600), color='black')

    results = inspection_system.inspect_image(dummy_image)

    # 6. Print and visualize results
    for result in results:
        print(f"Detected {result['class']}:")
        print(f"  - Bounding box: {result['bbox']}")
        print(f"  - Defect status: {result['defect_status']}")
        print(f"  - Defect confidence: {result['defect_confidence']}")
        print(f"  - Anomaly score: {result['anomaly_score']}")
        print(f"  - Is anomaly: {result['is_anomaly']}")

    # 7. Visualize results
    vis_image = inspection_system.visualize_results(dummy_image, results)
    vis_image.save("inspection_results.jpg")
    print("Saved visualization to inspection_results.jpg")


# 8. Training Script
def train_models(data_dir, output_dir, asset_categories=None):
    """
    Train both defect classifier and anomaly detector on the provided data.

    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save trained models and results
        asset_categories: List of asset categories to train on (default: use all categories)
    """
    os.makedirs(output_dir, exist_ok=True)

    if asset_categories is None:
        asset_categories = ASSET_CATEGORIES

    print(f"Training models for {len(asset_categories)} asset categories: {asset_categories}")

    # Define image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare dataset paths and labels
    train_normal_paths = []
    train_normal_assets = []
    train_fault_paths = []
    train_fault_assets = []

    val_normal_paths = []
    val_normal_assets = []
    val_fault_paths = []
    val_fault_assets = []

    test_normal_paths = []
    test_normal_assets = []
    test_fault_paths = []
    test_fault_assets = []
    test_anomaly_paths = []
    test_anomaly_assets = []

    # In a real implementation, you would load your dataset
    # For this example, we'll just create placeholders

    # 1. Train the defect classifier
    # Prepare datasets
    train_paths = train_normal_paths + train_fault_paths
    train_assets = train_normal_assets + train_fault_assets
    train_labels = [0] * len(train_normal_paths) + [1] * len(train_fault_paths)

    val_paths = val_normal_paths + val_fault_paths
    val_assets = val_normal_assets + val_fault_assets
    val_labels = [0] * len(val_normal_paths) + [1] * len(val_fault_paths)

    test_paths = test_normal_paths + test_fault_paths
    test_assets = test_normal_assets + test_fault_assets
    test_labels = [0] * len(test_normal_paths) + [1] * len(test_fault_paths)

    # Create datasets
    train_dataset = PowerlineDefectDataset(train_paths, train_assets, train_labels, transform=train_transform)
    val_dataset = PowerlineDefectDataset(val_paths, val_assets, val_labels, transform=val_transform)
    test_dataset = PowerlineDefectDataset(test_paths, test_assets, test_labels, transform=val_transform)

    # Create balanced samplers
    train_sampler = create_balanced_sampler(train_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create and train the defect classifier
    defect_model = DefectClassifier(num_classes=2, backbone="efficientnet_b0")
    defect_model, history = train_defect_classifier(
        train_loader, val_loader, defect_model,
        num_epochs=20, lr=0.001
    )

    # Save the trained model
    torch.save(defect_model.state_dict(), os.path.join(output_dir, "defect_model.pth"))

    # Evaluate the model
    test_preds, test_labels, test_assets = evaluate_defect_classifier(defect_model, test_loader)

    # 2. Train the anomaly detector
    # Create datasets for anomaly detection (only normal samples for training)
    train_anomaly_dataset = PowerlineDefectDataset(
        train_normal_paths, train_normal_assets,
        [0] * len(train_normal_paths), transform=train_transform
    )

    val_anomaly_dataset = PowerlineDefectDataset(
        val_normal_paths, val_normal_assets,
        [0] * len(val_normal_paths), transform=val_transform
    )

    test_normal_anomaly_dataset = PowerlineDefectDataset(
        test_normal_paths, test_normal_assets,
        [0] * len(test_normal_paths), transform=val_transform
    )

    test_anomaly_dataset = PowerlineDefectDataset(
        test_anomaly_paths, test_anomaly_assets,
        [1] * len(test_anomaly_paths), transform=val_transform
    )

    # Create data loaders
    train_anomaly_loader = DataLoader(train_anomaly_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_anomaly_loader = DataLoader(val_anomaly_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_normal_loader = DataLoader(test_normal_anomaly_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_anomaly_loader = DataLoader(test_anomaly_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create and train the anomaly detector
    anomaly_model = AnomalyDetector()
    anomaly_model, anomaly_history = train_anomaly_detector(
        train_anomaly_loader, val_anomaly_loader, anomaly_model,
        num_epochs=20, lr=0.001
    )

    # Save the trained model
    torch.save(anomaly_model.state_dict(), os.path.join(output_dir, "anomaly_model.pth"))

    # Evaluate the anomaly detector
    threshold, normal_errors, anomaly_errors = evaluate_anomaly_detector(
        anomaly_model, test_normal_loader, test_anomaly_loader
    )

    # Save the threshold
    with open(os.path.join(output_dir, "anomaly_threshold.txt"), "w") as f:
        f.write(str(threshold))

    print("Training and evaluation completed!")
    print(f"Models saved to {output_dir}")


if __name__ == "__main__":
    example_usage()
    # To train models, uncomment:
    # train_models("path/to/data", "output")