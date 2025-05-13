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