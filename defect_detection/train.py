import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# --- Configuration ---
# Set your dataset root directory
DATA_DIR = 'path/to/your/dataset' # e.g., './dataset'

# Image size for model input (common for many pre-trained models)
IMG_SIZE = 224

# Batch size for training and validation
BATCH_SIZE = 32

# Number of training epochs
NUM_EPOCHS = 25

# Learning rate for the optimizer
LEARNING_RATE = 0.001

# Momentum for SGD optimizer (if using SGD)
# MOMENTUM = 0.9

# Device to use for training ('cuda' if GPU available, 'cpu' otherwise)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading and Preparation ---

# Define transformations for training and validation data
# Training transformations include data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE), # Crop the image to IMG_SIZE
    transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
    transforms.RandomRotation(15),         # Randomly rotate the image by up to 15 degrees
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Randomly change brightness, contrast, saturation, hue
    transforms.ToTensor(),                 # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize using ImageNet mean and std
])

# Validation transformations are typically just resizing and normalization
val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),          # Resize the image
    transforms.CenterCrop(IMG_SIZE),      # Crop the center of the image
    transforms.ToTensor(),                # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize using ImageNet mean and std
])

# Load the dataset using ImageFolder
# ImageFolder automatically infers class labels from directory names
full_dataset = datasets.ImageFolder(DATA_DIR)

# Get class names (should be 'defect' and 'normal' or similar)
class_names = full_dataset.classes
print(f"Detected classes: {class_names}")

# Check if the expected classes are present
if 'defect' not in class_names or 'normal' not in class_names:
    print("Warning: 'defect' or 'normal' class not found. Please check directory names.")
    # You might need to map your directory names to 'defect' and 'normal'
    # For example, if your directories are 'has_defect' and 'no_defect':
    # class_mapping = {'has_defect': 0, 'no_defect': 1} # or vice versa
    # Then manually create dataset and assign targets based on this mapping.
    # For simplicity, this script assumes 'defect' and 'normal' directory names exist.


# Split the dataset into training and validation sets
# We'll use an 80/20 split as an example
# Stratify ensures that the proportion of classes is the same in both splits
train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=full_dataset.targets,
    random_state=42 # for reproducibility
)

# Create Subset datasets for training and validation
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Apply the transformations to the respective datasets
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

# Create DataLoaders
# DataLoaders handle batching, shuffling, and loading data in parallel
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # num_workers > 0 for parallel loading
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Store dataloaders and dataset sizes in a dictionary
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}


# --- Model Definition ---

# Load a pre-trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Get the number of features in the last fully connected layer
num_ftrs = model.fc.in_features

# Replace the last fully connected layer with a new one for binary classification (2 classes: Normal, Defect)
# The output layer will have 2 nodes, representing the scores for each class.
# We will use CrossEntropyLoss which expects raw scores (logits).
model.fc = nn.Linear(num_ftrs, 2)

# Move the model to the specified device (GPU or CPU)
model = model.to(DEVICE)

# --- Loss Function and Optimizer ---

# Define the loss function
# CrossEntropyLoss is suitable for multi-class classification (even binary)
# It combines LogSoftmax and NLLLoss internally.
# If you have class imbalance, you can use the 'weight' parameter:
# weights = torch.tensor([weight_for_normal, weight_for_defect], dtype=torch.float).to(DEVICE)
# criterion = nn.CrossEntropyLoss(weight=weights)
criterion = nn.CrossEntropyLoss()


# Define the optimizer
# Adam is a popular choice
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Or using Stochastic Gradient Descent (SGD) with momentum
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# --- Training Function ---

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    # Keep track of the best model state found so far
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Enable gradient calculation only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # Get the predicted class (index with highest score)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print() # Print a newline after each epoch phase

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# --- Training Execution ---

print("Starting training...")
model_ft = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=NUM_EPOCHS)

# --- Save the Model ---

# Create a directory to save the model if it doesn't exist
os.makedirs('models', exist_ok=True)

# Define the path to save the model
model_save_path = 'models/defect_classifier.pth'

# Save the model state dictionary
torch.save(model_ft.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

# --- How to Load the Model Later for Inference ---
# To load the model later for inference:
# loaded_model = models.resnet50(weights=None) # Load architecture without pre-trained weights
# num_ftrs_loaded = loaded_model.fc.in_features
# loaded_model.fc = nn.Linear(num_ftrs_loaded, 2) # Recreate the final layer
# loaded_model.load_state_dict(torch.load(model_save_path)) # Load the saved weights
# loaded_model = loaded_model.to(DEVICE)
# loaded_model.eval() # Set to evaluation mode before inference
