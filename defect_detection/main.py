import torch.optim as optim
from utils.util import *

# --- Configuration & Hyperparameters ---
CSV_FILE_PATH = '../dataset/image_labels_with_code.csv'  # Replace with your CSV file path
IMAGE_DIR = '../data/InsPLAD-fault/defect_supervised'  # Replace with the base directory of your images

IMG_HEIGHT = 224  # Image height for ResNet
IMG_WIDTH = 224  # Image width for ResNet
BATCH_SIZE = 16
NUM_WORKERS = 4  # Number of worker processes for DataLoader
LEARNING_RATE = 1e-5
NUM_EPOCHS = 50  # Max number of epochs (early stopping will be used)
EARLY_STOPPING_PATIENCE = 5
BEST_MODEL_PATH = '../models/defect_detection_yoke_model.pth'
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # --- Setup ---
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Warning: {CSV_FILE_PATH} not found.")

    # --- Data Loading and Preparation ---
    df = load_data_from_csv(CSV_FILE_PATH)
    df = df[df['category_code'] == 4]
    train_df, val_df = split_data(df, stratify_col='status')  # Ensure 'status' is the correct column

    # Define image transformations
    # For ResNet, normalization values are typically from ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # Add more augmentations if needed (e.g., ColorJitter)
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_loader, val_loader = get_data_loaders(
        train_df, val_df, IMAGE_DIR, train_transform, val_transform, BATCH_SIZE, NUM_WORKERS
    )

    # --- Model Initialization ---
    # num_classes=1 for binary classification with BCEWithLogitsLoss
    model = get_pretrained_resnet(num_classes=1, pretrained=True, freeze_base=False)

    # --- Loss Function and Optimizer ---
    # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
    # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss.

    # n_defect = len(train_df[train_df['status'] == 0])  # positives
    # n_good = len(train_df) - n_defect
    # pos_w = torch.tensor([n_good / n_defect]).to(DEVICE)
    #
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer - Adam is a common choice.
    # If only fine-tuning the last layer, only pass its parameters.
    if any(p.requires_grad for p in model.parameters()):  # Check if any parameters are trainable
        if model.fc.weight.requires_grad:  # If only fc layer is trainable
            print("Optimizing only the final layer.")
            optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
        else:  # Should not happen if freeze_base=True and fc is replaced, but as a fallback
            print("Optimizing all trainable parameters (unexpected for freeze_base=True).")
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    else:
        print("Error: No parameters to optimize. Check model freezing logic.")
        exit()

    # --- Train the Model ---
    # Check if DataLoaders are empty
    if len(train_loader) == 0 or len(val_loader) == 0:
        print("Error: One or both DataLoaders are empty. This might be due to:")
        print("1. No valid image paths found from the CSV in the specified IMAGE_DIR.")
        print("2. BATCH_SIZE being larger than the dataset size.")
        print("Please check your data, paths, and BATCH_SIZE.")
        exit()

    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, DEVICE,
        NUM_EPOCHS, EARLY_STOPPING_PATIENCE, BEST_MODEL_PATH
    )

    # --- Optional: Unfreeze some layers and fine-tune further ---
    # If you want to fine-tune more layers after initial training:
    # print("\n--- Starting Fine-tuning Phase (Unfreezing more layers) ---")
    # for param in model.parameters(): # Unfreeze all or some layers
    #     param.requires_grad = True
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE / 10) # Use a smaller LR
    # fine_tuned_model = train_model(
    #     model, train_loader, val_loader, criterion, optimizer, DEVICE,
    #     num_epochs=10, # Fewer epochs for fine-tuning
    #     early_stopping_patience=3,
    #     best_model_path='fine_tuned_best_model.pth'
    # )

    # --- How to load the best model for inference/evaluation later ---
    print(f"\nTo load the best model for inference:")
    print(f"model_inf = get_pretrained_resnet(num_classes=1, pretrained=False) # Or your specific architecture")
    print(f"model_inf.load_state_dict(torch.load('{BEST_MODEL_PATH}', map_location=DEVICE))")
    print(f"model_inf.to(DEVICE)")
    print(f"model_inf.eval()")

    print("\nScript finished.")

