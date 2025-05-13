import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class DefectClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Use a pretrained model as feature extractor
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add asset-specific classification heads
        self.asset_classifiers = nn.ModuleDict({
            'Glass Insulator': nn.Linear(2048, num_classes),
            'Lightning Rod Suspension': nn.Linear(2048, num_classes),
            'Polymer Insulator Upper Shackle': nn.Linear(2048, num_classes),
            'Vari-grip': nn.Linear(2048, num_classes),
            'Yoke Suspension': nn.Linear(2048, num_classes),
            # Add more asset types as needed
        })

    def forward(self, x, asset_type):
        # Extract features
        features = self.backbone(x)
        features = torch.flatten(features, 1)

        # Use the appropriate classifier head based on asset type
        if asset_type in self.asset_classifiers:
            return self.asset_classifiers[asset_type](features)
        else:
            # Default classifier for unknown assets
            return self.asset_classifiers[list(self.asset_classifiers.keys())[0]](features)


# Define the data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def process_detection(image, detection, defect_model):
    # Extract the detected object from the image
    x1, y1, x2, y2 = detection["bbox"]
    object_crop = image.crop((x1, y1, x2, y2))

    # Preprocess the cropped image
    input_tensor = transform(object_crop).unsqueeze(0)

    # Get the asset type from detection
    asset_type = detection["class"]

    # Predict defect status
    with torch.no_grad():
        defect_pred = defect_model(input_tensor, asset_type)
        defect_status = torch.argmax(defect_pred, dim=1).item()
        confidence = torch.softmax(defect_pred, dim=1)[0][defect_status].item()

    return {
        "bbox": detection["bbox"],
        "class": asset_type,
        "defect_status": "Fault" if defect_status == 1 else "Normal",
        "confidence": confidence
    }


def inspect_powerline_image(image_path, detector, defect_model):
    # Load image
    image = Image.open(image_path)

    # Run object detection
    detections = detector(image)

    # Process each detection for defects
    results = []
    for detection in detections:
        defect_result = process_detection(image, detection, defect_model)
        results.append(defect_result)

    return results