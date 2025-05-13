class PowerlineDefectDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, asset_types, labels, transform=None):
        self.image_paths = image_paths
        self.asset_types = asset_types
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        asset_type = self.asset_types[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, asset_type, label


def create_balanced_sampler(dataset):
    # Count samples per class
    class_counts = {}
    for _, _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Calculate weights
    weights = []
    for _, _, label in dataset:
        weight = 1.0 / class_counts[label]
        weights.append(weight)

    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler