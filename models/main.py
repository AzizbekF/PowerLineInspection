#!/usr/bin/env python3
# example_usage.py - Example of using the powerline defect detection system

import os
from powerline_defect import train_and_test_powerline_defect_detector


def main():
    """
    Example of how to use the powerline defect detection system with real data
    """
    # Define paths
    csv_path = "./dataset/labels_with_status_code.csv"
    rtdetr_model_path = "models/rt_detr_model.pth"
    output_dir = "results/defect_detection"
    test_images_dir = "data/test_images"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Train and test the model
    print("Starting training and testing pipeline...")
    model_data, test_accuracy = train_and_test_powerline_defect_detector(
        csv_path=csv_path,
        rtdetr_model_path=rtdetr_model_path,
        output_dir=output_dir,
        test_split=0.2,  # 20% of data for testing
        val_split=0.1,  # 10% of remaining data for validation
        num_epochs=30,  # Train for 30 epochs
        batch_size=32,  # Batch size of 32
        learning_rate=0.001,  # Initial learning rate
        backbone="efficientnet_b0",  # Use EfficientNet B0 backbone
        test_images_dir=test_images_dir
    )

    # Access the trained model and metadata
    state_model = model_data['state_model']
    category_to_idx = model_data['category_to_idx']
    status_to_idx = model_data['status_to_idx']

    print(f"\nTraining and testing completed!")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Model and results saved to: {output_dir}")

    # You can now use the trained model for inference
    print("\nModel can now be used for inference.")
    print("Categories in the model:")
    for category, idx in category_to_idx.items():
        print(f"  - {category} (index: {idx})")

    print("\nStates in the model:")
    for status, idx in status_to_idx.items():
        print(f"  - {status} (index: {idx})")


if __name__ == "__main__":
    main()