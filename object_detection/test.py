from ultralytics import RTDETR
import os
import random
from PIL import Image

# Load your trained model
model = RTDETR("model/object_detection.pt")  # Ensure correct path

# Path to your directory of random images
image_dir = "../data/InsPLAD-det/val/images"

# Get a list of all image files
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

num_test_images = 5  # Specify how many random images you want to test

for _ in range(num_test_images):
    random_image_path = random.choice(image_files)
    print(f"\n--- Processing image: {os.path.basename(random_image_path)} ---")
    results = model.predict(random_image_path)

    for result in results:
        print(f"  Type of result.boxes: {type(result.boxes)}")
        print(f"  Value of result.boxes: {result.boxes}")
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            confidences = boxes.conf
            cls_indices = boxes.cls

            print(f"  Type of boxes: {type(boxes)}")
            print(f"  Type of confidences: {type(confidences)}")
            print(f"  Type of cls_indices: {type(cls_indices)}")

            for box, confidence, class_id in zip(boxes.xyxy, confidences, cls_indices):
                x1, y1, x2, y2 = box.tolist()
                confidence_value = confidence.item()
                class_name = result.names[int(class_id)]
                print(f"    Detected {class_name} with confidence: {confidence_value:.2f} at ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})")

            annotated_image_np = result.plot()  # Get the annotated image as a NumPy array
            annotated_image_pil = Image.fromarray(annotated_image_np.astype('uint8')) # Convert to PIL Image
            save_path = f"no_train_detected_{os.path.basename(random_image_path)}"
            annotated_image_pil.save(save_path)  # Now you can use .save()
            print(f"  Saved annotated image to: {save_path}")
        else:
            print("  No objects detected in this image.")