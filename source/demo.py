from utils.util import *
from ultralytics import RTDETR
from pathlib import Path
from torchvision import models as torchvision_models
from PIL import ImageFont, ImageDraw
import numpy as np
from transformers import SamProcessor, SamModel

detector = RTDETR("../models/object_detection.pt")

detectable_classes = {7: 0, 5: 1, 11: 2, 9: 3, 1: 4}
class_to_model = {0: "efficientnet_b3_missing_cup.pt", 1: "rust_detector_resnet_1.pt", 2: "rust_detector_resnet_2.pt",
                  3: "efficientnet_b3_varigrip.pt", 4: "rust_detector_resnet_4.pt"}
class_to_problem = {0: "Missing cap", 1: "Rust", 2: "Rust", 3: {1: "Rust", 2: "Bird nest"}, 4: "Rust"}

models = {}

sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")#.to("mps")  # or "cuda"/"cpu"
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

def load_models():
    for i, v in class_to_model.items():
        model_path = os.path.join("../models", v)

        if i == 0:
            model = torchvision_models.efficientnet_b3(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        elif i == 3:
            model = torchvision_models.efficientnet_b3(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        else:
            model = torchvision_models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        models[i] = model

def get_model(class_id):
    return models.get(class_id, None)

def show_masks_on_image(raw_image, masks, scores, alpha=0.8):
    image_np = np.array(raw_image).astype(np.float32) / 255.0
    if len(masks.shape) == 4:
        masks = masks.squeeze(0)
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    nb_predictions = scores.shape[0]
    overlay = image_np.copy()

    for i in range(nb_predictions):
        mask = masks[i].cpu().detach().numpy()
        color = np.random.rand(3)
        mask_3d = np.stack([mask] * 3, axis=-1)
        overlay = np.where(mask_3d > 0.5,
                           (1 - alpha) * overlay + alpha * color,
                           overlay)

    overlay = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay)

def run_full_pipeline(img, image_path: str | Path = "", pad: int = 0):
    """
    • Detect parts with REDETR
    • Crop each part
    • Run defect classifier on the crops
    • Return processed images (object and defect images)
    """

    original_img = img.convert("RGB") if img else Image.open(image_path).convert("RGB")
    img_defections = original_img.copy()
    draw = ImageDraw.Draw(img_defections)

    results = detector.predict(original_img)
    detections = []
    if results:
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                confidences = boxes.conf
                cls_indices = boxes.cls

                for box, confidence, class_id in zip(boxes.xyxy, confidences, cls_indices):
                    x1, y1, x2, y2 = box.tolist()
                    c_id = int(class_id)
                    class_name = result.names[c_id]
                    print("Class name: ", class_name)
                    print("Class ID: ", c_id)
                    id = detectable_classes.get(c_id)
                    print("Class ID modified: ", id)
                    if id is not None:
                        crop = crop_object(original_img, box)
                        model = get_model(id)
                        if id == 0:
                            prob, label = predict_image_for_missing_part(crop, model, DEVICE, 0.5)
                        elif id == 3:
                            prob, label = predict_single_image_bird_nest(crop, model, DEVICE)
                        else:
                            prob, label = predict_image(crop, model, DEVICE)
                        print(f"Lable: {label} with prob {prob}")

                        if label != 0:
                            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
                            # draw.text((x1, y1 + 20), class_to_problem[id], fill="red")

                            font = ImageFont.load_default()

                            if id != 3:
                                text = class_to_problem[id]
                            else:
                                text = class_to_problem[id].get(label)

                            # New way to calculate text size
                            bbox = font.getbbox(text)  # returns (left, top, right, bottom)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]

                            # Background box and text
                            bg_box = [x1, y1 + 20, x1 + text_width + 6, y1 + 20 + text_height + 4]
                            draw.rectangle(bg_box, fill="red")
                            draw.text((x1 + 3, y1 + 22), text, fill="black", font=font)

                            detections.append(
                                {"id": id, "prob": prob, "label": label, "type": text, "box": box,
                                 "confidence": confidence})

                img_object_detection = Image.fromarray(result.plot().astype('uint8'))
                return img_object_detection, img_defections if detections else None

    return original_img, None  # Fallback if no detections


import gradio as gr
from PIL import Image


def gradio_wrapper(image):
    obj_img, defect_img = run_full_pipeline(image)
    return obj_img, defect_img


with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Image Defect Detection")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        submit_button = gr.Button("Run Detection")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🟦 All Detected Objects")
            object_output = gr.Image(label="Object Detection Result")

        with gr.Column():
            gr.Markdown("### 🟥 Defects Only (if any)")
            defect_output = gr.Image(label="Defect Detection Result")

    submit_button.click(fn=gradio_wrapper,
                        inputs=image_input,
                        outputs=[object_output, defect_output])

load_models()
demo.launch()
