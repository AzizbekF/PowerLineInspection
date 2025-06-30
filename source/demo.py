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

sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")  # .to("mps")  # or "cuda"/"cpu"
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


def overlay_masks_on_image(raw_image, masks, alpha=0.6):
    image_np = np.array(raw_image).astype(np.float32) / 255.0
    overlay = image_np.copy()

    if len(masks.shape) == 4:
        masks = masks.squeeze(0)  # [1, N, H, W] → [N, H, W]

    for mask in masks:
        mask_np = mask.cpu().numpy()
        color = np.random.rand(3)
        mask_3d = np.stack([mask_np] * 3, axis=-1)
        overlay = np.where(mask_3d > 0.5,
                           (1 - alpha) * overlay + alpha * color,
                           overlay)

    overlay = (overlay * 255).astype(np.uint8)

    # 🧼 Remove accidental batch dims
    if overlay.ndim == 4:
        overlay = np.squeeze(overlay, axis=0)

    return Image.fromarray(overlay)


def segment_with_sam(raw_image, input_boxes, device="cpu"):
    """
    raw_image: PIL.Image
    input_boxes: List[List[List[float]]] → [[[x1, y1, x2, y2], ...]]
    """
    # Prepare input
    inputs = sam_processor(raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)
    image_embeddings = sam_model.get_image_embeddings(inputs["pixel_values"])

    if isinstance(image_embeddings, tuple):
        image_embeddings = image_embeddings[0]

    inputs.pop("pixel_values", None)
    inputs["image_embeddings"] = image_embeddings

    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)  # only best mask per box

    # Postprocess
    masks_data = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    masks = masks_data[0].squeeze(0)  # [N, H, W]
    return overlay_masks_on_image(raw_image, masks)


from PIL import Image


def resize_cam_to_box(cam_mask: np.ndarray, box):
    """
    Resize CAM mask to match the bounding box size on the original image.
    Args:
        cam_mask (np.ndarray): CAM mask from Grad-CAM (shape = crop size).
        box (list/tuple): Bounding box in format [x1, y1, x2, y2].

    Returns:
        cam_pil (PIL Image): Resized CAM mask as grayscale image.
        position (tuple): (x1, y1) position where this mask should be pasted.
    """
    x1, y1, x2, y2 = map(int, box)
    width, height = x2 - x1, y2 - y1

    cam_pil = Image.fromarray((cam_mask * 255).astype(np.uint8)).resize(
        (width, height), resample=Image.BILINEAR
    )
    return cam_pil, (x1, y1)


def cam_to_heatmap_rgba(cam_image: Image.Image):
    cam_np = np.array(cam_image.convert("L"))
    heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGBA)
    return Image.fromarray(heatmap)


def apply_mask_on_image(
        highlight_layer: Image.Image,
        cam_image: Image.Image,
        position: tuple,
        alpha=220
):
    """
    Apply resized CAM mask onto highlight layer.
    Args:
        highlight_layer: RGBA Image same size as original.
        cam_image: Grayscale PIL image (resized CAM mask).
        position: (x, y) where to paste the mask.
        color: RGB tuple for highlight color.
        alpha: Transparency (0-255).
    """
    x, y = position

    # Convert grayscale mask to alpha mask
    # cam_alpha = cam_image.convert("L").point(lambda p: p * (alpha / 255))
    cam_alpha = cam_image.convert("L").point(lambda p: (p / 255) ** 0.6 * 255 * (alpha / 255))

    # Create colored RGBA overlay
    # colored_mask = Image.new("RGBA", cam_image.size, color + (0,))
    # colored_mask = Image.new("RGBA", cam_image.size, (255, 255, 0, 0))  # bright yellow
    colored_mask = cam_to_heatmap_rgba(cam_image)
    colored_mask.putalpha(cam_alpha)

    # Paste onto the highlight layer
    highlight_layer.paste(colored_mask, (x, y), colored_mask)


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
    input_boxes = []  # list of tensors
    gradcam_images = []  # list of tensors
    highlight_layer = Image.new("RGBA", original_img.size, (0, 0, 0, 0))

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
                        input_boxes.append([x1, y1, x2, y2])
                        crop = crop_object(original_img, box)
                        model = get_model(id)
                        if id == 0:
                            prob, label, heatmap_overlay, raw_cam = predict_image_for_missing_part_with_gradcam(crop,
                                                                                                                model,
                                                                                                                "features",
                                                                                                                DEVICE,
                                                                                                                0.5)
                        elif id == 3:
                            prob, label, heatmap_overlay, raw_cam = predict_with_gradcam_bird_nest(crop, model,
                                                                                                   "features", DEVICE)
                        else:
                            prob, label, heatmap_overlay, raw_cam = predict_image_with_gradcam(crop, model, DEVICE)
                        print(f"Lable: {label} with prob {prob}")

                        if label != 0:
                            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
                            # draw.text((x1, y1 + 20), class_to_problem[id], fill="red")

                            # Try larger system font
                            try:
                                if os.name == "nt":
                                    font_path = "C:/Windows/Fonts/arial.ttf"
                                else:
                                    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
                                font = ImageFont.truetype(font_path, size=24)
                            except OSError:
                                print("Warning: Custom font not found. Using default font.")
                                font = ImageFont.load_default()

                            if id != 3:
                                text = class_to_problem[id]
                            else:
                                text = class_to_problem[id].get(label)

                            # Scale text box if default font is used (bitmap and small)
                            scale = 2.5 if isinstance(font, ImageFont.ImageFont) else 1.0
                            bbox = font.getbbox(text)
                            text_width = int((bbox[2] - bbox[0]) * scale)
                            text_height = int((bbox[3] - bbox[1]) * scale)

                            # Background box and text
                            bg_box = [x1, y1 + 20, x1 + text_width + 6, y1 + 20 + text_height + 4]
                            draw.rectangle(bg_box, fill="red")
                            draw.text((x1 + 3, y1 + 22), text, fill="black", font=font)

                            detections.append(
                                {"id": id, "prob": prob, "label": label, "type": text, "box": box,
                                 "confidence": confidence})

                            cam_resized, position = resize_cam_to_box(raw_cam, box)
                            apply_mask_on_image(highlight_layer, cam_resized, position, alpha=220)

                            gradcam_images.append({"image": heatmap_overlay, "title": class_name, "defect": text})

                # if gradcam_images:
                #     show_images_horizontal(gradcam_images)

                img_object_detection = Image.fromarray(result.plot().astype('uint8'))
                final_highlighted_image = Image.alpha_composite(original_img.convert("RGBA"), highlight_layer)

                print("SIZE:")
                print(len([d["box"] for d in detections]))
                print(len(input_boxes))
                print(input_boxes)
                print(f"Detections: {detections}")

                result = [(img_object_detection, "Object Detections")]
                if len(input_boxes) > 0:
                    segmented = segment_with_sam(original_img, [input_boxes])
                    result += [(segmented, "Object Segmentation")]

                if len(detections) > 0:
                    result += [
                        (img_defections, "Defect Detection"),
                        (final_highlighted_image, "Highlighted Image")
                    ]

                return result

    return [(original_img, "No detections")]


import gradio as gr
from PIL import Image


def gradio_wrapper(image):
    return run_full_pipeline(image)


demo = gr.Blocks(css="""
/* Align columns top */
.gr-row {
    align-items: flex-start !important;
}

/* Right column full height */
.right-column {
    height: 92vh;
    display: flex;
    flex-direction: column;
}

/* Gallery expands vertically */
.right-column .gr-gallery {
    flex-grow: 1;
    height: 100% !important;
}

/* Button styling */
.gr-button {
    font-size: 1.2rem !important;
    font-weight: 600;
}
""")

with demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🔍 Image Defect Detection — Powerline Towers")
        with gr.Column(scale=2):
            gr.Markdown("## 🖼️ Results")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_button = gr.Button("🚀 Run Detection")

        with gr.Column(scale=2, elem_classes=["right-column"]):
            gallery_output = gr.Gallery(
                label="Detection Results", columns=[2], height="auto", show_label=True
            )

    submit_button.click(
        fn=gradio_wrapper,
        inputs=image_input,
        outputs=gallery_output,
    )

load_models()
demo.launch()
