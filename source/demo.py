from utils.util import *
from ultralytics import RTDETR
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


detector = RTDETR("../models/object_detection.pt")

classifier = get_pretrained_resnet(num_classes=1, pretrained=False)
classifier.load_state_dict(torch.load("../models/defect_detection_model.pth"))
classifier.to(DEVICE)

detectable_classes = {7:0, 5:1, 11:2, 1:4}
class_to_model = {0 : "defect_detection_glass_model.pth", 1 : "defect_detection_lighting_model.pth", 2: "defect_detection_polymer_model.pth", 4: "defect_detection_yoke_model.pth"}
class_to_problem = {0 : "Missing cap", 1 : "Rust", 2: "Rust", 4: "Rust"}

models = {}
def load_models():
    for i, v in class_to_model.items():
        model_path = os.path.join("../models", v)
        model = get_pretrained_resnet(num_classes=1, pretrained=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models[i] = model

def get_model(class_id):
    return models.get(class_id, None)

def run_full_pipeline(img, image_path: str | Path = "", pad: int = 0):
    """
    • Detect parts with REDETR
    • Crop each part
    • Run defect classifier on the crops
    • Return processed images (object and defect images)
    """
    from PIL import ImageDraw

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
                        prob, label = predict_single(crop, model, DEVICE)
                        print(f"Lable: {label} with prob {prob}")
                        if label != 0:
                            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=1)
                            draw.text((x1, y1 - 12), class_to_problem[id], fill="red")
                            detections.append({"id": id, "prob": prob, "label": label, "type": class_to_problem[id], "box": box, "confidence": confidence})

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