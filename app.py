"""Local demo: upload a retina photograph, see the predicted vessels in colour.

    .venv\\Scripts\\python.exe app.py

Then open http://127.0.0.1:7860 in a browser. Everything runs on this
computer; no image is sent anywhere. This is a class demo, not a medical tool.
"""

import gradio as gr
import numpy as np

from medskel_ai.data import resize_image_bilinear, resize_mask_nearest
from medskel_ai.predict import DEFAULT_CHECKPOINT, load_model, make_overlay, predict_probability

TRAIN_SIZE = 2048  # FIVES photographs are 2048 x 2048; the model learned vessels at that scale

model, device, default_threshold = load_model(DEFAULT_CHECKPOINT)


def segment(image: np.ndarray, threshold: float, match_scale: bool):
    if image is None:
        return None, None, "Upload an image first."
    image = image[..., :3]
    height, width = image.shape[:2]
    work = image
    if match_scale and max(height, width) != TRAIN_SIZE:
        # Bring the photo to the scale the model was trained on (bilinear is fine for photos).
        scale = TRAIN_SIZE / max(height, width)
        work = resize_image_bilinear(image, round(width * scale), round(height * scale))
    prob = predict_probability(model, work, device)  # tiles + stitching
    mask = (prob >= threshold).astype(np.uint8)
    if work is not image:
        mask = resize_mask_nearest(mask, width, height)  # masks: nearest neighbour only
    overlay = make_overlay(image, mask, color=(0, 255, 0))
    text = f"Vessel pixels: {mask.mean() * 100:.2f}% of the image | threshold {threshold:.2f} | device: {device}"
    return overlay, mask * 255, text


demo = gr.Interface(
    fn=segment,
    inputs=[
        gr.Image(type="numpy", label="Retina photograph"),
        gr.Slider(0.05, 0.95, value=default_threshold, step=0.05, label="Threshold (default = chosen on validation images)"),
        gr.Checkbox(value=True, label="Resize to the training scale (longest side 2048 px) before predicting"),
    ],
    outputs=[
        gr.Image(label="Overlay: predicted vessels in green"),
        gr.Image(label="Predicted mask"),
        gr.Textbox(label="Info"),
    ],
    title="MedSkel AI - retinal vessel segmentation (U-Net trained on FIVES)",
    description="University class demo. Research prototype only: not a medical device, no diagnosis.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
