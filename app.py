"""MedSkel AI demo: upload a retina photograph, see the predicted vessels.

    .venv\\Scripts\\python.exe app.py        then open http://127.0.0.1:7860

Everything runs on this computer; no image is sent anywhere.
Research prototype for a university class. Not a medical device.

The same file runs on a Hugging Face Space (CPU): there the weights sit in a
folder `model/` next to this file (see space/README.md).
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from medskel_ai.data import resize_image_bilinear, resize_mask_nearest
from medskel_ai.predict import load_model, make_overlay, predict_probability

HERE = Path(__file__).parent
TRAIN_SIZE = 2048  # FIVES photographs are 2048 x 2048; the model learned vessels at that scale
VESSEL_COLOR = (0, 255, 120)


def find_model_dir() -> Path:
    """1) MEDSKEL_MODEL_DIR if set, 2) ./model (Hugging Face Space), 3) the final local training run."""
    candidates = [os.environ.get("MEDSKEL_MODEL_DIR"), HERE / "model", HERE / "runs" / "final", HERE / "runs" / "unet_fives"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "best.pt").exists():
            return Path(candidate)
    raise SystemExit("No trained model found. Train one first or set MEDSKEL_MODEL_DIR.")


MODEL_DIR = find_model_dir()
ON_GPU = torch.cuda.is_available() and os.environ.get("MEDSKEL_DEVICE", "") != "cpu"
DEVICE = "cuda" if ON_GPU else "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 2)))
# Two thresholds were chosen on FIVES validation images: one for a single pass, one for 4 mirrored passes.
model, device, THRESHOLD_PLAIN = load_model(MODEL_DIR / "best.pt", device=DEVICE, flips=False)
_, _, THRESHOLD_FLIPS = load_model(MODEL_DIR / "best.pt", device=DEVICE, flips=True)
ABOUT_FILE = HERE / "ABOUT.md"
ABOUT = ABOUT_FILE.read_text(encoding="utf-8") if ABOUT_FILE.exists() else "Results table not found."


def example_images() -> list[str]:
    """Example photos: ./examples if it exists, else a few local FIVES test images (never uploaded anywhere)."""
    folder = HERE / "examples"
    if folder.is_dir():
        return [str(p) for p in sorted(folder.glob("*.png"))][:4]
    fives = Path(os.environ.get("FIVES_DIR", r"C:\data\FIVES")) / "test" / "Original"
    names = ["12_A.png", "60_D.png", "110_G.png", "160_N.png"]
    return [str(fives / n) for n in names if (fives / n).exists()]


def segment(image: np.ndarray | None, threshold: float, flips: bool, match_scale: bool):
    if image is None:
        raise gr.Error("Upload a retina photograph first.")
    image = np.ascontiguousarray(image[..., :3])
    height, width = image.shape[:2]
    work = image
    if match_scale and max(height, width) != TRAIN_SIZE:
        # Bring the photo to the scale the model was trained on (bilinear is fine for photos).
        scale = TRAIN_SIZE / max(height, width)
        work = resize_image_bilinear(image, round(width * scale), round(height * scale))
    prob = predict_probability(model, work, device, flips=flips, batch_size=4 if ON_GPU else 1)  # tiles + stitching
    mask = (prob >= threshold).astype(np.uint8)
    if work is not image:
        mask = resize_mask_nearest(mask, width, height)  # masks: nearest neighbour only
    overlay = make_overlay(image, mask, color=VESSEL_COLOR, strength=0.85)
    info = (
        f"**Vessel pixels: {mask.mean() * 100:.2f}%** of the image · threshold {threshold:.2f} · "
        f"{'4 mirrored passes' if flips else 'single pass'} · running on {device.type.upper()}"
    )
    return (image, overlay), mask * 255, info


def default_threshold(flips: bool) -> float:
    return THRESHOLD_FLIPS if flips else THRESHOLD_PLAIN


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#title h1 {font-size: 2.0rem; margin-bottom: 0.1rem;}
#title p {opacity: 0.8; margin-top: 0;}
#note {font-size: 0.85rem; opacity: 0.75;}
"""
THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")

with gr.Blocks(title="MedSkel AI - retinal vessel segmentation") as demo:
    gr.Markdown(
        "# MedSkel AI\nRetinal blood-vessel segmentation with a small U-Net trained from scratch on the FIVES dataset.",
        elem_id="title",
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300):
                    photo = gr.Image(type="numpy", label="Retina photograph", height=330)
                    threshold = gr.Slider(0.05, 0.95, value=default_threshold(ON_GPU), step=0.05, label="Threshold", info="A pixel is called vessel if its probability is at least this value. The default was chosen on validation images.")
                    flips = gr.Checkbox(value=ON_GPU, label="Higher quality: average 4 mirrored passes", info="About 4x slower. Recommended on a GPU.")
                    match_scale = gr.Checkbox(value=True, label="Resize to the training scale (longest side 2048 px)")
                    run = gr.Button("Find the vessels", variant="primary")
                    examples = example_images()
                    if examples:
                        gr.Examples(examples=examples, inputs=photo, label="Example photographs (FIVES test images)", cache_examples=False)
                        gr.Markdown(
                            "Examples: FIVES dataset, Jin et al., *Scientific Data* 2022, "
                            "[doi:10.1038/s41597-022-01564-3](https://doi.org/10.1038/s41597-022-01564-3), licence CC BY 4.0.",
                            elem_id="note",
                        )
                with gr.Column(scale=7, min_width=420):
                    slider = gr.ImageSlider(label="Drag the handle: photograph  |  predicted vessels", height=520)
                    info = gr.Markdown()
                    with gr.Accordion("Predicted mask (black and white)", open=False):
                        mask_out = gr.Image(label="Predicted mask", height=420)
            gr.Markdown("Research prototype for a university class. **Not a medical device. No diagnosis.**", elem_id="note")
        with gr.Tab("About"):
            gr.Markdown(ABOUT)

    run.click(segment, inputs=[photo, threshold, flips, match_scale], outputs=[slider, mask_out, info])
    flips.change(default_threshold, inputs=flips, outputs=threshold)

if __name__ == "__main__":
    fives_dir = os.environ.get("FIVES_DIR", r"C:\data\FIVES")
    demo.launch(theme=THEME, css=CSS, allowed_paths=[fives_dir] if Path(fives_dir).exists() else None)
