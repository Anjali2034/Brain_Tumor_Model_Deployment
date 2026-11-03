"""Brain tumour segmentation demo (U-Net, LGG MRI).

Upload a brain MRI slice, or pick a sample, and the model outlines the
tumour region as a mask and an overlay.
"""

import os

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODEL_PATH = "best_model.pth"
IMG_SIZE = 256
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


# ---------------------------------------------------------------- model ----
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ------------------------------------------------------------ inference ----
def to_rgb(image: np.ndarray) -> np.ndarray:
    """Accept grayscale, RGB or RGBA input and return 3-channel RGB uint8."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return image.astype(np.uint8)


def segment(image: np.ndarray):
    if image is None:
        raise gr.Error("Upload an MRI slice or pick one of the samples first.")

    rgb = to_rgb(image)
    h, w = rgb.shape[:2]

    # The model was trained on images read with cv2.imread (BGR order),
    # while Gradio passes RGB, so flip the channels before inference.
    bgr = rgb[:, :, ::-1]
    x = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)) / 255.0
    x = torch.tensor(np.transpose(x, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = model(x).squeeze().cpu().numpy()

    mask = (prob > THRESHOLD).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Red overlay plus an outline, so the region is easy to see.
    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[:, :, 0] = 255
    region = mask > 0
    overlay[region] = (0.55 * rgb[region] + 0.45 * red[region]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 60, 60), 2)

    area = region.mean() * 100
    if area < 0.05:
        verdict = "### No tumour region detected\nThe model did not mark any area on this slice."
    else:
        verdict = (
            f"### Tumour region detected\n"
            f"**{area:.2f}%** of the slice is marked, across **{len(contours)}** region(s)."
        )
    return Image.fromarray(overlay), Image.fromarray(mask), verdict


# ------------------------------------------------------------------- UI ----
EXAMPLES = [
    os.path.join(EXAMPLES_DIR, f)
    for f in ["sample_tumour.png", "sample_no_tumour.png"]
    if os.path.exists(os.path.join(EXAMPLES_DIR, f))
]

INTRO = """
# Brain Tumour Segmentation
A **U-Net** model that outlines low-grade glioma (brain tumour) regions on MRI slices.
Upload a brain MRI slice, or **click a sample below** to see it work in one click.
"""

HOW = """
**How it works**
1. The slice is resized to 256 x 256 and normalised.
2. A U-Net (4 down and 4 up blocks with skip connections) predicts a tumour probability for every pixel.
3. Pixels above 0.5 form the mask, drawn in red on the original scan.

**Trained on** the LGG MRI Segmentation dataset: 3,929 slices from 110 patients (TCGA / TCIA).
**Test set (590 slices):** mean Dice 0.84, mean IoU 0.81.

Code: [GitHub](https://github.com/Anjali2034/brain_tumor_model_deployment)
"""

DISCLAIMER = "Research demo only. Not a medical device and not for diagnosis."

with gr.Blocks(title="Brain Tumour Segmentation", theme=gr.themes.Soft(primary_hue="red")) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="MRI slice (PNG, JPG or TIF)", height=320)
            with gr.Row():
                clear = gr.ClearButton(value="Clear")
                run = gr.Button("Segment tumour", variant="primary")
            if EXAMPLES:
                gr.Examples(examples=EXAMPLES, inputs=inp, label="Try a sample scan")
        with gr.Column(scale=1):
            verdict = gr.Markdown("### Result\nYour result will appear here.")
            out_overlay = gr.Image(type="pil", label="Overlay: tumour region in red", height=320)
            out_mask = gr.Image(type="pil", label="Predicted mask", height=200)
    with gr.Accordion("How it works", open=False):
        gr.Markdown(HOW)
    gr.Markdown(f"<sub>{DISCLAIMER}</sub>")

    clear.add([inp, out_overlay, out_mask])
    run.click(segment, inputs=inp, outputs=[out_overlay, out_mask, verdict])
    inp.change(lambda img: segment(img) if img is not None else (None, None, "### Result\nYour result will appear here."),
               inputs=inp, outputs=[out_overlay, out_mask, verdict])

if __name__ == "__main__":
    demo.launch()
