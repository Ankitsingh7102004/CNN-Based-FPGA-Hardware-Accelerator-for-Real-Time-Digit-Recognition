"""
Phone Photo Digit Classifier
------------------------------
Takes a photo of a handwritten digit (e.g. from your phone), preprocesses
it to match the MNIST format the model was trained on, and classifies it
using the QAT_int8 quantized model (same math the FPGA will eventually run).

Preprocessing pipeline (mirrors how MNIST itself was built):
  1. Grayscale
  2. Auto-detect polarity (white paper + dark pen vs. MNIST's white-on-black)
     and invert if needed
  3. Threshold to find the digit's bounding box, crop to it
  4. Resize the digit to fit a 20x20 box (preserving aspect ratio)
  5. Paste onto a 28x28 canvas
  6. Re-center using center-of-mass (this is what real MNIST preprocessing
     does — it matters a lot for accuracy)
  7. Normalize with MNIST's mean/std

Usage:
    python phone_test.py path/to/photo.jpg
    python phone_test.py                      # scans ./phone_images/ folder

Setup:
    pip install pillow scipy
    mkdir phone_images   (drop your phone photos here if not passing a path)
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")
STEP2_DIR = "results/step2_quantization"
RESULTS_DIR = "results/step6_realworld"
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG_TO_USE = "QAT_int8"
WEIGHT_BITS, ACT_BITS = 8, 8
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


# ---------------------------------------------------------------------------
# Model (must exactly match train.py / quantize.py)
# ---------------------------------------------------------------------------
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def fake_quantize(x, scale, num_bits):
    qmax = 2 ** (num_bits - 1) - 1
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


class FakeQuantConv2d(nn.Module):
    def __init__(self, conv, weight_bits, act_bits):
        super().__init__()
        self.conv = conv
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.act_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x):
        w_scale = self.conv.weight.abs().max() / (2 ** (self.weight_bits - 1) - 1)
        w_scale = torch.clamp(w_scale, min=1e-8)
        q_weight = fake_quantize(self.conv.weight, w_scale, self.weight_bits)
        q_input = fake_quantize(x, self.act_scale, self.act_bits)
        return nn.functional.conv2d(q_input, q_weight, self.conv.bias,
                                     self.conv.stride, self.conv.padding)


class FakeQuantLinear(nn.Module):
    def __init__(self, linear, weight_bits, act_bits):
        super().__init__()
        self.linear = linear
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.act_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x):
        w_scale = self.linear.weight.abs().max() / (2 ** (self.weight_bits - 1) - 1)
        w_scale = torch.clamp(w_scale, min=1e-8)
        q_weight = fake_quantize(self.linear.weight, w_scale, self.weight_bits)
        q_input = fake_quantize(x, self.act_scale, self.act_bits)
        return nn.functional.linear(q_input, q_weight, self.linear.bias)


def build_quantized_model(weight_bits, act_bits):
    model = MnistCNN()
    model.conv1 = FakeQuantConv2d(model.conv1, weight_bits, act_bits)
    model.conv2 = FakeQuantConv2d(model.conv2, weight_bits, act_bits)
    model.fc1 = FakeQuantLinear(model.fc1, weight_bits, act_bits)
    model.fc2 = FakeQuantLinear(model.fc2, weight_bits, act_bits)
    return model


# ---------------------------------------------------------------------------
# Preprocessing: raw photo -> MNIST-style 28x28 tensor
# ---------------------------------------------------------------------------
def otsu_threshold(image):
    """Automatic threshold selection — adapts to each photo's lighting,
    instead of relying on one fixed brightness cutoff for every image."""
    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = image.size
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, weight_bg, best_var, best_t = 0.0, 0.0, 0.0, 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, t
    return best_t


def preprocess_photo(path, threshold=None, blur_sigma=0.6):
    img = Image.open(path).convert("L")  # grayscale
    arr = np.array(img).astype(np.float32)

    # 1. Auto-detect polarity: MNIST is white digit on black background.
    #    A phone photo of pen-on-paper is usually the opposite.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:  # bright border -> white paper background
        arr = 255.0 - arr

    # 2. Auto-threshold (Otsu) instead of one fixed value — adapts per photo
    if threshold is None:
        threshold = otsu_threshold(arr)
    mask = arr > threshold
    if not mask.any():
        raise ValueError(
            f"No digit detected in {path} — try better lighting/contrast."
        )

    # 3. Zero out everything that is NOT the digit stroke (this is the key
    #    fix: a plain bounding-box crop keeps paper texture/shadow as
    #    mid-gray background, which confuses the model badly)
    arr_clean = np.where(mask, arr, 0.0)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    rmin, rmax = rows[0], rows[-1]
    cmin, cmax = cols[0], cols[-1]
    digit = arr_clean[rmin:rmax + 1, cmin:cmax + 1]

    # 4. Contrast-stretch the stroke pixels to use the full 0-255 range
    #    (phone photo strokes are often a dim 60-180 range, not crisp
    #    black/white like MNIST's clean strokes)
    nz = digit > 0
    if nz.any():
        vmin, vmax = digit[nz].min(), digit[nz].max()
        digit = digit.copy()
        digit[nz] = (digit[nz] - vmin) / (vmax - vmin + 1e-8) * 255.0

    # 5. Resize digit to fit a 20x20 box, preserving aspect ratio
    h, w = digit.shape
    if h >= w:
        new_h = 20
        new_w = max(1, round(w * (20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, round(h * (20.0 / w)))
    digit_img = Image.fromarray(digit.astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    digit_resized = np.array(digit_img).astype(np.float32)

    # 6. Paste onto a 28x28 canvas, centered
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit_resized

    # 7. Re-center using center-of-mass (matches real MNIST preprocessing)
    cy, cx = ndimage.center_of_mass(canvas)
    if not (np.isnan(cy) or np.isnan(cx)):
        canvas = ndimage.shift(canvas, shift=(14 - cy, 14 - cx), cval=0)

    # 8. Slight blur — MNIST strokes are anti-aliased, not crisp binary
    #    lines, so a touch of blur matches the training distribution better
    canvas = ndimage.gaussian_filter(canvas, sigma=blur_sigma)

    # 9. Normalize
    normalized = canvas / 255.0
    normalized = (normalized - MNIST_MEAN) / MNIST_STD

    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, canvas, np.array(img)


def classify(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
    pred = int(probs.argmax())
    top3_idx = torch.topk(probs, 3).indices.tolist()
    top3 = [(i, float(probs[i])) for i in top3_idx]
    return pred, float(probs[pred]), top3


def process_and_show(path, model):
    tensor, preprocessed, original = preprocess_photo(path)
    pred, confidence, top3 = classify(model, tensor)

    print(f"\n{os.path.basename(path)}")
    print(f"  Predicted digit: {pred}  (confidence {confidence*100:.1f}%)")
    print(f"  Top-3: " + ", ".join(f"{d}: {p*100:.1f}%" for d, p in top3))

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original photo")
    axes[0].axis("off")
    axes[1].imshow(preprocessed, cmap="gray")
    axes[1].set_title(f"Preprocessed (28x28)\nPrediction: {pred} ({confidence*100:.1f}%)")
    axes[1].axis("off")
    plt.tight_layout()

    out_name = os.path.splitext(os.path.basename(path))[0]
    save_path = os.path.join(RESULTS_DIR, f"{out_name}_result.png")
    plt.savefig(save_path, dpi=150)
    plt.show()

    return pred, confidence


if __name__ == "__main__":
    state_path = os.path.join(STEP2_DIR, f"{CONFIG_TO_USE}_state.pth")
    model = build_quantized_model(WEIGHT_BITS, ACT_BITS)
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    model.eval()

    if len(sys.argv) > 1:
        image_paths = [sys.argv[1]]
    else:
        image_paths = sorted(glob.glob("phone_images/*.jpg") +
                              glob.glob("phone_images/*.jpeg") +
                              glob.glob("phone_images/*.png"))
        if not image_paths:
            print("No image path given, and no images found in ./phone_images/.")
            print("Either run: python phone_test.py path/to/photo.jpg")
            print("Or create a 'phone_images' folder and drop photos in it.")
            sys.exit(1)

    for path in image_paths:
        try:
            process_and_show(path, model)
        except ValueError as e:
            print(f"\n{os.path.basename(path)}: {e}")

    print(f"\nResults saved under: {RESULTS_DIR}/")
