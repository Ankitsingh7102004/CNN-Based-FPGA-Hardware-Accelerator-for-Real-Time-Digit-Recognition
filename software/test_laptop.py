"""
Step 3: Full 10k-Image Software Test (Golden Reference)
-----------------------------------------------------------
Runs the QAT_int8 quantized model (simulated in software, same fake-quant
math the FPGA will implement) over the full 10,000-image MNIST test set.

Produces:
  - Overall accuracy
  - Confusion matrix (plot + raw numbers)
  - Precision / recall / F1 per class + macro/weighted averages
  - Per-image inference latency (software reference — compare against
    FPGA latency in Step 4/5)

This becomes your GOLDEN REFERENCE: if FPGA outputs ever disagree with
this script's outputs on the same images, the bug is in the RTL/export,
not the model.

Run with:
    pip install scikit-learn seaborn
    python test_laptop.py
"""

import os
import time
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_recall_fscore_support)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
STEP2_DIR = "results/step2_quantization"
RESULTS_DIR = "results/step3_laptop_test"
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG_TO_TEST = "QAT_int8"   # matches the config name from quantize.py
WEIGHT_BITS, ACT_BITS = 8, 8   # must match the config above

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
# batch_size=1 so we get a genuine per-image latency number
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# ---------------------------------------------------------------------------
# Model + fake-quant wrappers (must exactly match quantize.py)
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
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    state_path = os.path.join(STEP2_DIR, f"{CONFIG_TO_TEST}_state.pth")
    model = build_quantized_model(WEIGHT_BITS, ACT_BITS)
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    model.eval()

    all_preds, all_labels, latencies = [], [], []

    print(f"Running full 10,000-image test on config: {CONFIG_TO_TEST}...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            start = time.perf_counter()
            output = model(imgs)
            latencies.append(time.perf_counter() - start)

            pred = output.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(labels.item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    latencies = np.array(latencies) * 1000  # convert to ms

    accuracy = (all_preds == all_labels).mean()

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)

    # --- Precision / recall / F1 ---
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(10))
    )
    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    report_text = classification_report(all_labels, all_preds)

    print(f"\nOverall accuracy on 10,000 images: {accuracy*100:.2f}%")
    print(f"Avg latency/image: {latencies.mean():.4f} ms "
          f"(min {latencies.min():.4f}, max {latencies.max():.4f})")
    print(f"\nPer-class report:\n{report_text}")

    # --- Plot confusion matrix ---
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {CONFIG_TO_TEST} (Software, 10k test images)")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color=color, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.show()

    # --- Plot latency distribution ---
    plt.figure(figsize=(7, 4))
    plt.hist(latencies, bins=50)
    plt.xlabel("Per-image inference latency (ms)")
    plt.ylabel("Count")
    plt.title("Software Inference Latency Distribution (golden reference)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_histogram.png"), dpi=150)
    plt.show()

    # --- Save everything ---
    results = {
        "config": CONFIG_TO_TEST,
        "num_images": len(all_labels),
        "overall_accuracy": accuracy,
        "avg_latency_ms": float(latencies.mean()),
        "min_latency_ms": float(latencies.min()),
        "max_latency_ms": float(latencies.max()),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            str(i): {"precision": float(precision[i]), "recall": float(recall[i]),
                     "f1": float(f1[i]), "support": int(support[i])}
            for i in range(10)
        },
        "macro_avg": report_dict["macro avg"],
        "weighted_avg": report_dict["weighted avg"],
    }

    with open(os.path.join(RESULTS_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Also save raw predictions — needed later to directly compare against
    # FPGA outputs image-by-image in Step 5
    np.savez(os.path.join(RESULTS_DIR, "predictions.npz"),
              preds=all_preds, labels=all_labels)

    print(f"\nAll results, plots, and predictions saved under: {RESULTS_DIR}/")
