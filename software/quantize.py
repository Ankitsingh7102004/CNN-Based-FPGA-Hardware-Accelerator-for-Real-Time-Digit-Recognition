"""
Step 2: Quantization (PTQ vs QAT-lite, int8 vs int4)
------------------------------------------------------
Takes the trained FP32 model from Step 1 and produces several quantized
versions, logging accuracy, model size, per-layer scale factors, and
quantization time for each — so you get a direct comparison table for
the report.

Configs compared:
  1. FP32                    (no quantization — reference)
  2. PTQ  int8 (weights+acts) (calibrate only, no retraining)
  3. QAT  int8 (weights+acts) (fine-tune with fake-quant in the loop)
  4. PTQ  int4 weights / int8 acts   (bit-width sweep)
  5. QAT  int4 weights / int8 acts   (bit-width sweep)

Run with:
    python quantize.py
(expects baseline_model.pth / augmented_model.pth from train.py's
 results/step1_training/ folder to already exist)
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
STEP1_DIR = "results/step1_training"
RESULTS_DIR = "results/step2_quantization"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Which Step 1 model to quantize — "augmented" is the deployment target
# (better real-world robustness); change to "baseline" if you want that instead.
MODEL_TO_QUANTIZE = "augmented"

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
calib_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(2000)),
                           batch_size=100, shuffle=False)
finetune_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# ---------------------------------------------------------------------------
# Model (must match train.py exactly)
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


# ---------------------------------------------------------------------------
# Symmetric fake-quantization (straight-through estimator)
# ---------------------------------------------------------------------------
def fake_quantize(x, scale, num_bits):
    """Symmetric quantize-dequantize with straight-through gradient."""
    qmax = 2 ** (num_bits - 1) - 1
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    dq = q * scale
    # straight-through estimator: gradient passes through unchanged
    return x + (dq - x).detach()


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


def wrap_model_for_quantization(model, weight_bits, act_bits):
    """Replaces conv/linear layers with fake-quant versions (weights copied)."""
    model.conv1 = FakeQuantConv2d(model.conv1, weight_bits, act_bits)
    model.conv2 = FakeQuantConv2d(model.conv2, weight_bits, act_bits)
    model.fc1 = FakeQuantLinear(model.fc1, weight_bits, act_bits)
    model.fc2 = FakeQuantLinear(model.fc2, weight_bits, act_bits)
    return model


def calibrate_activation_scales(model, loader, act_bits):
    """Runs calibration batches, recording max-abs activation per fake-quant layer."""
    model.eval()
    max_vals = {"conv1": 0.0, "conv2": 0.0, "fc1": 0.0, "fc2": 0.0}

    def hook(name):
        def fn(module, inp, out):
            max_vals[name] = max(max_vals[name], inp[0].abs().max().item())
        return fn

    handles = [
        model.conv1.register_forward_hook(hook("conv1")),
        model.conv2.register_forward_hook(hook("conv2")),
        model.fc1.register_forward_hook(hook("fc1")),
        model.fc2.register_forward_hook(hook("fc2")),
    ]
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs)
    for h in handles:
        h.remove()

    qmax = 2 ** (act_bits - 1) - 1
    for name, layer in [("conv1", model.conv1), ("conv2", model.conv2),
                         ("fc1", model.fc1), ("fc2", model.fc2)]:
        scale = max(max_vals[name], 1e-8) / qmax
        layer.act_scale.fill_(scale)
    return max_vals


def get_scale_report(model):
    report = {}
    for name, layer in [("conv1", model.conv1), ("conv2", model.conv2),
                         ("fc1", model.fc1), ("fc2", model.fc2)]:
        base = layer.conv if hasattr(layer, "conv") else layer.linear
        w_scale = (base.weight.abs().max() /
                   (2 ** (layer.weight_bits - 1) - 1)).item()
        report[name] = {"weight_scale": w_scale,
                         "act_scale": layer.act_scale.item()}
    return report


# ---------------------------------------------------------------------------
# Evaluation / fine-tuning
# ---------------------------------------------------------------------------
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def quantized_model_size_kb(model, weight_bits):
    """Effective size if weights were actually packed at weight_bits
    (fake-quant keeps them stored as float32, so we compute this analytically —
    real packing happens in the .mif export stage before FPGA deployment)."""
    weight_params = sum(
        m.weight.numel() for m in [model.conv1.conv, model.conv2.conv,
                                    model.fc1.linear, model.fc2.linear]
    )
    bias_params = sum(
        m.bias.numel() for m in [model.conv1.conv, model.conv2.conv,
                                  model.fc1.linear, model.fc2.linear]
    )
    size_bytes = weight_params * (weight_bits / 8) + bias_params * 4  # biases kept int32
    return size_bytes / 1024


def run_config(name, weight_bits, act_bits, do_finetune, base_state_dict):
    print(f"\n{'=' * 60}\nQuantization config: {name}\n{'=' * 60}")
    start = time.time()

    model = MnistCNN()
    model.load_state_dict(base_state_dict)
    model = wrap_model_for_quantization(model, weight_bits, act_bits)

    calibrate_activation_scales(model, calib_loader, act_bits)

    if do_finetune:
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(3):  # short QAT fine-tune
            for imgs, labels in finetune_loader:
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
            # re-calibrate activation ranges after each epoch of weight updates
            calibrate_activation_scales(model, calib_loader, act_bits)
            print(f"  QAT fine-tune epoch {epoch + 1}/3 done")

    acc = evaluate(model)
    elapsed = time.time() - start
    size_kb = quantized_model_size_kb(model, weight_bits)
    scales = get_scale_report(model)

    state_path = os.path.join(RESULTS_DIR, f"{name}_state.pth")
    torch.save(model.state_dict(), state_path)

    print(f"Accuracy: {acc*100:.2f}% | Size: {size_kb:.2f} KB | "
          f"Time: {elapsed:.1f}s | Saved: {state_path}")

    return {
        "name": name,
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "accuracy": acc,
        "size_kb": size_kb,
        "quantization_time_sec": elapsed,
        "layer_scales": scales,
    }


def print_summary_table(results, fp32_acc, fp32_size_kb):
    print(f"\n{'=' * 90}")
    print(f"{'Config':<20}{'W-bits':<8}{'A-bits':<8}{'Accuracy':<10}"
          f"{'Size(KB)':<12}{'Time(s)':<10}")
    print("-" * 90)
    print(f"{'FP32 (reference)':<20}{'32':<8}{'32':<8}{fp32_acc*100:<10.2f}"
          f"{fp32_size_kb:<12.2f}{'--':<10}")
    for r in results:
        print(f"{r['name']:<20}{r['weight_bits']:<8}{r['act_bits']:<8}"
              f"{r['accuracy']*100:<10.2f}{r['size_kb']:<12.2f}"
              f"{r['quantization_time_sec']:<10.1f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model_path = os.path.join(STEP1_DIR, f"{MODEL_TO_QUANTIZE}_model.pth")
    base_state_dict = torch.load(model_path, map_location=DEVICE)

    # FP32 reference accuracy (recomputed here for a fair apples-to-apples run)
    fp32_model = MnistCNN()
    fp32_model.load_state_dict(base_state_dict)
    fp32_acc = evaluate(fp32_model)
    fp32_size_kb = os.path.getsize(model_path) / 1024
    print(f"FP32 reference accuracy: {fp32_acc*100:.2f}% | "
          f"size: {fp32_size_kb:.2f} KB")

    configs = [
        ("PTQ_int8", 8, 8, False),
        ("QAT_int8", 8, 8, True),
        ("PTQ_int4w_int8a", 4, 8, False),
        ("QAT_int4w_int8a", 4, 8, True),
    ]

    all_results = []
    for name, wb, ab, ft in configs:
        result = run_config(name, wb, ab, ft, base_state_dict)
        all_results.append(result)

    print_summary_table(all_results, fp32_acc, fp32_size_kb)

    with open(os.path.join(RESULTS_DIR, "quantization_results.json"), "w") as f:
        json.dump({
            "model_quantized": MODEL_TO_QUANTIZE,
            "fp32_accuracy": fp32_acc,
            "fp32_size_kb": fp32_size_kb,
            "configs": all_results,
        }, f, indent=2)

    print(f"\nAll results saved under: {RESULTS_DIR}/")
