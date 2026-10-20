"""
Step 1: MNIST CNN Training (Baseline + Augmented)
---------------------------------------------------
Trains two versions of a small CNN on the full 60,000-image MNIST training
set, logs training curves, timing, CPU/RAM usage, model size, and
parameter/FLOP counts for both, and saves everything needed for the
project report and for Step 2 (quantization).

Run with:
    pip install torch torchvision matplotlib psutil torchinfo thop
    python train.py
"""

import os
import time
import json
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Optional (nice-to-have) libraries — script still works without them
try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")  # no dedicated GPU on this laptop — CPU is fine
RESULTS_DIR = "results/step1_training"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

augmented_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MnistCNN(nn.Module):
    """Small CNN sized to be easy to quantize into FPGA BRAM later."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # -> 8 x 14 x 14
        x = self.pool(self.relu(self.conv2(x)))   # -> 16 x 7 x 7
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss_sum += criterion(outputs, labels).item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total, loss_sum / total


def train_one_config(name, train_transform):
    """Trains one model configuration end-to-end and returns a results dict."""
    print(f"\n{'=' * 60}\nTraining configuration: {name}\n{'=' * 60}")

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MnistCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    process = psutil.Process(os.getpid())
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    cpu_samples, ram_samples = [], []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = loss_sum / total
        val_acc, val_loss = evaluate(model, test_loader)

        cpu_samples.append(process.cpu_percent(interval=0.1))
        ram_samples.append(process.memory_info().rss / (1024 ** 2))  # MB

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    total_time = time.time() - start_time

    # --- Model size / params / FLOPs ---
    model_path = os.path.join(RESULTS_DIR, f"{name}_model.pth")
    torch.save(model.state_dict(), model_path)
    model_size_kb = os.path.getsize(model_path) / 1024
    num_params = sum(p.numel() for p in model.parameters())

    flops = None
    if HAS_THOP:
        try:
            dummy_input = torch.randn(1, 1, 28, 28)
            macs, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
            flops = macs * 2  # 1 MAC = 2 FLOPs
        except Exception as e:
            print(f"(thop FLOPs count skipped: {e})")

    # --- Save everything for the report FIRST, before any optional printing ---
    result = {
        "name": name,
        "epochs": EPOCHS,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "total_training_time_sec": total_time,
        "avg_cpu_percent": sum(cpu_samples) / len(cpu_samples),
        "peak_ram_mb": max(ram_samples),
        "model_size_kb": model_size_kb,
        "num_params": num_params,
        "flops": flops,
        "history": history,
    }

    with open(os.path.join(RESULTS_DIR, f"{name}_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    # --- Optional pretty summary (never let this crash the run) ---
    if HAS_TORCHINFO:
        try:
            print("\nModel summary:")
            summary_str = str(torchinfo_summary(
                model, input_size=(1, 1, 28, 28), verbose=0
            ))
            # Windows terminals (cp1252) can't print some unicode box-drawing
            # characters torchinfo uses — fall back to ASCII-safe printing.
            print(summary_str.encode("ascii", errors="replace").decode("ascii"))
        except Exception as e:
            print(f"(torchinfo summary skipped due to display error: {e})")

    return result


def plot_comparison(results):
    """Plots baseline vs augmented accuracy curves side by side."""
    plt.figure(figsize=(8, 5))
    for r in results:
        plt.plot(r["history"]["epoch"], r["history"]["val_acc"],
                  marker="o", label=f"{r['name']} (val acc)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Baseline vs Augmented — Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.show()


def print_summary_table(results):
    print(f"\n{'=' * 70}")
    print(f"{'Config':<15}{'Val Acc':<10}{'Time(s)':<10}{'CPU%':<8}"
          f"{'RAM(MB)':<10}{'Size(KB)':<10}{'Params':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<15}{r['final_val_acc']*100:<10.2f}"
              f"{r['total_training_time_sec']:<10.1f}{r['avg_cpu_percent']:<8.1f}"
              f"{r['peak_ram_mb']:<10.1f}{r['model_size_kb']:<10.1f}{r['num_params']:<10}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_results = []
    all_results.append(train_one_config("baseline", base_transform))
    all_results.append(train_one_config("augmented", augmented_transform))

    plot_comparison(all_results)
    print_summary_table(all_results)

    print(f"\nAll results, models, and plots saved under: {RESULTS_DIR}/")
