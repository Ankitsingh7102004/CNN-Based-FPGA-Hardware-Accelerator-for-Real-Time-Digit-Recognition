"""
Step 4a: Integer-Only Reference Model + FPGA Export
-----------------------------------------------------
Takes the trained QAT_int8 model from Step 2 and:

  1. Converts it to a PURE INTEGER model (int8 weights, int32 biases,
     integer multiply + shift between layers, no floating point at all
     in the forward pass) - this is exactly what the FPGA will compute.
  2. Runs all 10,000 MNIST test images through it and compares against the
     fake-quant results from Step 3 (accuracy + prediction agreement).
  3. Exports everything the RTL needs:
        - .mif files   -> initialize FPGA block RAM/ROM in Quartus
        - .mem files   -> same data for $readmemh in simulation
        - quant_params.vh / .json -> multipliers and shifts per layer
        - test vectors (images + expected outputs) for your testbench

Integer pipeline per layer (this is the hardware recipe):

    acc   = sum(int8_input * int8_weight) + int32_bias      (int32)
    acc   = max(acc, 0)                                     (ReLU)
    acc   = maxpool2x2(acc)                                 (conv layers only)
    out   = clamp((acc * M + 2^(S-1)) >> S, -128, 127)      (requantize -> int8)

  (ReLU and max-pool are done BEFORE requantizing. This gives identical
   results because rounding/clamping is monotonic, and it means the
   requantizer only runs on 1/4 as many values.)
  The last layer (fc2) skips requantization: argmax is taken directly on
  the int32 accumulators.

Run with:
    python export_int.py
(expects results/step2_quantization/QAT_int8_state.pth from quantize.py)
"""

import os
import json
import numpy as np
import torch
import torchvision
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
STEP2_DIR = "results/step2_quantization"
STEP3_DIR = "results/step3_laptop_test"
OUT_DIR = "results/step4_export"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG = "QAT_int8"
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
N_TEST_VECTORS = 100      # number of test images exported for the RTL testbench
BATCH = 500               # batch size for the numpy integer forward pass

LAYERS = [("conv1", "conv"), ("conv2", "conv"), ("fc1", "linear"), ("fc2", "linear")]


# ---------------------------------------------------------------------------
# Load trained QAT_int8 weights and scales
# ---------------------------------------------------------------------------
state_path = os.path.join(STEP2_DIR, f"{CONFIG}_state.pth")
state = torch.load(state_path, map_location="cpu")


def tensor(name):
    return state[name].detach().cpu().numpy()


def make_multiplier(m):
    """Convert a real number m (the requant scale ratio) into an integer
    multiplier M (fits in 16 bits) and a right-shift S so that
        x * m  ~=  (x * M) >> S
    """
    shift = 0
    while m * (1 << shift) < (1 << 14):
        shift += 1
        if shift > 45:
            raise ValueError(f"Cannot represent multiplier {m}")
    M = int(round(m * (1 << shift)))
    assert 0 < M <= (1 << 15), f"Multiplier out of range: {M}"
    return M, shift


int_layers = {}
for name, kind in LAYERS:
    w32 = tensor(f"{name}.{kind}.weight").astype(np.float32)
    b64 = tensor(f"{name}.{kind}.bias").astype(np.float64)
    s_in = float(tensor(f"{name}.act_scale"))

    # Same weight quantization as the fake-quant model (per-tensor symmetric int8)
    w_scale = np.float32(np.abs(w32).max() / 127.0)
    w_scale = np.float32(max(float(w_scale), 1e-8))
    w_int = np.clip(np.round(w32 / w_scale), -128, 127).astype(np.int64)

    # Bias lives in accumulator units: real_bias / (input_scale * weight_scale)
    b_int = np.round(b64 / (s_in * float(w_scale))).astype(np.int64)

    int_layers[name] = {
        "w_int": w_int, "b_int": b_int,
        "s_in": s_in, "w_scale": float(w_scale),
    }

# Requantization constants: layer i output -> layer i+1 input scale
order = [n for n, _ in LAYERS]
for i, name in enumerate(order[:-1]):
    nxt = int_layers[order[i + 1]]
    m_real = (int_layers[name]["s_in"] * int_layers[name]["w_scale"]) / nxt["s_in"]
    M, S = make_multiplier(m_real)
    int_layers[name].update({"m_real": m_real, "M": M, "shift": S})

# Input LUT: raw pixel (0..255) -> normalized -> int8 code for conv1's input.
# In hardware this is a 256-entry ROM, so the FPGA never needs floating point.
p = np.arange(256, dtype=np.float32)
x_norm = (p / np.float32(255.0) - np.float32(MNIST_MEAN)) / np.float32(MNIST_STD)
input_lut = np.clip(
    np.round(x_norm / np.float32(int_layers["conv1"]["s_in"])), -128, 127
).astype(np.int64)


# ---------------------------------------------------------------------------
# Integer-only forward pass (numpy)
# Float64 matmul is used purely as a fast exact-integer engine: every sum here
# stays below 2^53, so float64 represents the integers exactly.
# ---------------------------------------------------------------------------
def conv3x3(q_in, w_int, b_int):
    N, C, H, W = q_in.shape
    O = w_int.shape[0]
    padded = np.pad(q_in, ((0, 0), (0, 0), (1, 1), (1, 1)))          # zero padding
    win = sliding_window_view(padded, (3, 3), axis=(2, 3))            # N,C,H,W,3,3
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, C * 9)
    out = cols.astype(np.float64) @ w_int.reshape(O, C * 9).T.astype(np.float64)
    out = np.rint(out).astype(np.int64).reshape(N, H, W, O).transpose(0, 3, 1, 2)
    return out + b_int.reshape(1, O, 1, 1)


def relu_pool(acc):
    acc = np.maximum(acc, 0)
    N, O, H, W = acc.shape
    return acc.reshape(N, O, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def requant(acc, M, shift):
    rnd = (1 << (shift - 1)) if shift > 0 else 0
    return np.clip((acc.astype(np.int64) * M + rnd) >> shift, -128, 127)


def linear(q_in, w_int, b_int):
    out = q_in.astype(np.float64) @ w_int.T.astype(np.float64)
    return np.rint(out).astype(np.int64) + b_int.reshape(1, -1)


max_abs = {"conv1": 0, "conv2": 0, "fc1": 0, "fc2": 0}


def integer_forward(pixels_u8):
    """pixels_u8: (N, 28, 28) uint8 raw MNIST pixels -> (N, 10) int32 logits."""
    L = int_layers
    q0 = input_lut[pixels_u8].reshape(-1, 1, 28, 28)

    a1 = conv3x3(q0, L["conv1"]["w_int"], L["conv1"]["b_int"])
    max_abs["conv1"] = max(max_abs["conv1"], int(np.abs(a1).max()))
    q1 = requant(relu_pool(a1), L["conv1"]["M"], L["conv1"]["shift"])    # N,8,14,14

    a2 = conv3x3(q1, L["conv2"]["w_int"], L["conv2"]["b_int"])
    max_abs["conv2"] = max(max_abs["conv2"], int(np.abs(a2).max()))
    q2 = requant(relu_pool(a2), L["conv2"]["M"], L["conv2"]["shift"])    # N,16,7,7

    flat = q2.reshape(q2.shape[0], -1)          # channel-major flatten (c*49 + h*7 + w)
    a3 = linear(flat, L["fc1"]["w_int"], L["fc1"]["b_int"])
    max_abs["fc1"] = max(max_abs["fc1"], int(np.abs(a3).max()))
    q3 = requant(np.maximum(a3, 0), L["fc1"]["M"], L["fc1"]["shift"])    # N,32

    logits = linear(q3, L["fc2"]["w_int"], L["fc2"]["b_int"])            # N,10
    max_abs["fc2"] = max(max_abs["fc2"], int(np.abs(logits).max()))
    return logits


# ---------------------------------------------------------------------------
# Run the full 10k test set through the integer model
# ---------------------------------------------------------------------------
test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True)
pixels = test_set.data.numpy().astype(np.int64)      # (10000, 28, 28), values 0..255
labels = test_set.targets.numpy()

all_logits = []
for i in range(0, len(pixels), BATCH):
    all_logits.append(integer_forward(pixels[i:i + BATCH]))
all_logits = np.concatenate(all_logits)
int_preds = all_logits.argmax(axis=1)

int_acc = (int_preds == labels).mean()
print("=" * 64)
print("INTEGER-ONLY MODEL RESULTS (10,000 MNIST test images)")
print("=" * 64)
print(f"Integer-only accuracy: {int_acc * 100:.2f}%")

ref_path = os.path.join(STEP3_DIR, "predictions.npz")
if os.path.exists(ref_path):
    ref = np.load(ref_path)
    agree = (int_preds == ref["preds"]).mean()
    n_mismatch = int((int_preds != ref["preds"]).sum())
    fq_acc = (ref["preds"] == ref["labels"]).mean()
    print(f"Fake-quant accuracy (Step 3): {fq_acc * 100:.2f}%")
    print(f"Prediction agreement with fake-quant model: {agree * 100:.2f}% "
          f"({n_mismatch} of 10000 differ)")
else:
    print("(Step 3 predictions.npz not found - skipping agreement check)")

# ---------------------------------------------------------------------------
# Per-layer table (for the report + RTL sizing)
# ---------------------------------------------------------------------------
print("\nPer-layer quantization parameters and accumulator sizing")
print("-" * 64)
print(f"{'Layer':<7}{'act_scale':<12}{'w_scale':<12}{'M':<8}{'shift':<7}{'max|acc|':<11}{'bits':<5}")
for name, _ in LAYERS:
    d = int_layers[name]
    bits = int(max_abs[name]).bit_length() + 1      # +1 for sign
    M = d.get("M", "-")
    S = d.get("shift", "-")
    print(f"{name:<7}{d['s_in']:<12.6f}{d['w_scale']:<12.6f}{str(M):<8}{str(S):<7}"
          f"{max_abs[name]:<11}{bits:<5}")
print("(bits = signed accumulator width needed for that layer, worst case seen)")


# ---------------------------------------------------------------------------
# Export files for Quartus / simulation
# ---------------------------------------------------------------------------
def write_mif(path, values, width):
    mask, digits = (1 << width) - 1, width // 4
    with open(path, "w") as f:
        f.write(f"WIDTH={width};\nDEPTH={len(values)};\n\n")
        f.write("ADDRESS_RADIX=UNS;\nDATA_RADIX=HEX;\n\nCONTENT BEGIN\n")
        for addr, v in enumerate(values):
            f.write(f"    {addr} : {int(v) & mask:0{digits}X};\n")
        f.write("END;\n")


def write_mem(path, values, width):
    mask, digits = (1 << width) - 1, width // 4
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v) & mask:0{digits}X}\n")


def export(name, values, width):
    values = np.asarray(values).reshape(-1)
    write_mif(os.path.join(OUT_DIR, f"{name}.mif"), values, width)
    write_mem(os.path.join(OUT_DIR, f"{name}.mem"), values, width)
    return len(values)


depths = {}
for name, _ in LAYERS:
    d = int_layers[name]
    # weight order: [out_channel][in_channel][kh][kw]  (or [out][in] for FC)
    depths[f"{name}_w"] = export(f"{name}_w", d["w_int"], 8)
    depths[f"{name}_b"] = export(f"{name}_b", d["b_int"], 32)
depths["input_lut"] = export("input_lut", input_lut, 8)

# Test vectors for the RTL testbench (first N test images)
n = N_TEST_VECTORS
depths["test_images"] = export("test_images", pixels[:n], 8)           # n * 784 bytes
depths["test_labels"] = export("test_labels", labels[:n], 8)
depths["expected_pred"] = export("expected_pred", int_preds[:n], 8)
depths["expected_logits"] = export("expected_logits", all_logits[:n], 32)

# Verilog header with the requant constants
with open(os.path.join(OUT_DIR, "quant_params.vh"), "w") as f:
    f.write("// Auto-generated by export_int.py - do not edit by hand\n")
    for name, _ in LAYERS[:-1]:
        tag = name.upper()
        f.write(f"localparam integer M_{tag}     = {int_layers[name]['M']};\n")
        f.write(f"localparam integer SHIFT_{tag} = {int_layers[name]['shift']};\n")

# JSON with everything (for the report)
summary = {
    "config": CONFIG,
    "integer_accuracy": float(int_acc),
    "layers": {
        name: {
            "act_scale": int_layers[name]["s_in"],
            "weight_scale": int_layers[name]["w_scale"],
            "requant_multiplier_M": int_layers[name].get("M"),
            "requant_shift": int_layers[name].get("shift"),
            "max_abs_accumulator": max_abs[name],
        } for name, _ in LAYERS
    },
    "memory_depths": depths,
}
with open(os.path.join(OUT_DIR, "quant_params.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Golden predictions from the integer model (compare the FPGA against THIS)
np.savez(os.path.join(OUT_DIR, "integer_model_outputs.npz"),
         preds=int_preds, labels=labels, logits=all_logits)

total_w = sum(depths[f"{n}_w"] for n, _ in LAYERS)
print(f"\nExported to {OUT_DIR}/")
print(f"  Weight ROM depths: " +
      ", ".join(f"{n}={depths[f'{n}_w']}" for n, _ in LAYERS) +
      f"  (total {total_w} bytes)")
print(f"  Test vectors: {n} images with expected predictions and int32 logits")
print("  Files: *.mif (Quartus), *.mem (simulation), quant_params.vh/.json, "
      "integer_model_outputs.npz")
