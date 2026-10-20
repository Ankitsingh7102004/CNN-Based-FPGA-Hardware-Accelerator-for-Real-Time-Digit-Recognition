# CNN-Based FPGA Hardware Accelerator for Real-Time Digit Recognition

An end-to-end MNIST handwritten-digit classifier: a small CNN is trained in PyTorch, quantized to **int8**
(quantization-aware training), converted to an **integer-only** reference model, and executed by a hand-written
**Verilog** inference core on a **DE10-Standard (Intel Cyclone V SoC)** board. The predicted digit is shown on a
7-segment display. The whole design runs in FPGA fabric (RTL, LUTs, registers, block RAM); the ARM HPS is not used.

> **Status.** Version 1 (stored MNIST test images -> FPGA -> 7-segment) is complete and verified on hardware.
> Version 2 (live image input: laptop webcam / TRDB-D5M 5 MP camera) is in progress and is not part of this release.

## Pipeline

```
PyTorch training -> QAT int8 -> integer-only model -> .mem/.vh export -> Verilog core -> DE10-Standard -> HEX0
   train.py         quantize.py    export_int.py        (weights, LUT)    cnn_core_v1.v   mnist_fpga_top.v
```

## Network

| Layer | Shape | Notes |
|---|---|---|
| Input | 1 x 28 x 28 | MNIST pixel, mapped through a 256-entry input LUT (normalization + int8 quantization) |
| Conv1 | 1 -> 8, 3x3, pad 1 | ReLU, 2x2 max-pool -> 8 x 14 x 14 |
| Conv2 | 8 -> 16, 3x3, pad 1 | ReLU, 2x2 max-pool -> 16 x 7 x 7 |
| FC1 | 784 -> 32 | ReLU |
| FC2 | 32 -> 10 | argmax on int32 accumulators |

Integer recipe per layer (identical in Python and RTL):

```
acc = sum(int8_in * int8_w) + int32_bias ; acc = max(acc,0) ; [maxpool]
out = clamp((acc * M + 2^(S-1)) >> S, -128, 127)
```

ReLU and max-pool are applied before requantization, so the requantizer runs on a quarter of the values.

## Results

| Stage | Result |
|---|---|
| Float / QAT int8 model, full 10,000-image MNIST test set (laptop) | **99.01 %** |
| Integer-only export, same 10,000 images | **99.01 %**, 0 of 10,000 predictions differ from the fake-quant model |
| RTL simulation vs. integer model (100 test images) | **100 / 100 bit-exact** (logits and prediction) |
| Latency, serial core | 351,021 cycles/image = about 7 ms at 50 MHz |
| Board demo (DE10-Standard) | Stored images classified correctly; HEX0 shows the prediction, HEX2 the true label, LEDR1 = correct |

The core is deliberately simple and serial (one MAC at a time) to make bit-exact verification easy; a parallel
MAC-array version is the obvious next step for lower latency.

## Repository layout

```
software/   train.py  quantize.py  export_int.py  test_laptop.py  phone_test.py
fpga/v1/    cnn_core_v1.v  mnist_fpga_top.v  tb_cnn_v1.v   (+ the exported .mem files and quant_params.vh, see below)
fpga/quartus/  de10_setup.tcl  cnn_core_v1.sdc
```

## Reproduce

### 1. Train, quantize, export (laptop)
```bash
pip install torch torchvision matplotlib psutil scikit-learn seaborn pillow scipy
cd software
python train.py          # trains the CNN on 60,000 MNIST images, saves curves and stats
python quantize.py       # QAT int8
python test_laptop.py    # 10,000-image golden reference: accuracy, confusion matrix, per-class P/R/F1
python export_int.py     # integer-only model + FPGA export
```
`export_int.py` writes `quant_params.vh`, `input_lut.mem`, `conv1_w/b.mem`, `conv2_w/b.mem`, `fc1_w/b.mem`,
`fc2_w/b.mem`, `test_images.mem`, `test_labels.mem`, `expected_pred.mem`, `expected_logits.mem`
(in `results/step4_export`). Copy all of them into `fpga/v1/`.

### 2. Simulate (Icarus Verilog)
```bash
cd fpga/v1
iverilog -g2005 -o sim.vvp tb_cnn_v1.v cnn_core_v1.v && vvp sim.vvp +n=100
```

### 3. Build and run on the board (Quartus Prime Lite 25.1, DE10-Standard)
1. New project in `fpga/v1/`, device `5CSXFC6D6F31C6`, top-level entity `mnist_fpga_top`.
2. Add `mnist_fpga_top.v`, `cnn_core_v1.v`, `quant_params.vh` and `fpga/quartus/cnn_core_v1.sdc`.
3. Tools -> Tcl Scripts -> run `fpga/quartus/de10_setup.tcl` (pin assignments).
4. Compile, then program the `.sof` with the USB-Blaster.

Board controls: **KEY0** reset, **KEY1/KEY2** next/previous stored test image, **KEY3** re-run.
**HEX0** predicted digit, **HEX2** true label, **HEX5:HEX4** image index, **LEDR0** busy, **LEDR1/2** correct/wrong.

## Roadmap
- Version 2: live input from a laptop webcam and the TRDB-D5M camera (preprocess to 28 x 28, classify on the FPGA)
- Full 10,000-image FPGA test with confusion matrix, precision/recall/F1
- Parallel MAC-array core, resource and power reporting

## Author
Ankit Singh - B.Tech ECE, VIT Chennai. License: MIT.
