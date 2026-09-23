# zeckendorf-prune

Structured sparsity via the Zeckendorf adjacency constraint: **no two adjacent weights active**.

One rule from one matrix (`M = [[1,1],[1,0]]`) gives you:
- **Pruning** — 50% structured sparsity, competitive with NVIDIA 2:4 (0.50% gap on ResNet-20/CIFAR-10)
- **Encoding** — Fibonacci-coded weights enable shift-and-add multiplication (86% multiplier area reduction)
- **Integrity** — Free corruption detection via adjacency check (32% of single-bit flips caught, zero overhead)
- **Serialization** — Self-delimiting bitstreams, no length headers, analytically optimal compression

No NVIDIA hardware required. No sparse tensor cores. The constraint is simple enough for any architecture to exploit.

## Quick Start

```python
from zeckendorf_prune import prune, finetune, check

# 1. Prune
model, masks = prune(trained_model)
# → 50% density, adjacency-constrained

# 2. Fine-tune (mask-aware: gradients zeroed on pruned positions; amp=True runs fp16 on CUDA)
finetune(model, train_loader, epochs=40, masks=masks, val_loader=test_loader)

# 3. Verify
report = check(model, masks)
assert report["_summary"]["all_masks_valid"]
```

## Fibonacci Encoding

```python
from zeckendorf_prune import FibonacciEncoder

encoder = FibonacciEncoder(n_digits=8)  # 55 quantization levels
# Encode surviving weights, layer by layer
scales = {}  # name → (scale, offset) for cassini_check, save_checkpoint, export_bitstream
for name, param in model.named_parameters():
    if name in masks:
        # encode_tensor returns the scale; the offset is the smallest kept weight (level 0)
        offset = param.data[masks[name].bool()].min().item()
        encoded, scale, rmse = encoder.encode_tensor(param.data, mask=masks[name])
        param.data.copy_(encoded)
        scales[name] = (scale, offset)
# → Weights are now sums of non-consecutive Fibonacci numbers
# → Multiplication = shift-and-add (no multiplier needed)
```

## Integrity Checking

```python
from zeckendorf_prune import cassini_check

# Check all weights for corruption (free — just scan for adjacent 1s)
report = cassini_check(model, masks, encoder, scales)
print(f"Pass rate: {report['_aggregate']['pass_rate']:.1%}")
```

## Export

```python
from zeckendorf_prune.export import save_checkpoint, export_onnx, export_bitstream

# PyTorch checkpoint (includes masks + metadata)
save_checkpoint(model, masks, "model_zeck.pt", encoder=encoder, scales=scales)

# ONNX (sparse weights baked in, runs on any runtime)
export_onnx(model, "model_zeck.onnx")

# Self-delimiting bitstream (minimal format for edge deployment)
stats = export_bitstream(model, masks, encoder, scales, "model.zeck")
print(f"{stats['bits_per_weight']:.1f} bits/weight")
```

## CLI

```bash
# Verify checkpoint integrity
zeck check model_zeck.pt

# Print pruning statistics
zeck info model_zeck.pt
```

## Benchmark Results (ResNet-20, CIFAR-10)

| Method | Accuracy | Drop | Density |
|--------|----------|------|---------|
| Dense baseline | 91.66% | — | 100% |
| Zeckendorf pruned | 88.05% | 3.61% | 50.8% |
| NVIDIA 2:4 pruned | 88.55% | 3.11% | 50.9% |
| Zeckendorf + Fibonacci encoded | 87.45% | 4.21% | 50.8% |

Estimated hardware savings (vs dense binary), not measured: 86% multiplier area and 71% power-delay product, which combine the 50.8% density with the 73% multiplier-area and 43% power-delay-product reductions that DATE 2021 reported for its Fibonacci weight encoding (1 − 0.508 × 0.27 and 1 − 0.508 × 0.57, in `compute_hardware_cost` of [`.docs/experiment/phase1_stacked.py`](.docs/experiment/phase1_stacked.py)).
That paper's encoding keeps weights whose binary form has no two adjacent 1s, while this package stores Zeckendorf digits, for which no circuit cost has been estimated.
Free integrity: 32% single-bit-flip detection via adjacency check.

## Benchmark Results (ResNets, CIFAR-10 at 224 px, Colab T4)

In short:

- Pruning zeroes about half of each network's conv weights.
  - Why: the Zeckendorf rule never keeps two neighboring output channels of a layer, so at most half of them survive; the pruner keeps the heaviest set the rule allows, which lands just under half (48.8–49.8% here).
- With the original ImageNet weights (V1), 5 epochs of retraining bring every pruned model back to within 2–7 points of its unpruned accuracy.
  - Why: right after pruning every model scores 10%, a single predicted class, yet V1 models climb back to 82–93% in the first retraining epoch alone, so the surviving weights still carry most of what was learned; retraining with the pruned weights held at zero teaches them to cover for the missing channels.
- Among the V1 models, the bigger the network, the smaller the loss: ResNet-18 goes from 94.81% unpruned to 88.05% pruned (6.8 points), ResNet-152 from 97.47% to 95.28% (2.2 points).
  - Why (likely, not tested here): a bigger network has more channels doing overlapping work, so more is left to cover for the ones pruned away; the trend holds even among the three models retrained at the same batch size (ResNet-18, -34 and -50 lose 6.8, 5.2 and 4.2 points).
- With torchvision's newer V2 weights, the same retraining leaves the pruned models 15–29 points below their unpruned accuracy: ResNet-101 goes from 97.53% to 68.35%, and ResNet-152, the best of them, from 97.70% to 82.27%.
  - Why: not established. The pruned V2 models start from the same 10% but climb far more slowly, reaching 47.8%, 39.5% and 58.9% (ResNet-50, -101, -152) after the first retraining epoch, against 85.8%, 91.4% and 92.8% for V1. Two untested explanations: pruning removes more of what the V2 weights rely on, or 5 epochs at this small learning rate are too few for them to recover; a longer or higher-learning-rate retraining would tell them apart.

| Model | ImageNet weights | Params | Dense | Pruned | Drop (pts) | Conv density | Pruned convs | Minutes |
|-------|------------------|--------|-------|--------|------------|--------------|--------------|---------|
| ResNet-18 | V1 | 11.2M | 94.81% | 88.05% | 6.76 | 49.8% | 20 | — |
| ResNet-34 | V1 | 21.3M | 96.22% | 91.02% | 5.20 | 49.7% | 36 | 11.2 |
| ResNet-50 | V1 | 23.5M | 96.02% | 91.80% | 4.22 | 49.5% | 53 | 19.8 |
| ResNet-50 | V2 | 23.5M | 96.70% | 70.49% | 26.21 | 49.5% | 53 | 19.8 |
| ResNet-101 | V1 | 42.5M | 97.18% | 94.20% | 2.98 | 48.9% | 104 | 31.7 |
| ResNet-101 | V2 | 42.5M | 97.53% | 68.35% | 29.18 | 49.3% | 104 | 31.2 |
| ResNet-152 | V1 | 58.2M | 97.47% | 95.28% | 2.19 | 48.8% | 155 | 44.6 |
| ResNet-152 | V2 | 58.2M | 97.70% | 82.27% | 15.43 | 49.2% | 155 | 44.1 |

One run of [`.docs/quickstart_gpu.ipynb`](.docs/quickstart_gpu.ipynb) with zeckendorf-prune 0.2.1 on 2026-09-23: ResNet-18 from its walkthrough sections, the others from its variant cells.
Each model starts from torchvision's ImageNet weights with a new 10-class `fc` head, gets 2 dense fine-tuning epochs, has every conv layer pruned while `fc` stays dense, and gets 5 mask-aware fine-tuning epochs; training is SGD with a cosine schedule under fp16 autocast.
The learning rate is 0.01 for the dense epochs and 0.001 after pruning at batch 128 (ResNet-18 to ResNet-50), halved along with the batch for ResNet-101 and ResNet-152 at batch 64.
Dense and Pruned are the best CIFAR-10 test accuracy over each stage's epochs; before fine-tuning, every pruned model scored 10.00%, a single predicted class.
Minutes is the wall time from the dense fine-tune through the final mask check; ResNet-18 took about a minute per epoch.

## How It Works

The Zeckendorf constraint ("no two consecutive 1s") is applied along the output channel axis of each layer.
The model's last Conv2d or Linear layer — the classifier head in standard architectures — stays dense by default, because along that axis its positions are the classes and the constraint would keep at most half of them (`prune_head=True` prunes it too; `exclude=` keeps any other layer whole).
A dynamic programming algorithm finds the maximum-weight independent set — the optimal subset of channels to keep given their magnitudes — subject to this constraint.

The same constraint governs the Fibonacci number system: every positive integer has a unique representation as a sum of non-consecutive Fibonacci numbers. Encoding weights in this system enables multiplication via shift-and-add (each Fibonacci number is the sum of two predecessors), and corrupted codewords are detectable by scanning for the forbidden "11" pattern.

All three properties — structured sparsity, efficient arithmetic, free error detection — derive from the single matrix `M = [[1,1],[1,0]]`.

## Citation

```
@software{zeckendorf_prune,
  title={Zeckendorf-Prune: Structured Sparsity via the Adjacency Constraint},
  year={2026},
  url={https://github.com/ezexe/phi-prune}
}
```
