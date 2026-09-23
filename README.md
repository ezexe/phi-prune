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
# Encode surviving weights
encoded_tensor, scale, rmse = encoder.encode_tensor(weight, mask=mask)
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

Hardware savings (vs dense binary): 86% multiplier area, 71% power-delay product.
Free integrity: 32% single-bit-flip detection via adjacency check.

## How It Works

The Zeckendorf constraint ("no two consecutive 1s") is applied along the output channel axis of each layer.
The last eligible layer — the classifier head in standard architectures — stays dense by default, because along that axis its positions are the classes and the constraint would keep at most half of them (`prune_head=True` prunes it too; `exclude=` keeps any other layer whole).
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
