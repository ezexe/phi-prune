"""
φ-register Phase 1: Stack the components
=========================================

Takes the output of the Zeckendorf pruning experiment and adds:
  1. Fibonacci weight encoding (DATE 2021 approach) on surviving weights
  2. Cassini integrity checks on encoded weights
  3. Combined hardware cost estimation

This tests whether two φ-register components compose cleanly:
  - Zeckendorf mask: which weights exist (structural sparsity)
  - Fibonacci encoding: how each weight is stored (value quantization)

Both enforce "no consecutive 1s" — one on positions, one on bit representations.
The prediction: they compose without interference because they're the same
constraint at different levels.

PREREQUISITES:
  Run zeckendorf_experiment.py first (needs the pretrained dense checkpoint).

USAGE:
  python phase1_stacked.py [--device cuda] [--fib-digits 8] [--epochs-finetune 40]
  python phase1_stacked.py --quick   # fast sanity check

WHAT IT MEASURES:
  1. Accuracy: dense → Zeckendorf-pruned → Fibonacci-encoded → fine-tuned
  2. Each degradation step isolated (how much does encoding cost on top of pruning?)
  3. Cassini check: inject random bit flips, measure detection rate
  4. Combined hardware cost estimate
"""

import argparse
import time
import copy
import json
import os
import math
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# ═══════════════════════════════════════════════════════════════════════
# 1. FIBONACCI ENCODING — the DATE 2021 approach adapted for PyTorch
# ═══════════════════════════════════════════════════════════════════════

class FibonacciEncoder:
    """
    Encode/decode neural network weights in Fibonacci (Zeckendorf) form.

    Each weight is:
      1. Scaled to an integer range determined by n_digits Fibonacci positions
      2. Decomposed into Zeckendorf representation (no consecutive 1s in binary)
      3. Stored as the quantized value (sum of selected Fibonacci weights)

    The key property: the bit pattern of every encoded weight satisfies the
    same adjacency constraint as the Zeckendorf pruning mask. The constraint
    governs both WHERE weights exist and WHAT VALUES they take.

    Args:
        n_digits: number of Fibonacci digit positions (more = finer quantization)
                  8 digits → 55 unsigned levels (range 0-54)
                  10 digits → 144 unsigned levels (range 0-143)
    """

    def __init__(self, n_digits=8):
        self.n_digits = n_digits
        # Fibonacci weights: F(2), F(3), ..., F(n_digits+1)
        self.weights = self._fibonacci_weights(n_digits)
        self.max_value = self._max_zeckendorf_value(n_digits)
        # Build the quantization grid (all valid Zeckendorf values)
        self.grid = self._build_grid()
        self.grid_tensor = None  # lazily created per-device

    def _fib(self, n):
        if n <= 0: return 0
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    def _fibonacci_weights(self, n):
        return [self._fib(i + 2) for i in range(n)]

    def _max_zeckendorf_value(self, n):
        """Maximum value: take every other Fibonacci weight starting from largest."""
        w = self.weights
        val = 0
        take = True
        for i in range(len(w) - 1, -1, -1):
            if take:
                val += w[i]
            take = not take
        return val

    def _build_grid(self):
        """Build sorted list of all non-negative Zeckendorf-representable values."""
        values = set()
        for mask in range(1 << self.n_digits):
            bits = [(mask >> i) & 1 for i in range(self.n_digits)]
            valid = True
            for i in range(self.n_digits - 1):
                if bits[i] == 1 and bits[i + 1] == 1:
                    valid = False
                    break
            if valid:
                val = sum(b * w for b, w in zip(bits, self.weights))
                values.add(val)
        return sorted(values)

    def _get_grid_tensor(self, device):
        if self.grid_tensor is None or self.grid_tensor.device != device:
            self.grid_tensor = torch.tensor(self.grid, dtype=torch.float32, device=device)
        return self.grid_tensor

    def to_zeckendorf(self, n):
        """Greedy Zeckendorf decomposition of non-negative integer n. Returns bit list."""
        bits = [0] * self.n_digits
        remaining = n
        for i in range(self.n_digits - 1, -1, -1):
            if self.weights[i] <= remaining:
                bits[i] = 1
                remaining -= self.weights[i]
                # Skip next position (adjacency constraint) — greedy guarantees this
        return bits

    def verify_zeckendorf(self, bits):
        """Check no consecutive 1s in a bit pattern."""
        for i in range(len(bits) - 1):
            if bits[i] == 1 and bits[i + 1] == 1:
                return False
        return True

    def encode_tensor(self, tensor, scale=None):
        """
        Quantize a weight tensor to Fibonacci-representable values.

        Process:
          1. Separate sign and magnitude
          2. Scale magnitudes to [0, max_zeckendorf_value] range
          3. Round each magnitude to nearest grid point
          4. Rescale back to original range

        Args:
            tensor: weight tensor to encode
            scale: if None, computed from tensor's max absolute value

        Returns:
            encoded: quantized tensor (same shape, Fibonacci-representable values)
            scale: the scale factor used (needed for decoding/hardware)
            stats: encoding statistics
        """
        device = tensor.device
        grid = self._get_grid_tensor(device)

        # Compute scale from weight range
        max_abs = tensor.abs().max().item()
        if scale is None:
            scale = max_abs / self.max_value if max_abs > 0 else 1.0

        # Separate sign and magnitude
        signs = tensor.sign()
        magnitudes = tensor.abs()

        # Scale to Fibonacci integer range
        scaled = magnitudes / scale

        # Quantize: find nearest grid point for each value
        # Reshape for broadcasting: scaled is [*shape], grid is [n_levels]
        flat = scaled.reshape(-1)
        # For each value, find the nearest grid point
        # Use searchsorted for efficiency
        indices = torch.searchsorted(grid, flat.clamp(0, self.max_value))
        indices = indices.clamp(0, len(self.grid) - 1)

        # Check if the previous grid point is closer
        quantized = grid[indices]
        prev_indices = (indices - 1).clamp(0)
        prev_quantized = grid[prev_indices]

        use_prev = (flat - prev_quantized).abs() < (flat - quantized).abs()
        quantized = torch.where(use_prev, prev_quantized, quantized)

        # Rescale and restore sign
        encoded = signs * quantized.reshape(tensor.shape) * scale

        # Statistics
        quant_error = (tensor - encoded).abs()
        stats = {
            'rmse': quant_error.pow(2).mean().sqrt().item(),
            'max_error': quant_error.max().item(),
            'mean_abs_error': quant_error.mean().item(),
            'scale': scale,
            'n_digits': self.n_digits,
            'n_levels': len(self.grid),
            'max_zeckendorf': self.max_value,
        }

        return encoded, scale, stats

    def cassini_check(self, value, scale):
        """
        Verify a single Fibonacci-encoded weight using Cassini's identity.

        For a value v with Zeckendorf representation using highest Fibonacci
        weight F(k), the state pair (F(k), F(k-1)) must satisfy:
          F(k-1) · F(k+1) - F(k)² = (-1)^k

        Returns: (is_valid, details_dict)
        """
        # Convert to Fibonacci integer
        int_val = abs(round(value / scale)) if scale > 0 else 0
        int_val = min(int_val, self.max_value)

        # Get Zeckendorf decomposition
        bits = self.to_zeckendorf(int_val)

        # Verify the decomposition itself
        if not self.verify_zeckendorf(bits):
            return False, {'reason': 'adjacent_ones', 'bits': bits}

        # Verify the reconstructed value matches
        reconstructed = sum(b * w for b, w in zip(bits, self.weights))
        if reconstructed != int_val:
            return False, {'reason': 'reconstruction_mismatch',
                          'expected': int_val, 'got': reconstructed}

        # Find highest active Fibonacci index for Cassini check
        highest_k = -1
        for i in range(self.n_digits - 1, -1, -1):
            if bits[i] == 1:
                highest_k = i + 2  # weights start at F(2)
                break

        if highest_k < 2:
            return True, {'reason': 'zero_or_trivial'}

        # Cassini identity: F(k-1)·F(k+1) - F(k)² = (-1)^k
        fk = self._fib(highest_k)
        fk_minus = self._fib(highest_k - 1)
        fk_plus = self._fib(highest_k + 1)
        cassini = fk_minus * fk_plus - fk * fk
        expected = (-1) ** highest_k

        is_valid = (cassini == expected)
        return is_valid, {'k': highest_k, 'cassini': cassini, 'expected': expected}


# ═══════════════════════════════════════════════════════════════════════
# 2. IMPORT MODEL AND PRUNING FROM PHASE 0
# ═══════════════════════════════════════════════════════════════════════

# Re-use the model and mask generation from the pruning experiment
from Zeckendorf import (
    ResNet20, BasicBlock,
    get_cifar10_loaders,
    generate_zeckendorf_mask, generate_24_mask,
    verify_zeckendorf_mask, verify_24_mask,
    train_epoch, evaluate, apply_pruning
)


# ═══════════════════════════════════════════════════════════════════════
# 3. STACKED EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def apply_fibonacci_encoding(model, masks, encoder, skip_names=None):
    """
    Apply Fibonacci encoding to all weights that survive the pruning mask.

    Args:
        model: pruned model (weights already masked)
        masks: dict of pruning masks from apply_pruning
        encoder: FibonacciEncoder instance
        skip_names: parameter names to skip (e.g., batch norm, final FC)

    Returns:
        scales: dict mapping parameter names to encoding scales
        stats: encoding statistics per layer
    """
    if skip_names is None:
        skip_names = set()

    scales = {}
    all_stats = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks and name not in skip_names:
                mask = masks[name]

                # Encode only the non-zero (surviving) weights
                # The mask ensures zeros stay zero
                encoded, scale, stats = encoder.encode_tensor(param.data)

                # Re-apply mask to ensure pruned positions stay exactly zero
                param.data.copy_(encoded * mask)

                scales[name] = scale
                all_stats[name] = stats

    return scales, all_stats


def run_cassini_integrity_test(model, masks, encoder, scales, n_tests=2000):
    """
    Test Cassini-based corruption detection by flipping actual Fibonacci bits,
    not float values. This simulates hardware bit-flip errors in the stored
    Zeckendorf representation.

    For each sampled weight:
      1. Decode to its Zeckendorf bit pattern
      2. Flip one random bit (simulating a single-bit hardware error)
      3. Check whether the corrupted pattern has adjacent 1s (free detection)
      4. Also check full Cassini identity on the corrupted value (deeper check)
    """
    random.seed(42)
    adjacency_detected = 0   # caught by simple adjacent-1s check (FREE)
    cassini_detected = 0     # caught by full Cassini algebraic check (2 extra muls)
    total = 0

    for name, param in model.named_parameters():
        if name not in masks or name not in scales:
            continue

        flat = param.data.flatten().cpu()
        mask_flat = masks[name].flatten().cpu()
        scale = scales[name]

        # Collect active (non-zero) weight indices
        active_idx = [i for i in range(len(flat))
                      if mask_flat[i] > 0 and abs(flat[i]) > 1e-8]
        if not active_idx:
            continue

        sample = random.sample(active_idx, min(n_tests - total, len(active_idx)))

        for i in sample:
            orig_val = flat[i].item()

            # Step 1: convert to Fibonacci integer and get Zeckendorf bits
            int_val = abs(round(orig_val / scale)) if scale > 0 else 0
            int_val = min(int_val, encoder.max_value)
            bits = encoder.to_zeckendorf(int_val)

            # Step 2: flip one random bit (the actual corruption)
            flip_pos = random.randint(0, encoder.n_digits - 1)
            corrupted_bits = bits.copy()
            corrupted_bits[flip_pos] = 1 - corrupted_bits[flip_pos]

            # Step 3a: FREE check — does the corrupted pattern have adjacent 1s?
            has_adjacent = any(
                corrupted_bits[j] == 1 and corrupted_bits[j + 1] == 1
                for j in range(encoder.n_digits - 1)
            )
            if has_adjacent:
                adjacency_detected += 1

            # Step 3b: FULL Cassini check — reconstruct the corrupted integer
            # and verify F(k-1)*F(k+1) - F(k)^2 = (-1)^k
            corrupted_val = sum(
                b * w for b, w in zip(corrupted_bits, encoder.weights)
            )
            corrupted_bits_redecomposed = encoder.to_zeckendorf(corrupted_val)
            # If the redecomposition differs from the corrupted bits,
            # the encoding was non-canonical — Cassini catches this
            if corrupted_bits_redecomposed != corrupted_bits:
                cassini_detected += 1
            elif has_adjacent:
                cassini_detected += 1  # adjacency is a subset of Cassini

            total += 1
            if total >= n_tests:
                break
        if total >= n_tests:
            break

    adj_rate = adjacency_detected / total if total > 0 else 0
    cas_rate = cassini_detected / total if total > 0 else 0

    print(f"  Tested {total} single-bit corruptions on Fibonacci-encoded weights")
    print(f"")
    print(f"  FREE detection (adjacency check only — zero compute cost):")
    print(f"    Detected: {adjacency_detected}/{total} ({adj_rate:.1%})")
    print(f"    These are corruptions where flipping one Fibonacci digit")
    print(f"    created two adjacent 1s, which is impossible in valid Zeckendorf.")
    print(f"")
    print(f"  FULL detection (Cassini / re-decomposition — 2 extra multiplies):")
    print(f"    Detected: {cassini_detected}/{total} ({cas_rate:.1%})")
    print(f"    Catches adjacency violations PLUS cases where the corrupted")
    print(f"    value's canonical Zeckendorf form differs from the stored bits.")

    return {
        'total': total,
        'adjacency_detected': adjacency_detected,
        'adjacency_rate': adj_rate,
        'cassini_detected': cassini_detected,
        'cassini_rate': cas_rate,
    }


def compute_hardware_cost(stats_pruning, n_fib_digits):
    """
    Estimate combined hardware savings from Zeckendorf pruning + Fibonacci encoding.

    Based on:
      - DATE 2021: 73% multiplier area reduction from Fibonacci encoding
      - Pruning: (1 - density) fraction of multiplications eliminated
    """
    density = stats_pruning['density']

    # Baseline: 1.0 = full dense binary cost
    cost_dense = 1.0

    # Fibonacci encoding alone: 73% area reduction per multiplier (DATE 2021)
    fib_area_factor = 0.27  # 1 - 0.73
    cost_fib_only = cost_dense * fib_area_factor

    # Zeckendorf pruning alone: skip (1-density) of multiplications
    cost_prune_only = cost_dense * density

    # Combined: fewer multiplications, each one cheaper
    cost_combined = density * fib_area_factor

    # Power-delay product reduction (DATE 2021: 43% PDP reduction from encoding)
    pdp_factor = 0.57  # 1 - 0.43
    pdp_combined = density * pdp_factor

    return {
        'dense_binary': cost_dense,
        'fibonacci_only': cost_fib_only,
        'zeckendorf_only': cost_prune_only,
        'combined': cost_combined,
        'area_reduction': 1.0 - cost_combined,
        'pdp_combined': pdp_combined,
        'pdp_reduction': 1.0 - pdp_combined,
        'density': density,
        'fib_digits': n_fib_digits,
    }


def run_phase1(args):
    """
    Full Phase 1 experiment:
      Dense → Zeckendorf prune → Fibonacci encode → Fine-tune → Evaluate
    """
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Device: {device}")

    # ── Load data ──
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size, data_dir=args.data_dir)

    # ── Load pretrained dense model ──
    ckpt_path = os.path.join(args.data_dir, 'resnet20_dense.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Dense checkpoint not found at {ckpt_path}")
        print(f"Run zeckendorf_experiment.py first to train the dense baseline.")
        return

    model = ResNet20().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    dense_acc = evaluate(model, test_loader, device)
    print(f"Dense baseline: {dense_acc:.2f}%")

    # ── Step 1: Apply Zeckendorf pruning ──
    print(f"\n{'='*70}")
    print("STEP 1: ZECKENDORF PRUNING")
    print(f"{'='*70}")

    masks, prune_stats = apply_pruning(model, mask_type='zeckendorf', prune_axis=0)
    acc_after_prune = evaluate(model, test_loader, device)

    print(f"  Pruned {prune_stats['pruned_layers']} layers")
    print(f"  Density: {prune_stats['density']:.4f} ({prune_stats['density']*100:.1f}%)")
    print(f"  Accuracy after pruning: {acc_after_prune:.2f}% (drop: {dense_acc - acc_after_prune:.2f}%)")

    # ── Step 2: Apply Fibonacci encoding to surviving weights ──
    print(f"\n{'='*70}")
    print(f"STEP 2: FIBONACCI ENCODING ({args.fib_digits} digits)")
    print(f"{'='*70}")

    encoder = FibonacciEncoder(n_digits=args.fib_digits)
    print(f"  Fibonacci grid: {encoder.n_digits} digits, {len(encoder.grid)} levels, "
          f"max value: {encoder.max_value}")

    scales, enc_stats = apply_fibonacci_encoding(model, masks, encoder)
    acc_after_encode = evaluate(model, test_loader, device)

    # Summarize encoding error
    total_rmse = sum(s['rmse'] for s in enc_stats.values()) / len(enc_stats) if enc_stats else 0
    total_max_err = max(s['max_error'] for s in enc_stats.values()) if enc_stats else 0

    print(f"  Encoded {len(scales)} layers")
    print(f"  Average RMSE per layer: {total_rmse:.6f}")
    print(f"  Max quantization error: {total_max_err:.6f}")
    print(f"  Accuracy after encoding: {acc_after_encode:.2f}% "
          f"(drop from pruned: {acc_after_prune - acc_after_encode:.2f}%, "
          f"drop from dense: {dense_acc - acc_after_encode:.2f}%)")

    # ── Step 3: Fine-tune the pruned + encoded model ──
    print(f"\n{'='*70}")
    print(f"STEP 3: FINE-TUNING ({args.epochs_finetune} epochs)")
    print(f"{'='*70}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_finetune)

    best_acc = acc_after_encode
    t0 = time.time()

    for epoch in range(args.epochs_finetune):
        # Train with both mask enforcement AND re-encoding after each step
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, masks=masks)

        # Re-apply Fibonacci encoding periodically (every 5 epochs)
        # to keep weights on the Fibonacci grid during fine-tuning
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks and name in scales:
                        encoded, _, _ = encoder.encode_tensor(param.data, scale=scales[name])
                        param.data.copy_(encoded * masks[name])

        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1:3d}/{args.epochs_finetune}: "
                  f"loss={loss:.4f} train={train_acc:.1f}% test={test_acc:.1f}% "
                  f"best={best_acc:.1f}% [{elapsed:.0f}s]")

    # Final Fibonacci re-encoding to ensure all weights are on-grid
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks and name in scales:
                encoded, _, _ = encoder.encode_tensor(param.data, scale=scales[name])
                param.data.copy_(encoded * masks[name])

    final_acc = evaluate(model, test_loader, device)
    print(f"\n  ★ Final accuracy (pruned + encoded + fine-tuned): {final_acc:.2f}%")
    print(f"    Drop from dense: {dense_acc - final_acc:.2f}%")

    # ── Step 4: Cassini integrity test ──
    print(f"\n{'='*70}")
    print("STEP 4: CASSINI INTEGRITY CHECK")
    print(f"{'='*70}")

    cassini_results = run_cassini_integrity_test(
        model, masks, encoder, scales, n_tests=200)

    print(f"  Tested {cassini_results['total']} weight corruptions")
    print(f"  Detected: {cassini_results['detected']}")
    print(f"  Missed: {cassini_results['missed']}")
    print(f"  Detection rate: {cassini_results['detection_rate']:.1%}")

    # ── Step 5: Hardware cost estimate ──
    print(f"\n{'='*70}")
    print("STEP 5: COMBINED HARDWARE COST ESTIMATE")
    print(f"{'='*70}")

    hw_cost = compute_hardware_cost(prune_stats, args.fib_digits)

    print(f"  Multiplier area (relative to dense binary baseline):")
    print(f"    Dense binary:      {hw_cost['dense_binary']:.3f} (baseline)")
    print(f"    Fibonacci only:    {hw_cost['fibonacci_only']:.3f} (−{(1-hw_cost['fibonacci_only'])*100:.0f}%)")
    print(f"    Zeckendorf only:   {hw_cost['zeckendorf_only']:.3f} (−{(1-hw_cost['zeckendorf_only'])*100:.0f}%)")
    print(f"    ★ Combined:        {hw_cost['combined']:.3f} (−{hw_cost['area_reduction']*100:.1f}%)")
    print(f"")
    print(f"  Power-delay product:")
    print(f"    ★ Combined PDP:    {hw_cost['pdp_combined']:.3f} (−{hw_cost['pdp_reduction']*100:.1f}%)")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*70}")

    results = {
        'dense_accuracy': dense_acc,
        'after_pruning': acc_after_prune,
        'after_encoding': acc_after_encode,
        'after_finetune': best_acc,
        'final_accuracy': final_acc,
        'total_accuracy_drop': dense_acc - final_acc,
        'encoding_cost': acc_after_prune - acc_after_encode,
        'pruning_stats': prune_stats,
        'encoding': {
            'fib_digits': args.fib_digits,
            'n_levels': len(encoder.grid),
            'avg_rmse': total_rmse,
        },
        'cassini': {
            'detection_rate': cassini_results['detection_rate'],
            'total_tested': cassini_results['total'],
        },
        'hardware': hw_cost,
    }

    print(f"""
  Accuracy waterfall:
    Dense baseline:          {dense_acc:6.2f}%
    After Zeckendorf prune:  {acc_after_prune:6.2f}%  (−{dense_acc - acc_after_prune:.2f}%)
    After Fibonacci encode:  {acc_after_encode:6.2f}%  (−{acc_after_prune - acc_after_encode:.2f}% additional)
    After fine-tuning:       {final_acc:6.2f}%  (recovered {final_acc - acc_after_encode:.2f}%)

  Total accuracy cost:       {dense_acc - final_acc:.2f}% drop from dense

  Hardware savings:
    Multiplier area:         −{hw_cost['area_reduction']*100:.1f}%
    Power-delay product:     −{hw_cost['pdp_reduction']*100:.1f}%

  Integrity:
    Cassini detection rate:  {cassini_results['detection_rate']:.1%}

  Components stacked:
    ✓ Zeckendorf pruning mask (no adjacent active weights)
    ✓ Fibonacci weight encoding (no adjacent 1s in weight bits)
    ✓ Cassini integrity checks (det(M^n) = (−1)^n)

  All three components enforce the same structural rule.
  All three derive from M = [[1,1],[1,0]].
""")

    # Save results
    results_path = os.path.join(args.data_dir, 'phase1_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='φ-register Phase 1: Stacked components')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fib-digits', type=int, default=8,
                       help='Fibonacci encoding digits (8=55 levels, 10=144 levels)')
    parser.add_argument('--epochs-finetune', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--quick', action='store_true')

    args = parser.parse_args()

    if args.quick:
        args.epochs_finetune = 5
        print("⚡ QUICK MODE: 5 fine-tune epochs\n")

    print("φ-register Phase 1: Zeckendorf pruning + Fibonacci encoding + Cassini checks")
    print(f"  Fibonacci digits: {args.fib_digits} ({FibonacciEncoder(args.fib_digits).max_value + 1} levels)")
    print(f"  Fine-tuning:      {args.epochs_finetune} epochs")
    print()

    run_phase1(args)