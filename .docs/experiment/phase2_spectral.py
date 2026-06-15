"""
φ-register Phase 2: Activation Spectral Analysis
==================================================

WHAT THIS TESTS:
  The Q-matrix SSM analysis predicted that the stabilized M/φ kernel is a
  low-pass filter, structurally suited to signals where local correlations
  dominate. If Zeckendorf-pruned weight matrices (with their enforced gaps
  between active positions) produce activations that are measurably more
  low-pass than dense or 2:4-pruned activations, then the Q-matrix kernel
  is the RIGHT filter for post-processing these signals.

  This is the bridge between the weight-side work (Phases 0-1) and the
  signal-side theory (Q-matrix SSM from Session 1). If the spectral
  prediction holds, Phase 3 (kernel integration) is motivated. If not,
  the signal-side theory needs revision.

METHOD:
  1. Run inference on all three model variants (dense, Zeckendorf, 2:4)
  2. Hook into every Conv2d layer and record output activations
  3. Compute 2D FFT power spectrum of each activation map
  4. Measure the ratio of low-frequency to high-frequency energy
  5. Compare across the three model variants

  No training. No new architecture. Just measurement.

PREDICTION:
  Zeckendorf-pruned activations will have HIGHER low-frequency energy ratio
  than both dense and 2:4 activations, because the adjacency constraint on
  weight positions suppresses high-frequency spatial patterns in the outputs.

USAGE:
  As standalone (after Phase 0 has run):
    python phase2_spectral.py --data-dir ./data --device cuda

  As Colab cell: paste the analyze_spectra() function and call it with
  the models already in memory from earlier cells.

REQUIREMENTS:
  - Trained models from Phase 0 (dense, Zeckendorf-pruned, 2:4-pruned)
  - numpy, torch, torchvision (matplotlib optional, for plots)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import json
import os
import time

# ═══════════════════════════════════════════════════════════════════════
# 1. ACTIVATION HOOKS — capture intermediate layer outputs
# ═══════════════════════════════════════════════════════════════════════

class ActivationRecorder:
    """
    Attach forward hooks to all Conv2d layers in a model.
    Records output activations for spectral analysis.

    Usage:
        recorder = ActivationRecorder(model)
        model(input_batch)  # forward pass populates recorder.activations
        spectra = recorder.compute_spectra()
        recorder.clear()
    """

    def __init__(self, model):
        self.activations = {}
        self.hooks = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name):
        """Create a hook that stores the activation tensor for a named layer."""
        def hook_fn(module, input, output):
            # Store on CPU to save GPU memory — we're collecting many layers
            self.activations[name] = output.detach().cpu()
        return hook_fn

    def clear(self):
        """Clear stored activations (call between batches to save memory)."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all hooks from the model."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_spectra(self):
        """
        Compute the 2D power spectrum of each stored activation map.

        For each layer's output [batch, channels, H, W]:
          1. Compute 2D FFT per channel per sample
          2. Take magnitude squared (power spectrum)
          3. Average across batch and channels
          4. Compute radial profile (energy as function of spatial frequency)
          5. Compute low-frequency energy ratio

        Returns dict mapping layer names to spectral statistics.
        """
        results = {}

        for name, act in self.activations.items():
            # act shape: [batch, channels, H, W]
            if act.dim() != 4 or act.shape[2] < 4 or act.shape[3] < 4:
                continue  # skip tiny activations

            B, C, H, W = act.shape

            # 2D FFT of each activation map, shifted so DC is at center
            fft2d = torch.fft.fft2(act.float())
            fft2d_shifted = torch.fft.fftshift(fft2d, dim=(-2, -1))
            power = (fft2d_shifted.abs() ** 2)

            # Average power spectrum across batch and channels
            avg_power = power.mean(dim=(0, 1))  # shape: [H, W]

            # Compute radial frequency profile
            # Distance of each frequency bin from DC (center)
            cy, cx = H // 2, W // 2
            y_freq = torch.arange(H).float() - cy
            x_freq = torch.arange(W).float() - cx
            yy, xx = torch.meshgrid(y_freq, x_freq, indexing='ij')
            radius = torch.sqrt(xx**2 + yy**2)

            # Maximum radius (Nyquist)
            max_radius = min(cy, cx)

            # Bin the power by radial frequency
            n_bins = max(4, max_radius)
            bin_edges = torch.linspace(0, max_radius, int(n_bins) + 1)
            radial_power = []

            for bi in range(len(bin_edges) - 1):
                mask = (radius >= bin_edges[bi]) & (radius < bin_edges[bi + 1])
                if mask.sum() > 0:
                    radial_power.append(float(avg_power[mask].mean()))
                else:
                    radial_power.append(0.0)

            radial_power = np.array(radial_power)

            # Low-frequency energy ratio:
            # fraction of total energy in the lowest 25% of frequency bins
            total_energy = float(avg_power.sum())
            if total_energy < 1e-10:
                lf_ratio = 0.0
            else:
                # Low-freq mask: radius < 25% of max
                lf_mask = radius < (max_radius * 0.25)
                lf_energy = float(avg_power[lf_mask].sum())
                lf_ratio = lf_energy / total_energy

            # Mid-frequency ratio (25%-50% of max radius)
            mf_mask = (radius >= max_radius * 0.25) & (radius < max_radius * 0.5)
            mf_energy = float(avg_power[mf_mask].sum()) / total_energy if total_energy > 1e-10 else 0

            # High-frequency ratio (>50% of max radius)
            hf_mask = radius >= max_radius * 0.5
            hf_energy = float(avg_power[hf_mask].sum()) / total_energy if total_energy > 1e-10 else 0

            # Spectral centroid: average frequency weighted by power
            if total_energy > 1e-10:
                spectral_centroid = float((avg_power * radius).sum() / avg_power.sum())
            else:
                spectral_centroid = 0.0

            # Spectral rolloff: frequency below which 85% of energy lies
            cumulative = torch.zeros_like(radius)
            sorted_radii = torch.sort(radius.flatten())[0]
            sorted_power = avg_power.flatten()[torch.argsort(radius.flatten())]
            cum_power = torch.cumsum(sorted_power, dim=0)
            if total_energy > 1e-10:
                rolloff_idx = (cum_power >= total_energy * 0.85).nonzero()
                rolloff_freq = float(sorted_radii[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(max_radius)
            else:
                rolloff_freq = 0.0

            results[name] = {
                'shape': [B, C, H, W],
                'lf_ratio': lf_ratio,       # low-freq energy fraction (< 25% radius)
                'mf_ratio': mf_energy,       # mid-freq energy fraction (25-50% radius)
                'hf_ratio': hf_energy,       # high-freq energy fraction (> 50% radius)
                'spectral_centroid': spectral_centroid,  # mean frequency
                'spectral_rolloff_85': rolloff_freq,     # 85% energy cutoff
                'max_radius': float(max_radius),
                'radial_profile': radial_power.tolist(),
                'total_energy': total_energy,
            }

        return results


# ═══════════════════════════════════════════════════════════════════════
# 2. SPECTRAL COMPARISON — run all three models and compare
# ═══════════════════════════════════════════════════════════════════════

def analyze_spectra(model_dense, model_zeck, model_24,
                    masks_zeck, masks_24,
                    test_loader, device, n_batches=5):
    """
    Run spectral analysis on all three model variants.

    Feeds the same input batches through each model, records activations
    at every Conv2d layer, computes power spectra, and compares the
    low-frequency energy ratios.

    Args:
        model_dense: fully-trained dense model
        model_zeck: Zeckendorf-pruned (and optionally Fibonacci-encoded) model
        model_24: 2:4-pruned model
        masks_zeck, masks_24: pruning masks (for reference, not used in analysis)
        test_loader: CIFAR-10 test data
        device: torch device
        n_batches: how many batches to average over (more = smoother, slower)

    Returns:
        comparison: dict with per-layer spectral statistics for each model variant
    """
    models = {
        'dense': model_dense,
        'zeckendorf': model_zeck,
        '2:4': model_24,
    }

    all_spectra = {name: {} for name in models}

    # Collect input batches (same data for all models)
    batches = []
    for i, (inputs, targets) in enumerate(test_loader):
        if i >= n_batches:
            break
        batches.append(inputs.to(device))

    print(f"  Analyzing {len(batches)} batches × {batches[0].shape[0]} samples each")

    for model_name, model in models.items():
        model.eval()
        model.to(device)
        recorder = ActivationRecorder(model)

        # Accumulate spectral stats across batches
        layer_stats_accum = {}

        with torch.no_grad():
            for batch in batches:
                _ = model(batch)
                batch_spectra = recorder.compute_spectra()

                for layer_name, stats in batch_spectra.items():
                    if layer_name not in layer_stats_accum:
                        layer_stats_accum[layer_name] = {
                            'lf_ratios': [],
                            'mf_ratios': [],
                            'hf_ratios': [],
                            'centroids': [],
                            'rolloffs': [],
                            'shape': stats['shape'],
                        }
                    layer_stats_accum[layer_name]['lf_ratios'].append(stats['lf_ratio'])
                    layer_stats_accum[layer_name]['mf_ratios'].append(stats['mf_ratio'])
                    layer_stats_accum[layer_name]['hf_ratios'].append(stats['hf_ratio'])
                    layer_stats_accum[layer_name]['centroids'].append(stats['spectral_centroid'])
                    layer_stats_accum[layer_name]['rolloffs'].append(stats['spectral_rolloff_85'])

                recorder.clear()

        # Average across batches
        for layer_name, accum in layer_stats_accum.items():
            all_spectra[model_name][layer_name] = {
                'lf_ratio': np.mean(accum['lf_ratios']),
                'mf_ratio': np.mean(accum['mf_ratios']),
                'hf_ratio': np.mean(accum['hf_ratios']),
                'centroid': np.mean(accum['centroids']),
                'rolloff': np.mean(accum['rolloffs']),
                'shape': accum['shape'],
            }

        recorder.remove_hooks()
        print(f"    {model_name}: {len(layer_stats_accum)} layers analyzed")

    return all_spectra


def print_spectral_comparison(all_spectra):
    """
    Print a formatted comparison of spectral properties across models.
    Tests the prediction: Zeckendorf activations are more low-pass.
    """
    print(f"\n{'='*70}")
    print("SPECTRAL COMPARISON: LOW-FREQUENCY ENERGY RATIO PER LAYER")
    print(f"{'='*70}")
    print(f"  Higher LF ratio = more low-pass = more local correlation")
    print(f"  Prediction: Zeckendorf > 2:4 ≥ Dense\n")

    # Get common layers
    layers = sorted(all_spectra['dense'].keys())

    print(f"  {'Layer':<35} {'Dense':>8} {'Zeck':>8} {'2:4':>8} {'Z>D?':>6} {'Z>T?':>6}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

    n_zeck_wins_dense = 0
    n_zeck_wins_24 = 0
    n_layers = 0

    # Accumulators for global averages
    dense_lf_total, zeck_lf_total, t24_lf_total = [], [], []
    dense_hf_total, zeck_hf_total, t24_hf_total = [], [], []
    dense_centroid_total, zeck_centroid_total, t24_centroid_total = [], [], []

    for layer in layers:
        if layer not in all_spectra['zeckendorf'] or layer not in all_spectra['2:4']:
            continue

        d = all_spectra['dense'][layer]
        z = all_spectra['zeckendorf'][layer]
        t = all_spectra['2:4'][layer]

        z_wins_d = '✓' if z['lf_ratio'] > d['lf_ratio'] else '✗'
        z_wins_t = '✓' if z['lf_ratio'] > t['lf_ratio'] else '✗'

        if z['lf_ratio'] > d['lf_ratio']: n_zeck_wins_dense += 1
        if z['lf_ratio'] > t['lf_ratio']: n_zeck_wins_24 += 1
        n_layers += 1

        dense_lf_total.append(d['lf_ratio'])
        zeck_lf_total.append(z['lf_ratio'])
        t24_lf_total.append(t['lf_ratio'])

        dense_hf_total.append(d['hf_ratio'])
        zeck_hf_total.append(z['hf_ratio'])
        t24_hf_total.append(t['hf_ratio'])

        dense_centroid_total.append(d['centroid'])
        zeck_centroid_total.append(z['centroid'])
        t24_centroid_total.append(t['centroid'])

        # Truncate long layer names for display
        short_name = layer if len(layer) <= 35 else '...' + layer[-32:]
        print(f"  {short_name:<35} {d['lf_ratio']:>7.4f} {z['lf_ratio']:>7.4f} "
              f"{t['lf_ratio']:>7.4f}  {z_wins_d:>4}  {z_wins_t:>4}")

    # Global averages
    print(f"\n  {'AVERAGE':<35} {np.mean(dense_lf_total):>7.4f} {np.mean(zeck_lf_total):>7.4f} "
          f"{np.mean(t24_lf_total):>7.4f}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Low-frequency energy ratio (fraction of energy below 25% of Nyquist):")
    print(f"    Dense:      {np.mean(dense_lf_total):.4f} ± {np.std(dense_lf_total):.4f}")
    print(f"    Zeckendorf: {np.mean(zeck_lf_total):.4f} ± {np.std(zeck_lf_total):.4f}")
    print(f"    2:4:        {np.mean(t24_lf_total):.4f} ± {np.std(t24_lf_total):.4f}")

    print(f"\n  High-frequency energy ratio (fraction of energy above 50% of Nyquist):")
    print(f"    Dense:      {np.mean(dense_hf_total):.4f}")
    print(f"    Zeckendorf: {np.mean(zeck_hf_total):.4f}")
    print(f"    2:4:        {np.mean(t24_hf_total):.4f}")

    print(f"\n  Spectral centroid (mean frequency — lower = more low-pass):")
    print(f"    Dense:      {np.mean(dense_centroid_total):.3f}")
    print(f"    Zeckendorf: {np.mean(zeck_centroid_total):.3f}")
    print(f"    2:4:        {np.mean(t24_centroid_total):.3f}")

    print(f"\n  Layers where Zeckendorf is more low-pass than Dense: "
          f"{n_zeck_wins_dense}/{n_layers}")
    print(f"  Layers where Zeckendorf is more low-pass than 2:4:   "
          f"{n_zeck_wins_24}/{n_layers}")

    # The falsifiable prediction
    zeck_avg = np.mean(zeck_lf_total)
    dense_avg = np.mean(dense_lf_total)
    t24_avg = np.mean(t24_lf_total)

    print(f"\n  ★ PREDICTION TEST: Zeckendorf LF ratio > Dense LF ratio?")
    if zeck_avg > dense_avg:
        print(f"    {zeck_avg:.4f} > {dense_avg:.4f} → ✓ CONFIRMED")
        print(f"    Zeckendorf activations are {((zeck_avg/dense_avg)-1)*100:.1f}% more low-pass")
    else:
        print(f"    {zeck_avg:.4f} ≤ {dense_avg:.4f} → ✗ FALSIFIED")
        print(f"    Zeckendorf activations are NOT more low-pass than dense")

    print(f"\n  ★ PREDICTION TEST: Zeckendorf LF ratio > 2:4 LF ratio?")
    if zeck_avg > t24_avg:
        print(f"    {zeck_avg:.4f} > {t24_avg:.4f} → ✓ CONFIRMED")
        print(f"    Zeckendorf activations are {((zeck_avg/t24_avg)-1)*100:.1f}% more low-pass than 2:4")
    else:
        print(f"    {zeck_avg:.4f} ≤ {t24_avg:.4f} → ✗ NOT CONFIRMED")
        diff = ((t24_avg/zeck_avg)-1)*100 if zeck_avg > 0 else 0
        print(f"    2:4 is {diff:.1f}% more low-pass (or they're comparable)")

    # Interpretation for Phase 3 decision
    print(f"\n{'='*70}")
    print("PHASE 3 DECISION")
    print(f"{'='*70}")

    if zeck_avg > dense_avg * 1.01:  # at least 1% more low-pass
        print(f"""
  The Zeckendorf-pruned network produces measurably more low-pass
  activations than the dense baseline. This validates the Q-matrix
  kernel theory: the stabilized M/φ filter (a decaying Fibonacci
  low-pass filter) is structurally matched to these activations.

  → Phase 3 is MOTIVATED: integrate the Q-matrix kernel as a
    signal-conditioning layer after Zeckendorf-pruned convolutions.
""")
    elif abs(zeck_avg - dense_avg) / max(dense_avg, 1e-10) < 0.01:
        print(f"""
  The spectral profiles are essentially identical across all three
  model variants. The adjacency constraint on weight positions does
  NOT produce a measurably different spectral signature in activations.

  → Phase 3 needs REVISION: the Q-matrix kernel cannot be justified
    by a spectral matching argument. The structural coherence is real
    at the weight level but does not propagate to the signal level.
""")
    else:
        print(f"""
  The results are mixed or show the opposite of the prediction.
  Further analysis needed before Phase 3.
""")

    return {
        'lf_ratio': {
            'dense': float(np.mean(dense_lf_total)),
            'zeckendorf': float(np.mean(zeck_lf_total)),
            '2:4': float(np.mean(t24_lf_total)),
        },
        'hf_ratio': {
            'dense': float(np.mean(dense_hf_total)),
            'zeckendorf': float(np.mean(zeck_hf_total)),
            '2:4': float(np.mean(t24_hf_total)),
        },
        'centroid': {
            'dense': float(np.mean(dense_centroid_total)),
            'zeckendorf': float(np.mean(zeck_centroid_total)),
            '2:4': float(np.mean(t24_centroid_total)),
        },
        'zeck_wins_vs_dense': n_zeck_wins_dense,
        'zeck_wins_vs_24': n_zeck_wins_24,
        'total_layers': n_layers,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. STANDALONE RUNNER (for use outside Colab)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='φ-register Phase 2: Spectral Analysis')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-batches', type=int, default=5,
                       help='Number of test batches to analyze (more = smoother)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"φ-register Phase 2: Activation Spectral Analysis")
    print(f"Device: {device}\n")

    # Import model and pruning utilities from Phase 0
    from Zeckendorf import (
        ResNet20, get_cifar10_loaders,
        apply_pruning, evaluate
    )

    # Load data
    _, test_loader = get_cifar10_loaders(batch_size=128, data_dir=args.data_dir)

    # Load dense model
    ckpt_path = os.path.join(args.data_dir, 'resnet20_dense.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint at {ckpt_path}. Run Phase 0 first.")
        exit(1)

    model_dense = ResNet20().to(device)
    model_dense.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    dense_acc = evaluate(model_dense, test_loader, device)
    print(f"Dense baseline: {dense_acc:.2f}%")

    # Create pruned variants
    model_zeck = copy.deepcopy(model_dense)
    masks_zeck, stats_z = apply_pruning(model_zeck, mask_type='zeckendorf')
    zeck_acc = evaluate(model_zeck, test_loader, device)
    print(f"Zeckendorf pruned: {zeck_acc:.2f}% (density: {stats_z['density']:.1%})")

    model_24 = copy.deepcopy(model_dense)
    masks_24, stats_t = apply_pruning(model_24, mask_type='2:4')
    t24_acc = evaluate(model_24, test_loader, device)
    print(f"2:4 pruned: {t24_acc:.2f}% (density: {stats_t['density']:.1%})")

    # Run spectral analysis
    print(f"\n{'='*70}")
    print("SPECTRAL ANALYSIS")
    print(f"{'='*70}")

    all_spectra = analyze_spectra(
        model_dense, model_zeck, model_24,
        masks_zeck, masks_24,
        test_loader, device,
        n_batches=args.n_batches
    )

    summary = print_spectral_comparison(all_spectra)

    # Save results
    results_path = os.path.join(args.data_dir, 'phase2_spectral.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {results_path}")