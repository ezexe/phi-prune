"""
φ-register experiment: Zeckendorf vs NVIDIA 2:4 structured sparsity
====================================================================

PURPOSE:
  Test whether pruning neural network weights with the Zeckendorf constraint
  ("no two adjacent weights active") preserves accuracy comparably to
  NVIDIA's 2:4 structured sparsity ("exactly 2 of every 4 weights active").

WHAT THIS TESTS:
  The core prediction from the φ-register framework: the adjacency constraint
  (same rule that governs the Q-matrix, Fibonacci coding, CSD encoding, and
  golden mean shift capacity) works as a structured pruning mask for neural
  networks. If it does, every component of the φ-register architecture speaks
  the same structural language.

SETUP:
  - Model: ResNet-20 on CIFAR-10 (small enough to iterate fast, large enough
    to be meaningful — standard benchmark in pruning literature)
  - Three conditions:
    1. DENSE    — full model, no pruning (baseline)
    2. ZECK     — Zeckendorf-constrained masks (no consecutive active weights)
    3. TWO_FOUR — NVIDIA 2:4 masks (exactly 2 of every 4 active)
  - Procedure: train dense → prune → fine-tune → evaluate
  - Primary metric: top-1 accuracy after fine-tuning
  - Secondary metrics: actual sparsity, parameter count, FLOPs estimate

USAGE:
  python zeckendorf_experiment.py [--device cuda] [--epochs-dense 160] [--epochs-finetune 40]

  For a quick sanity check (few epochs, won't reach full accuracy but will
  show whether the pipeline works and the relative ordering is reasonable):
  python zeckendorf_experiment.py --quick

REQUIREMENTS:
  pip install torch torchvision
"""

import argparse
import time
import copy
import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# ═══════════════════════════════════════════════════════════════════════
# 1. MASK GENERATION — the core contribution
# ═══════════════════════════════════════════════════════════════════════

def generate_zeckendorf_mask(weight_tensor, axis=0):
    """
    Generate a Zeckendorf-compliant pruning mask for a weight tensor.

    The rule: along the specified axis, no two adjacent positions can both
    be active (1). This is the (1,∞)-RLL constraint, identical to the
    "no consecutive 1s" rule in Zeckendorf representation.

    Algorithm: magnitude-based greedy selection with adjacency repair.
    1. Score each position by the sum of absolute weight values along that axis
    2. Sort positions by score (descending — keep the most important ones)
    3. Greedily accept each position unless it would create two adjacent actives

    This is equivalent to finding a maximum-weight independent set on a path
    graph, which has an O(n) dynamic programming solution. We use the greedy
    approach for clarity; it gives near-optimal results for smooth weight
    distributions.

    Args:
        weight_tensor: the weight matrix (any shape, but at least 2D)
        axis: which dimension to apply the adjacency constraint along
              (typically 0 = output channels, 1 = input channels)

    Returns:
        mask: binary tensor of same shape as weight_tensor
    """
    # Compute importance score for each position along the axis
    # Score = L1 norm of the weight slice at that position
    dims_to_reduce = list(range(weight_tensor.dim()))
    dims_to_reduce.remove(axis)
    if len(dims_to_reduce) == 0:
        # 1D tensor — score is just absolute value
        scores = weight_tensor.abs()
    else:
        scores = weight_tensor.abs().sum(dim=dims_to_reduce)

    n = scores.shape[0]

    # --- Dynamic programming for maximum-weight independent set on path graph ---
    # This is the optimal algorithm, not just greedy.
    # dp[i][0] = best score using positions 0..i where position i is NOT active
    # dp[i][1] = best score using positions 0..i where position i IS active

    scores_np = scores.detach().cpu().numpy()

    dp = [[0.0, 0.0] for _ in range(n)]
    dp[0][0] = 0.0
    dp[0][1] = float(scores_np[0])

    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1])          # don't take i
        dp[i][1] = dp[i-1][0] + float(scores_np[i])      # take i (prev must be off)

    # Backtrack to find the optimal mask
    active = [0] * n
    if dp[n-1][1] >= dp[n-1][0]:
        active[n-1] = 1
        choice = 1
    else:
        choice = 0

    for i in range(n-2, -1, -1):
        if choice == 1:
            # Current position is active, previous must be inactive
            choice = 0
        else:
            # Current position is inactive, previous takes the better option
            if dp[i][1] >= dp[i][0]:
                active[i] = 1
                choice = 1
            else:
                choice = 0

    # Build the full mask by broadcasting the 1D active pattern
    mask_1d = torch.tensor(active, dtype=weight_tensor.dtype, device=weight_tensor.device)

    # Reshape mask_1d to broadcast across all other dimensions
    shape = [1] * weight_tensor.dim()
    shape[axis] = n
    mask = mask_1d.view(shape).expand_as(weight_tensor)

    return mask


def generate_24_mask(weight_tensor, axis=0):
    """
    Generate an NVIDIA 2:4 structured sparsity mask.

    The rule: along the specified axis, divide positions into groups of 4.
    In each group, keep exactly the 2 positions with highest magnitude.

    If the axis length isn't divisible by 4, the remainder positions at
    the end are kept active (no pruning for the partial group).

    Args:
        weight_tensor: the weight matrix
        axis: which dimension to apply the constraint along

    Returns:
        mask: binary tensor of same shape as weight_tensor
    """
    # Compute importance scores along the axis
    dims_to_reduce = list(range(weight_tensor.dim()))
    dims_to_reduce.remove(axis)
    if len(dims_to_reduce) == 0:
        scores = weight_tensor.abs()
    else:
        scores = weight_tensor.abs().sum(dim=dims_to_reduce)

    n = scores.shape[0]
    active = [0] * n

    for group_start in range(0, n, 4):
        group_end = min(group_start + 4, n)
        group_size = group_end - group_start

        if group_size < 4:
            # Partial group at the end — keep all
            for j in range(group_start, group_end):
                active[j] = 1
        else:
            # Find the 2 largest in this group of 4
            group_scores = [(float(scores[group_start + j]), j) for j in range(4)]
            group_scores.sort(reverse=True)
            for _, j in group_scores[:2]:
                active[group_start + j] = 1

    mask_1d = torch.tensor(active, dtype=weight_tensor.dtype, device=weight_tensor.device)
    shape = [1] * weight_tensor.dim()
    shape[axis] = n
    mask = mask_1d.view(shape).expand_as(weight_tensor)

    return mask


def verify_zeckendorf_mask(mask_1d):
    """Verify no two adjacent positions are both active. Returns True if valid."""
    for i in range(len(mask_1d) - 1):
        if mask_1d[i] == 1 and mask_1d[i+1] == 1:
            return False
    return True


def verify_24_mask(mask_1d):
    """Verify exactly 2 of every 4 positions are active. Returns True if valid."""
    n = len(mask_1d)
    for i in range(0, n - 3, 4):
        if sum(mask_1d[i:i+4]) != 2:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════
# 2. MODEL — ResNet-20 for CIFAR-10
# ═══════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """Standard ResNet basic block with two 3x3 convolutions."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """
    ResNet-20 for CIFAR-10.
    3 groups of 3 blocks each, widths [16, 32, 64].
    ~270K parameters. Standard benchmark for pruning experiments.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)

        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ═══════════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def get_cifar10_loaders(batch_size=128, num_workers=2, data_dir='./data'):
    """Standard CIFAR-10 data loading with standard augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════
# 4. TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, masks=None):
    """Train for one epoch. If masks provided, re-apply after each optimizer step."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Re-zero pruned weights after optimizer step
        # This is the standard approach in structured pruning:
        # the mask is fixed, gradients flow through active weights only
        if masks:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.mul_(masks[name])

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model accuracy on a data loader."""
    model.eval()
    correct, total = 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)

    return 100.0 * correct / total


# ═══════════════════════════════════════════════════════════════════════
# 5. PRUNING — apply masks to a trained model
# ═══════════════════════════════════════════════════════════════════════

def apply_pruning(model, mask_type='zeckendorf', prune_axis=0, min_dim=8):
    """
    Apply structured pruning masks to all Conv2d layers in the model.

    Args:
        model: trained model to prune
        mask_type: 'zeckendorf' or '2:4'
        prune_axis: axis to apply constraint along (0 = output channels)
        min_dim: minimum dimension size to prune (skip tiny layers)

    Returns:
        masks: dict mapping parameter names to mask tensors
        stats: dict with pruning statistics
    """
    masks = {}
    stats = {'total_params': 0, 'active_params': 0, 'pruned_layers': 0, 'skipped_layers': 0}

    mask_fn = generate_zeckendorf_mask if mask_type == 'zeckendorf' else generate_24_mask

    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name and param.dim() == 4:
            # Only prune conv layers with sufficient dimension
            if param.shape[prune_axis] >= min_dim:
                mask = mask_fn(param.data, axis=prune_axis)
                masks[name] = mask

                # Apply mask immediately
                param.data.mul_(mask)

                n_total = param.numel()
                n_active = mask.sum().item()
                stats['total_params'] += n_total
                stats['active_params'] += n_active
                stats['pruned_layers'] += 1

                # Verify mask validity on the 1D pattern
                mask_1d = mask.select(prune_axis, 0)
                while mask_1d.dim() > 1:
                    mask_1d = mask_1d[0]
                # The actual 1D pattern along the prune axis
                active_pattern = []
                for i in range(param.shape[prune_axis]):
                    slc = [slice(None)] * param.dim()
                    slc[prune_axis] = i
                    active_pattern.append(int(mask[tuple(slc)].flatten()[0].item()))

                if mask_type == 'zeckendorf':
                    valid = verify_zeckendorf_mask(active_pattern)
                else:
                    valid = verify_24_mask(active_pattern)

                if not valid:
                    print(f"  ⚠ WARNING: {name} mask INVALID! Pattern: {active_pattern[:20]}...")
            else:
                stats['skipped_layers'] += 1
                stats['total_params'] += param.numel()
                stats['active_params'] += param.numel()

    # Also count unpruned params (batchnorm, fc, biases)
    for name, param in model.named_parameters():
        if name not in masks:
            stats['total_params'] += param.numel()
            stats['active_params'] += param.numel()

    stats['sparsity'] = 1.0 - stats['active_params'] / stats['total_params']
    stats['density'] = stats['active_params'] / stats['total_params']

    return masks, stats


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(args):
    """Run the complete experiment: train dense → prune → fine-tune → compare."""

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Device: {device}")
    print(f"{'='*70}")

    # ── Data ──
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size, data_dir=args.data_dir)
    print(f"CIFAR-10 loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    # ── Phase 1: Train dense baseline ──
    print(f"\n{'='*70}")
    print("PHASE 1: DENSE BASELINE TRAINING")
    print(f"{'='*70}")

    model_dense = ResNet20().to(device)
    n_params = sum(p.numel() for p in model_dense.parameters())
    print(f"ResNet-20: {n_params:,} parameters")

    # Check if a pretrained checkpoint exists
    ckpt_path = os.path.join(args.data_dir, 'resnet20_dense.pt')
    if os.path.exists(ckpt_path) and not args.retrain:
        print(f"Loading pretrained dense model from {ckpt_path}")
        model_dense.load_state_dict(torch.load(ckpt_path, map_location=device))
        dense_acc = evaluate(model_dense, test_loader, device)
        print(f"Dense baseline accuracy: {dense_acc:.2f}%")
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_dense.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

        best_acc = 0
        t0 = time.time()
        for epoch in range(args.epochs_dense):
            loss, train_acc = train_epoch(model_dense, train_loader, optimizer, criterion, device)
            test_acc = evaluate(model_dense, test_loader, device)
            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model_dense.state_dict(), ckpt_path)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1:3d}/{args.epochs_dense}: "
                      f"loss={loss:.4f} train={train_acc:.1f}% test={test_acc:.1f}% "
                      f"best={best_acc:.1f}% [{elapsed:.0f}s]")

        model_dense.load_state_dict(torch.load(ckpt_path, map_location=device))
        dense_acc = evaluate(model_dense, test_loader, device)
        print(f"\n  Dense baseline: {dense_acc:.2f}% ({time.time()-t0:.0f}s total)")

    dense_acc = evaluate(model_dense, test_loader, device)

    # ── Phase 2: Prune and fine-tune each method ──
    results = {'dense': {'accuracy': dense_acc, 'sparsity': 0.0, 'density': 1.0}}

    for method_name, mask_type in [('zeckendorf', 'zeckendorf'), ('2:4', '2:4')]:
        print(f"\n{'='*70}")
        print(f"PHASE 2: {method_name.upper()} PRUNING")
        print(f"{'='*70}")

        # Deep copy the trained dense model
        model = copy.deepcopy(model_dense)

        # Apply pruning masks
        masks, stats = apply_pruning(model, mask_type=mask_type, prune_axis=0)

        print(f"  Pruned layers: {stats['pruned_layers']}")
        print(f"  Skipped layers: {stats['skipped_layers']}")
        print(f"  Active params: {stats['active_params']:,.0f} / {stats['total_params']:,.0f}")
        print(f"  Density: {stats['density']:.4f} ({stats['density']*100:.1f}%)")
        print(f"  Sparsity: {stats['sparsity']:.4f} ({stats['sparsity']*100:.1f}%)")

        # Accuracy immediately after pruning (before fine-tuning)
        acc_before_ft = evaluate(model, test_loader, device)
        print(f"  Accuracy after pruning (before fine-tune): {acc_before_ft:.2f}%")
        print(f"  Accuracy drop from dense: {dense_acc - acc_before_ft:.2f}%")

        # Fine-tune with masks frozen
        print(f"\n  Fine-tuning for {args.epochs_finetune} epochs...")
        criterion = nn.CrossEntropyLoss()
        # Lower learning rate for fine-tuning
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_finetune)

        best_acc = acc_before_ft
        t0 = time.time()
        for epoch in range(args.epochs_finetune):
            loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, masks=masks)
            test_acc = evaluate(model, test_loader, device)
            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc

            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed = time.time() - t0
                print(f"    Epoch {epoch+1:3d}/{args.epochs_finetune}: "
                      f"loss={loss:.4f} train={train_acc:.1f}% test={test_acc:.1f}% "
                      f"best={best_acc:.1f}% [{elapsed:.0f}s]")

        print(f"\n  ★ {method_name.upper()} RESULT: {best_acc:.2f}% "
              f"(drop from dense: {dense_acc - best_acc:.2f}%, "
              f"density: {stats['density']*100:.1f}%)")

        results[method_name] = {
            'accuracy': best_acc,
            'accuracy_before_ft': acc_before_ft,
            'sparsity': stats['sparsity'],
            'density': stats['density'],
            'active_params': stats['active_params'],
            'total_params': stats['total_params'],
            'pruned_layers': stats['pruned_layers'],
        }

    # ── Phase 3: Summary ──
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'Method':<14} {'Accuracy':>10} {'Drop':>8} {'Density':>10} {'Active Params':>14}")
    print(f"  {'─'*14} {'─'*10} {'─'*8} {'─'*10} {'─'*14}")

    for name in ['dense', 'zeckendorf', '2:4']:
        r = results[name]
        drop = dense_acc - r['accuracy']
        active = r.get('active_params', n_params)
        total = r.get('total_params', n_params)
        print(f"  {name:<14} {r['accuracy']:>9.2f}% {drop:>+7.2f}% "
              f"{r['density']:>9.1%} {active:>10,.0f}/{total:,.0f}")

    # The key comparison
    z = results['zeckendorf']
    t = results['2:4']
    print(f"\n  ★ Zeckendorf vs 2:4:")
    print(f"    Accuracy gap: {z['accuracy'] - t['accuracy']:+.2f}% "
          f"({'Zeckendorf better' if z['accuracy'] > t['accuracy'] else '2:4 better' if t['accuracy'] > z['accuracy'] else 'tied'})")
    print(f"    Density:  Zeckendorf={z['density']:.1%} vs 2:4={t['density']:.1%}")
    print(f"    Zeckendorf uses {(1 - z['density']/t['density'])*100:.1f}% fewer active parameters")

    # The falsifiable prediction
    acc_gap = abs(z['accuracy'] - t['accuracy'])
    print(f"\n  PREDICTION TEST: accuracy within 2% of each other?")
    print(f"    Gap: {acc_gap:.2f}% → {'✓ CONFIRMED' if acc_gap <= 2.0 else '✗ FALSIFIED'}")

    # Efficiency metric: accuracy per active parameter
    z_eff = z['accuracy'] / (z['active_params'] / 1000)
    t_eff = t['accuracy'] / (t['active_params'] / 1000)
    print(f"\n  Accuracy per 1K active params:")
    print(f"    Zeckendorf: {z_eff:.4f}")
    print(f"    2:4:        {t_eff:.4f}")
    print(f"    Ratio: {z_eff/t_eff:.3f}x")

    # Save results
    results_path = os.path.join(args.data_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='φ-register: Zeckendorf vs 2:4 structured sparsity')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--epochs-dense', type=int, default=160, help='Dense training epochs')
    parser.add_argument('--epochs-finetune', type=int, default=40, help='Fine-tuning epochs after pruning')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--retrain', action='store_true', help='Force retrain dense model')
    parser.add_argument('--quick', action='store_true', help='Quick run (10 epochs each) for sanity checking')

    args = parser.parse_args()

    if args.quick:
        args.epochs_dense = 10
        args.epochs_finetune = 5
        print("⚡ QUICK MODE: 10 dense epochs + 5 fine-tune epochs")
        print("   (won't reach full accuracy — just validates the pipeline)\n")

    os.makedirs(args.data_dir, exist_ok=True)

    print("φ-register experiment: Zeckendorf vs NVIDIA 2:4 structured sparsity")
    print(f"  Dense training:  {args.epochs_dense} epochs")
    print(f"  Fine-tuning:     {args.epochs_finetune} epochs per method")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Data directory:  {args.data_dir}")
    print()

    results = run_experiment(args)