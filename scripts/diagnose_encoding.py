"""
Diagnose why Fibonacci encoding breaks a pruned checkpoint.

After encoding every pruned layer with 10-digit codewords, ResNet-152 V2
scores exactly 10.00% on CIFAR-10 (README, "Baseline and encoding"). An
exact 10.00% is what NaN logits give: argmax of an all-NaN row is class 0,
and CIFAR-10's test set has 1000 images per class. The notebook evaluates
under fp16 autocast, whose largest finite value is 65504, so one candidate
cause is encoding error that grows activations past fp16's range; the
other is encoding error large enough to break the model in fp32 too.

This script separates the two. It loads a checkpoint saved by the
notebook's variant cells (state_dict, masks, metadata["arch"]) and prints:

  1. accuracy of the pruned and encoded models, in fp32 and under fp16
     autocast (CUDA only), with the number of test images whose logits
     are not finite
  2. per-block max |activation| on one batch, pruned vs encoded, and the
     first block whose output leaves fp16's range or goes non-finite
  3. the pruned layers with the largest encoding error, with the ratios
     that drive it (quantization step and largest weight, both over the
     layer's weight std)

Usage (Colab, after the notebook has saved checkpoints/):
    python scripts/diagnose_encoding.py checkpoints/resnet152_IMAGENET1K_V2_pruned.pt
"""

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from zeckendorf_prune.encoding import FibonacciEncoder

FP16_MAX = 65504.0
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_checkpoint(path, arch=None):
    import torchvision

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = arch or ckpt.get("metadata", {}).get("arch")
    if arch is None:
        raise SystemExit("checkpoint has no metadata['arch']: pass --arch")
    net = torchvision.models.get_model(arch, weights=None)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt["masks"], arch


def cifar_batches(root, batch, size, device, limit=None):
    """CIFAR-10 test set, preprocessed as the notebook's GPULoader does for eval."""
    import torchvision

    ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2)
    y = torch.as_tensor(ds.targets)
    if limit:
        x, y = x[:limit], y[:limit]
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    batches = []
    for i in range(0, len(y), batch):
        xb = x[i:i + batch].to(device).float().div_(255).sub_(mean).div_(std)
        xb = F.interpolate(xb, size=(size, size), mode="bilinear", align_corners=False)
        batches.append((xb.contiguous(memory_format=torch.channels_last), y[i:i + batch].to(device)))
    return batches


def encode_model(net, masks, encoder, axis=None):
    """A copy of net with every masked layer Fibonacci-encoded, and per-layer error stats."""
    enc = copy.deepcopy(net)
    params = dict(enc.named_parameters())
    stats = []
    with torch.no_grad():
        for name, mask in masks.items():
            p = params[name]
            mask = mask.to(p.device)
            keep = mask.bool()
            w = p[keep].double()
            encoded, scale, _ = encoder.encode_tensor(p.data, mask=mask, axis=axis)
            p.copy_(encoded)
            err = p[keep].double() - w
            std = w.std().item() if w.numel() > 1 else 0.0
            std = std or 1.0
            # coarsest step among kept weights (fully pruned channels carry a placeholder scale of 1)
            scales = torch.as_tensor(scale, dtype=torch.float64).expand(p.shape)[keep.cpu()]
            step = float((1.0 / scales).max()) if scales.numel() else 0.0
            stats.append({
                "layer": name,
                "rel_err": (err.norm() / w.norm().clamp_min(1e-30)).item(),
                "shift": err.mean().item() / std,
                "step": step / std,
                "peak": w.abs().max().item() / std,
            })
    return enc, stats


def evaluate(net, batches, fp16):
    """(accuracy %, images with non-finite logits)."""
    correct = bad = total = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        for x, y in batches:
            logits = net(x).float()
            bad += (~torch.isfinite(logits).all(1)).sum().item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return 100.0 * correct / total, bad


def trace_blocks(net, x, fp16):
    """Max |output| of the stem and of every residual block, in forward order."""
    import torchvision.models.resnet as R

    names, peaks, hooks = [], [], []
    for name, mod in net.named_modules():
        if name == "maxpool" or isinstance(mod, (R.BasicBlock, R.Bottleneck)):
            def hook(_m, _i, out, name=name):
                names.append(name)
                o = out.float()
                peaks.append(float("inf") if not torch.isfinite(o).all() else o.abs().max().item())
            hooks.append(mod.register_forward_hook(hook))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        net(x)
    for h in hooks:
        h.remove()
    return names, peaks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--arch", help="torchvision architecture, if the checkpoint has no metadata['arch']")
    ap.add_argument("--digits", type=int, default=10)
    ap.add_argument("--data", default="./data", help="CIFAR-10 root (downloaded if missing)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, help="evaluate only the first N test images")
    ap.add_argument("--top", type=int, default=10, help="layers to list by encoding error")
    ap.add_argument("--per-channel", action="store_true",
                    help="a scale and offset per output channel (the notebook's default from 0.2.3); "
                         "without it one per layer, as the README's encoding table was measured")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, masks, arch = load_checkpoint(args.checkpoint, args.arch)
    net = net.to(device, memory_format=torch.channels_last)
    enc, stats = encode_model(net, masks, FibonacciEncoder(args.digits), axis=0 if args.per_channel else None)
    batches = cifar_batches(args.data, args.batch, args.img_size, device, args.limit)
    report(arch, net, enc, stats, batches, device.type == "cuda", args.top)


def report(arch, net, enc, stats, batches, cuda, top):
    precisions = [("fp32", False)] + ([("fp16", True)] if cuda else [])
    print(f"{arch}: {len(stats)} encoded layers, {sum(len(y) for _, y in batches)} test images\n")
    print(f"{'model':<9}{'precision':<11}{'accuracy':>9}{'non-finite':>12}")
    acc = {}
    for label, model in (("pruned", net), ("encoded", enc)):
        for prec, fp16 in precisions:
            a, bad = evaluate(model, batches, fp16)
            acc[label, prec] = a
            print(f"{label:<9}{prec:<11}{a:>8.2f}%{bad:>12}")
    if not cuda:
        print("(no CUDA: fp16 autocast rows skipped; run on a GPU to compare)")

    x = batches[0][0]
    fp16 = cuda
    names, pruned_peaks = trace_blocks(net, x, fp16)
    _, enc_peaks = trace_blocks(enc, x, fp16)
    prec = "fp16" if fp16 else "fp32"
    print(f"\nmax |block output| on one batch ({prec}), every 4th block, last, and any past fp16's range:")
    print(f"{'block':<16}{'pruned':>12}{'encoded':>12}")
    for i, name in enumerate(names):
        over = max(pruned_peaks[i], enc_peaks[i]) > FP16_MAX
        if i % 4 == 0 or i == len(names) - 1 or over:
            print(f"{name:<16}{pruned_peaks[i]:>12.4g}{enc_peaks[i]:>12.4g}" + ("  <- over fp16" if over else ""))
    first = next((names[i] for i, p in enumerate(enc_peaks) if p > FP16_MAX), None)

    print(f"\nlayers with the largest encoding error (step and peak over the layer's weight std):")
    print(f"{'layer':<34}{'rel err':>9}{'shift':>9}{'step':>8}{'peak':>8}")
    for s in sorted(stats, key=lambda s: -s["rel_err"])[:top]:
        print(f"{s['layer']:<34}{s['rel_err']:>9.4f}{s['shift']:>+9.4f}{s['step']:>8.3f}{s['peak']:>8.1f}")

    print("\nverdict:")
    if first:
        print(f"  encoded activations first leave fp16's range at {first}")
    if cuda:
        drop32 = acc["pruned", "fp32"] - acc["encoded", "fp32"]
        drop16 = acc["pruned", "fp16"] - acc["encoded", "fp16"]
        if drop16 > drop32 + 5:
            print(f"  encoding costs {drop32:.2f} pts in fp32 but {drop16:.2f} under fp16 autocast: "
                  "the collapse is fp16 overflow, not the encoding alone")
        else:
            print(f"  encoding costs {drop32:.2f} pts in fp32 and {drop16:.2f} under fp16: "
                  "the encoding itself breaks the model; see the layers above")
    else:
        print(f"  encoding costs {acc['pruned', 'fp32'] - acc['encoded', 'fp32']:.2f} pts in fp32")


if __name__ == "__main__":
    main()
