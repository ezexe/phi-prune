"""
Probe Fibonacci encoding on ImageNet-pretrained ResNets, fp32 vs fp16.

Encoded ResNet-152 V2 scored exactly 10.00% on CIFAR-10 in the notebook,
which is what non-finite logits give. This script needs no fine-tuned
checkpoint or GPU: it loads torchvision's ImageNet weights, and for each
model compares four variants on CIFAR-10 test images upscaled as in the
notebook:

  dense            the pretrained weights
  dense+enc        every conv layer encoded with an all-ones mask
                   (isolates encoding from pruning)
  pruned           Zeckendorf-pruned convs, as the notebook prunes them
  pruned+enc       the pruned convs encoded, as the notebook's encoding cell does

For each variant it prints the share of images whose logits are not finite
in fp32 and under fp16 autocast (CPU autocast works; it is slower than
fp32), how often the top-1 ImageNet class agrees with the dense fp32 model,
and the largest block output in fp32, against fp16's largest finite value
65504. Pruned variants are not fine-tuned here, so their agreement is low
by construction; the fp16 columns are the point.

Usage:
    python scripts/probe_pretrained_encoding.py \
        --models resnet152:IMAGENET1K_V2,resnet152:IMAGENET1K_V1 --images 64
"""

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from zeckendorf_prune import prune
from zeckendorf_prune.encoding import FibonacciEncoder

FP16_MAX = 65504.0
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def cifar_images(root, n, size, fake=False):
    if fake:  # offline smoke test
        x = torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8)
    else:
        import torchvision

        ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
        x = torch.from_numpy(ds.data[:n]).permute(0, 3, 1, 2)
    x = x.float().div(255)
    x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def encode_convs(model, masks, encoder):
    """Encode each masked parameter in place, as the notebook's encoding cell does."""
    params = dict(model.named_parameters())
    worst = (0.0, "")
    with torch.no_grad():
        for name, mask in masks.items():
            p = params[name]
            keep = mask.bool()
            w = p[keep].clone()
            encoded, _, _ = encoder.encode_tensor(p.data, mask=mask)
            p.copy_(encoded)
            rel = ((p[keep] - w).norm() / w.norm().clamp_min(1e-30)).item()
            worst = max(worst, (rel, name))
    return worst


def full_masks(model):
    return {
        f"{n}.weight": torch.ones_like(m.weight)
        for n, m in model.named_modules() if isinstance(m, nn.Conv2d)
    }


def run(model, x, batch, fp16):
    """Logits for x, in fp32 or under CPU fp16 autocast."""
    outs = []
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.float16, enabled=fp16):
        for i in range(0, len(x), batch):
            outs.append(model(x[i:i + batch]).float())
    return torch.cat(outs)


def block_peak(model, x):
    """Largest |output| of any residual block, and the first block past fp16's range (fp32)."""
    import torchvision.models.resnet as R

    peaks = []
    hooks = [
        m.register_forward_hook(lambda _m, _i, o, n=n: peaks.append((o.abs().max().item(), n)))
        for n, m in model.named_modules() if isinstance(m, (R.BasicBlock, R.Bottleneck))
    ]
    with torch.inference_mode():
        model(x)
    for h in hooks:
        h.remove()
    first_over = next((n for p, n in peaks if p > FP16_MAX), "-")
    return max(peaks)[0], first_over


def probe(arch, weights, x, x16, batch, digits):
    import torchvision

    dense = torchvision.models.get_model(arch, weights=weights).eval()
    enc = FibonacciEncoder(digits)

    dense_enc = copy.deepcopy(dense)
    worst_dense = encode_convs(dense_enc, full_masks(dense_enc), enc)
    pruned, masks = prune(dense, layer_types=(nn.Conv2d,))
    pruned_enc = copy.deepcopy(pruned)
    worst_pruned = encode_convs(pruned_enc, masks, enc)

    ref = run(dense, x, batch, False).argmax(1)
    rows = []
    for label, m in (("dense", dense), ("dense+enc", dense_enc),
                     ("pruned", pruned), ("pruned+enc", pruned_enc)):
        l32 = run(m, x, batch, False)
        l16 = run(m, x16, batch, True)
        peak, first = block_peak(m, x[:batch])
        rows.append((label,
                     (~torch.isfinite(l32).all(1)).float().mean().item(),
                     (~torch.isfinite(l16).all(1)).float().mean().item(),
                     (l32.argmax(1) == ref).float().mean().item(),
                     (l16.argmax(1) == ref[:len(l16)]).float().mean().item(),
                     peak, first))
    return rows, worst_dense, worst_pruned


def encode_strategy(model, encoder, strategy):
    """
    Encode every conv weight of model in place with one quantization strategy.

    tensor      one (scale, offset) per layer, as encode_tensor does today
    channel     one (scale, offset) per output channel
    clipP       per layer, after clamping weights to their [100-P, P] percentiles
    Returns the per-layer relative errors.
    """
    errs = []
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, nn.Conv2d):
                continue
            w = m.weight.data
            orig = w.clone()
            if strategy == "tensor":
                w.copy_(encoder.encode_tensor(w)[0])
            elif strategy == "channel":
                for c in range(w.shape[0]):
                    w[c].copy_(encoder.encode_tensor(w[c])[0])
            elif strategy.startswith("clip"):
                p = float(strategy[4:]) / 100
                flat = w.flatten().double()
                lo, hi = torch.quantile(flat, 1 - p).item(), torch.quantile(flat, p).item()
                w.copy_(encoder.encode_tensor(w.clamp(lo, hi))[0])
            else:
                raise ValueError(strategy)
            errs.append(((w - orig).norm() / orig.norm().clamp_min(1e-30)).item())
    return errs


def compare_strategies(specs, strategies, x, batch, digits):
    import torchvision

    enc = FibonacciEncoder(digits)
    for spec in specs:
        arch, weights = spec.split(":")
        dense = torchvision.models.get_model(arch, weights=weights).eval()
        ref = run(dense, x, batch, False)
        print(f"{arch} {weights}: dense+encoded vs dense, fp32, {len(x)} images")
        print(f"  {'strategy':<11}{'agree':>8}{'logit rel err':>15}{'mean layer err':>16}{'worst':>8}")
        for s in strategies:
            m = copy.deepcopy(dense)
            errs = encode_strategy(m, enc, s)
            out = run(m, x, batch, False)
            agree = (out.argmax(1) == ref.argmax(1)).float().mean().item()
            lerr = ((out - ref).norm() / ref.norm()).item()
            print(f"  {s:<11}{agree:>8.1%}{lerr:>15.3f}{sum(errs) / len(errs):>16.4f}{max(errs):>8.4f}",
                  flush=True)
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="resnet152:IMAGENET1K_V2,resnet152:IMAGENET1K_V1",
                    help="comma-separated arch:weights pairs")
    ap.add_argument("--images", type=int, default=64, help="CIFAR-10 test images for fp32")
    ap.add_argument("--fp16-images", type=int, default=32, help="first N of them also run in fp16")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--digits", type=int, default=10)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--fake-data", action="store_true", help="random images (offline smoke test)")
    ap.add_argument("--strategies", default="",
                    help="instead of the fp16 probe, compare quantization strategies on the dense "
                         "models, e.g. tensor,channel,clip99.9")
    args = ap.parse_args()

    torch.manual_seed(0)
    x = cifar_images(args.data, args.images, args.img_size, args.fake_data)
    if args.strategies:
        compare_strategies(args.models.split(","), args.strategies.split(","), x, args.batch, args.digits)
        return
    x16 = x[:args.fp16_images]
    print(f"{len(x)} images at {args.img_size} px ({len(x16)} in fp16), "
          f"{args.digits}-digit codewords, {torch.get_num_threads()} threads\n")
    for spec in args.models.split(","):
        arch, weights = spec.split(":")
        start = time.perf_counter()
        rows, wd, wp = probe(arch, weights, x, x16, args.batch, args.digits)
        print(f"{arch} {weights}  ({(time.perf_counter() - start) / 60:.1f} min)")
        print(f"  {'variant':<12}{'nonfinite32':>12}{'nonfinite16':>12}{'agree32':>9}{'agree16':>9}"
              f"{'max|block|':>12}  first block over 65504")
        for label, n32, n16, a32, a16, peak, first in rows:
            print(f"  {label:<12}{n32:>12.1%}{n16:>12.1%}{a32:>9.1%}{a16:>9.1%}{peak:>12.4g}  {first}")
        print(f"  worst layer rel. error: dense+enc {wd[0]:.4f} ({wd[1]}), "
              f"pruned+enc {wp[0]:.4f} ({wp[1]})\n", flush=True)


if __name__ == "__main__":
    main()
