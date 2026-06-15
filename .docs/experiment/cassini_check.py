"""Just the Cassini check — run after stacked.py has already trained."""
import torch, random, copy, sys
sys.path.insert(0, '.')
from Zeckendorf import ResNet20, get_cifar10_loaders, apply_pruning, evaluate
from phase1_stacked import FibonacciEncoder

device = torch.device('cpu')
_, test_loader = get_cifar10_loaders(data_dir='./quick')

# Rebuild the stacked model (fast — no training, just load + prune + encode)
model = ResNet20().to(device)
model.load_state_dict(torch.load('./quick/resnet20_dense.pt', map_location=device, weights_only=True))
masks, stats = apply_pruning(model, mask_type='zeckendorf')

encoder = FibonacciEncoder(n_digits=8)
scales = {}
with torch.no_grad():
    for name, param in model.named_parameters():
        if name in masks:
            encoded, scale, _ = encoder.encode_tensor(param.data)
            param.data.copy_(encoded * masks[name])
            scales[name] = scale

# ── The actual Cassini test (bit-level corruption) ──
random.seed(42)
n_corrupt, n_adj, n_cassini = 0, 0, 0

for name, param in model.named_parameters():
    if name not in scales: continue
    flat = param.data.flatten().cpu()
    mask_flat = masks[name].flatten().cpu()
    scale = scales[name]
    active = [i for i in range(len(flat)) if mask_flat[i] > 0 and abs(flat[i]) > 1e-8]
    if not active: continue

    for i in random.sample(active, min(500, len(active))):
        int_val = abs(round(flat[i].item() / scale)) if scale > 0 else 0
        int_val = min(int_val, encoder.max_value)
        bits = encoder.to_zeckendorf(int_val)

        # Flip one random Fibonacci digit
        flip_pos = random.randint(0, encoder.n_digits - 1)
        corrupted = bits.copy()
        corrupted[flip_pos] = 1 - corrupted[flip_pos]

        # FREE check: adjacent 1s?
        has_adj = any(corrupted[j] == 1 and corrupted[j+1] == 1
                      for j in range(encoder.n_digits - 1))
        if has_adj: n_adj += 1

        # DEEPER check: re-decompose and compare
        corrupted_val = sum(b * w for b, w in zip(corrupted, encoder.weights))
        canonical = encoder.to_zeckendorf(corrupted_val)
        if has_adj or canonical != corrupted: n_cassini += 1

        n_corrupt += 1
    if n_corrupt >= 2000: break

print(f"Tested {n_corrupt} single-bit flips\n")
print(f"FREE (adjacency check):     {n_adj}/{n_corrupt} ({n_adj/n_corrupt:.1%})")
print(f"FULL (re-decomposition):    {n_cassini}/{n_corrupt} ({n_cassini/n_corrupt:.1%})")