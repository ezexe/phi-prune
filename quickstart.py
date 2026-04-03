"""
Quick start: Prune a pretrained ResNet-18 with Zeckendorf constraint.
Run from inside the zeckendorf-prune directory.

    pip install -e .
    python quickstart.py
"""

import torch
import torchvision
import torchvision.transforms as transforms
from zeckendorf_prune import prune, finetune, check
from zeckendorf_prune.encoding import FibonacciEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Load pretrained ResNet-18 ──
print("\n1. Loading pretrained ResNet-18...")
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.to(device)
model.fc = torch.nn.Linear(512, 10)
model.eval()
print(f"   {sum(p.numel() for p in model.parameters()):,} parameters")

# ── 2. Load CIFAR-10 (resized to 224×224 for ResNet-18) ──
print("\n2. Loading CIFAR-10...")
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=0)

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=0
)

# ── 3. Evaluate dense baseline ──
print("\n3. Dense baseline accuracy...")
model.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
dense_acc = 100 * correct / total
print(f"   Dense: {dense_acc:.2f}%")

# ── 4. Prune ──
print("\n4. Applying Zeckendorf pruning...")
pruned_model, masks = prune(model, inplace=False)
stats = pruned_model._zeck_prune_stats
print(f"   Density: {stats['density']:.1%}")
print(f"   Pruned layers: {stats['pruned_layers']}")

# ── 5. Verify masks ──
print("\n5. Verifying masks...")
report = check(pruned_model, masks)
print(f"   All masks valid: {report['_summary']['all_masks_valid']}")
print(f"   All zeros enforced: {report['_summary']['all_zeros_enforced']}")

# ── 6. Evaluate pruned (before fine-tune) ──
print("\n6. Accuracy after pruning (before fine-tune)...")
pruned_model.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = pruned_model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
pruned_acc = 100 * correct / total
print(f"   Pruned (no fine-tune): {pruned_acc:.2f}%")

# ── 7. Fine-tune (short — 5 epochs for demo) ──
print("\n7. Fine-tuning (5 epochs)...")
ft_results = finetune(
    pruned_model, train_loader, epochs=5, masks=masks,
    lr=0.001, device=device, val_loader=test_loader, verbose=True
)
print(f"   Best val accuracy: {ft_results['best_val_acc']:.2f}%")

# ── 8. Fibonacci encoding ──
print("\n8. Fibonacci encoding...")
encoder = FibonacciEncoder(n_digits=10)  # 144 levels
print(f"   Grid: {encoder.n_levels} levels, max value: {encoder.max_value}")

sample_param = next(n for n, p in pruned_model.named_parameters() if n in masks)
param = dict(pruned_model.named_parameters())[sample_param]
encoded, scale, rmse = encoder.encode_tensor(param.data, mask=masks[sample_param])
print(f"   Sample layer ({sample_param}): RMSE = {rmse:.6f}")

# ── 9. Summary ──
print(f"\n{'='*50}")
print(f"  Dense:              {dense_acc:.2f}%")
print(f"  Pruned (no ft):     {pruned_acc:.2f}%")
print(f"  Pruned (5 ep ft):   {ft_results['best_val_acc']:.2f}%")
print(f"  Density:            {stats['density']:.1%}")
print(f"  Masks valid:        {report['_summary']['all_masks_valid']}")
print(f"  Encoding levels:    {encoder.n_levels}")
print(f"{'='*50}")

from zeckendorf_prune.export import save_checkpoint
save_checkpoint(pruned_model, masks, "model_pruned.pt")
print("Saved model_pruned.pt")