"""
Inference with a Zeckendorf-pruned ResNet-18.
Classifies images into CIFAR-10 categories.

Usage:
    python classify.py photo.jpg
    python classify.py photo1.jpg photo2.png photo3.jpg
    python classify.py --webcam          (requires opencv)
    python classify.py --demo            (downloads a sample image)
"""

import sys
import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CHECKPOINT = "model_pruned.pt"

# Preprocessing: match CIFAR-10 via ImageNet normalization at 224×224
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_model():
    """Load pruned ResNet-18 adapted for CIFAR-10."""
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Run quickstart.py first to produce the pruned model.")
        sys.exit(1)

    data = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    # Rebuild architecture: ResNet-18 with 10-class head
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(data["state_dict"])
    model.eval()

    masks = data.get("masks", {})
    stats = data.get("prune_stats", {})

    print(f"Loaded pruned model from {CHECKPOINT}")
    if stats:
        densities = [s["density"] for s in stats.values() if isinstance(s, dict) and "density" in s]
        if densities:
            print(f"  Density: {sum(densities)/len(densities):.1%} across {len(densities)} layers")

    return model, masks


def classify(model, image_path):
    """Classify a single image."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    top3 = probs.topk(3)
    print(f"\n  {image_path}:")
    for prob, idx in zip(top3.values, top3.indices):
        print(f"    {CLASSES[idx]:12s}  {prob.item():.1%}")


def demo_mode(model):
    """Download a sample image and classify it."""
    import urllib.request

    samples = {
        "cat.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    }

    for name, url in samples.items():
        if not os.path.exists(name):
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, name)
        classify(model, name)


def webcam_mode(model):
    """Live classification from webcam."""
    try:
        import cv2
    except ImportError:
        print("Webcam mode requires opencv: pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        sys.exit(1)

    print("Press 'q' to quit, 'c' to classify current frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show live feed
        cv2.imshow("Zeckendorf Classifier - press 'c' to classify, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            # Convert OpenCV BGR → PIL RGB
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = transform(img).unsqueeze(0)

            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)[0]

            top3 = probs.topk(3)
            print("\n  Live capture:")
            for prob, idx in zip(top3.values, top3.indices):
                label = CLASSES[idx]
                pct = prob.item()
                print(f"    {label:12s}  {pct:.1%}")

            # Overlay on frame
            for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
                text = f"{CLASSES[idx]}: {prob.item():.0%}"
                cv2.putText(frame, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Classification", frame)
            cv2.waitKey(2000)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model, masks = load_model()

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python classify.py photo.jpg       # classify an image")
        print("  python classify.py --demo           # download & classify samples")
        print("  python classify.py --webcam          # live webcam classification")
        sys.exit(0)

    if sys.argv[1] == "--demo":
        demo_mode(model)
    elif sys.argv[1] == "--webcam":
        webcam_mode(model)
    else:
        for path in sys.argv[1:]:
            if os.path.exists(path):
                classify(model, path)
            else:
                print(f"  File not found: {path}")