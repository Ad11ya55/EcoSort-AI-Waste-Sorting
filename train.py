"""Train EcoSort's MobileNetV2 image classifier on a folder-per-class dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# ============================================================================
# 1. MAKE EXPERIMENTS REPEATABLE
# Fix random-number generators so a repeated split is easier to compare.
# ============================================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 2. PREPARE TRAINING AND VALIDATION IMAGE TRANSFORMS
# Augmentation applies only to training; validation uses a consistent resize.
# ============================================================================
def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# ============================================================================
# 3. RUN ONE TRAINING OR VALIDATION EPOCH
# Gradients and optimizer updates are enabled only during training.
# ============================================================================
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    loss_sum = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


# ============================================================================
# 4. READ COMMAND-LINE SETTINGS AND PREPARE THE DATASET
# ImageFolder uses each class-named directory as a label.
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder with one subfolder per class")
    parser.add_argument("--output-dir", default=".", help="Where to write model.pt, classes.json and training_history.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers (0 is Windows-friendly)")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    seed_everything(args.seed)
    data_dir, out_dir = Path(args.data_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(out_dir / ".torch_cache"))
    train_tf, eval_tf = make_transforms()
    base = datasets.ImageFolder(data_dir)
    if len(base) < 2 or len(base.classes) < 2:
        raise ValueError("Dataset must contain at least two classes and two images.")
    classes = base.classes
    generator = torch.Generator().manual_seed(args.seed)
    train_size = int(0.8 * len(base))
    train_size = min(max(train_size, 1), len(base) - 1)
    train_indices, val_indices = torch.utils.data.random_split(range(len(base)), [train_size, len(base)-train_size], generator=generator)
    train_ds = datasets.ImageFolder(data_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir, transform=eval_tf)
    train_ds = Subset(train_ds, train_indices.indices)
    val_ds = Subset(val_ds, val_indices.indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # ========================================================================
    # 5. CREATE THE TRANSFER-LEARNING MODEL
    # Train the new output layer first, then fine-tune the backbone more gently.
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; images: {len(base)}; classes: {classes}; train: {len(train_ds)}; validation: {len(val_ds)}")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    head_epochs = max(args.epochs - min(3, args.epochs), 0)
    history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}
    best_accuracy = -1.0
    best_path = out_dir / "model.pt"
    start = time.time()

    # ========================================================================
    # 6. TRAIN, VALIDATE, AND KEEP THE BEST CHECKPOINT
    # By default, the final three epochs unfreeze the feature extractor at one-tenth LR.
    # ========================================================================
    for epoch in range(args.epochs):
        fine_tune = epoch >= head_epochs
        if fine_tune and epoch == head_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
        lr = args.learning_rate / 10 if fine_tune else args.learning_rate
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, False)
        for key, value in [("train_loss", train_loss), ("train_accuracy", train_acc), ("validation_loss", val_loss), ("validation_accuracy", val_acc)]:
            history[key].append(value)
        print(f"Epoch {epoch+1}/{args.epochs} ({'fine-tune' if fine_tune else 'head'}): train loss {train_loss:.4f}, train acc {train_acc:.1%}, val loss {val_loss:.4f}, val acc {val_acc:.1%}")
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), best_path)

    # ========================================================================
    # 7. MEASURE PER-CLASS PERFORMANCE AND SAVE PROJECT ARTIFACTS
    # The validation score is optimistic because it also selected the checkpoint.
    # ========================================================================
    # Report per-class validation metrics for the checkpoint selected above.
    # This is a validation score, not an unbiased test score, because the same
    # split chose the checkpoint; evaluate.py accepts a separate held-out folder.
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in val_loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            for actual, predicted in zip(labels, predictions):
                confusion[actual, predicted] += 1
    per_class = {}
    for index, name in enumerate(classes):
        true_positive = int(confusion[index, index])
        precision = true_positive / max(int(confusion[:, index].sum()), 1)
        recall = true_positive / max(int(confusion[index, :].sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion[index, :].sum())}
    macro = {metric: sum(row[metric] for row in per_class.values()) / len(per_class) for metric in ("precision", "recall", "f1")}
    evaluation = {
        "accuracy": int(confusion.diag().sum()) / max(len(val_ds), 1),
        "macro": macro,
        "per_class": per_class,
        "class_order": classes,
        "confusion_matrix": confusion.tolist(),
        "image_count": len(val_ds),
        "evaluation_set": "validation split used for checkpoint selection; score may be optimistic",
    }
    (out_dir / "evaluation_results.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    elapsed = time.time() - start
    # Store a portable dataset reference rather than a machine-specific absolute path.
    try:
        dataset_reference = data_dir.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        dataset_reference = data_dir.name
    metadata = {
        "dataset": dataset_reference, "image_count": len(base), "class_counts": {name: sum(1 for _, label in base.samples if label == i) for i, name in enumerate(classes)},
        "class_names": classes, "train_images": len(train_ds), "validation_images": len(val_ds),
        "split": "80/20 random split", "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "fine_tune_learning_rate": args.learning_rate / 10, "best_validation_accuracy": best_accuracy,
        "device": str(device), "elapsed_seconds": elapsed,
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best_accuracy:.2%}; model saved to {best_path}; elapsed: {elapsed/60:.1f} min")
    print("Note: validation accuracy is used for checkpoint selection; evaluate on a separate test set for an unbiased estimate.")


if __name__ == "__main__":
    main()

