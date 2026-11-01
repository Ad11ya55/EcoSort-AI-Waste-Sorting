"""Evaluate a saved EcoSort checkpoint on a labelled folder-per-class dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================================
# 1. LOAD A LABELLED EVALUATION SET AND MATCH ITS CLASS ORDER
# Use a separate held-out folder-per-class dataset for an unbiased test score.
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder-per-class evaluation dataset")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--classes", default="classes.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()
    class_names = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    dataset = datasets.ImageFolder(args.data_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]))
    if dataset.classes != class_names:
        raise ValueError(f"Dataset classes {dataset.classes} do not match model classes {class_names}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device).eval()

    # ========================================================================
    # 2. RUN INFERENCE AND BUILD THE CONFUSION MATRIX
    # Rows are actual classes; columns are the model's predicted classes.
    # ========================================================================
    matrix = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            for actual, predicted in zip(labels, predictions):
                matrix[actual, predicted] += 1

    # ========================================================================
    # 3. CALCULATE PER-CLASS AND MACRO-AVERAGED METRICS
    # Macro averages give each category equal weight, including the small class.
    # ========================================================================
    metrics = {}
    for i, name in enumerate(class_names):
        tp = int(matrix[i, i])
        precision = tp / max(int(matrix[:, i].sum()), 1)
        recall = tp / max(int(matrix[i, :].sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[i, :].sum())}
    macro = {key: sum(row[key] for row in metrics.values()) / len(metrics) for key in ("precision", "recall", "f1")}
    result = {"accuracy": int(matrix.diag().sum()) / max(len(dataset), 1), "macro": macro, "per_class": metrics, "class_order": class_names, "confusion_matrix": matrix.tolist(), "image_count": len(dataset)}

    # ========================================================================
    # 4. SAVE THE EVALUATION REPORT
    # The JSON output can be loaded by the Streamlit Evaluation page.
    # ========================================================================
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

