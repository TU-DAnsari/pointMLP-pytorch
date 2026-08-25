import yaml
import argparse 
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import models as models
from util.s3dis_dataset import S3DISDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Path to checkpoint dir")

    args = parser.parse_args()

    return args

def main(model_dir):
    with open(model_dir / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        args = SimpleNamespace(**cfg)

    label_remap = args.label_remap
    labels_classes = args.labels_classes

    test_paths = [Path(path).expanduser().resolve() for path in args.test_paths]
    n_classes = len(set(label_remap.values())) if label_remap else 13

    device = torch.device("cpu")
    model = models.__dict__[args.model](n_classes, args.num_points, args.n_inputs)
    checkpoint = torch.load(model_dir / "best_insiou_model.pth", weights_only=False, map_location=device)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()

    test_dataset = S3DISDataset(test_paths,
                            num_points=args.num_points,
                            min_points=args.min_points,
                            noise_std=args.noise_std,
                            block_size=args.block_size,
                            stride=args.stride,
                            normalize=args.normalize,
                            label_remap=label_remap,
                            )

    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size, 
                            shuffle=False,
                            num_workers=args.workers, 
                            drop_last=False, 
                            persistent_workers=True)

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for points_batch, features_batch, labels_batch in tqdm(test_loader, total=len(test_loader), smoothing=0.9):
            points_batch = points_batch.float().permute(0, 2, 1)
            features_batch = features_batch.float().permute(0, 2, 1)
            labels_batch = labels_batch.long()

            seg_pred = model(points_batch, features_batch)
            pred_choice = seg_pred.data.max(2)[1]

            gt_flat   = labels_batch.view(-1).cpu().numpy()
            pred_flat = pred_choice.view(-1).cpu().numpy()
            confusion += np.bincount(gt_flat * n_classes + pred_flat,
                                      minlength=n_classes * n_classes).reshape(n_classes, n_classes)

    intersection = np.diag(confusion)
    support = confusion.sum(axis=1)  # ground-truth points per class
    union = confusion.sum(axis=0) + support - intersection
    present = union > 0

    per_class_iou = np.full(n_classes, np.nan, dtype=np.float64)
    per_class_iou[present] = intersection[present] / union[present]
    mean_iou = np.nanmean(per_class_iou)

    per_class_acc = np.full(n_classes, np.nan, dtype=np.float64)
    per_class_acc[support > 0] = intersection[support > 0] / support[support > 0]
    mean_acc = np.nanmean(per_class_acc)

    overall_acc = intersection.sum() / confusion.sum()
    results_dir = model_dir / "results"
    Path.mkdir(results_dir, exist_ok=True)

    save_results(results_dir, labels_classes, per_class_iou, per_class_acc, mean_iou, mean_acc, overall_acc, confusion)


def save_results(results_dir, labels_classes, per_class_iou, per_class_acc, mean_iou, mean_acc, overall_acc, confusion):
    results = {
        "mean_iou": float(mean_iou),
        "mean_acc": float(mean_acc),
        "overall_acc": float(overall_acc),
        "per_class_iou": {
            name: (float(v) if not np.isnan(v) else None)
            for name, v in zip(labels_classes, per_class_iou)
        },
        "per_class_acc": {
            name: (float(v) if not np.isnan(v) else None)
            for name, v in zip(labels_classes, per_class_acc)
        },
    }
    with open(results_dir / "eval_results.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    np.save(results_dir / "confusion_matrix.npy", confusion)

    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=np.float64), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        confusion_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels_classes, yticklabels=labels_classes,
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix (row-normalized)")
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    n_classes = len(labels_classes)
    row_labels = list(labels_classes) + ["Mean"]
    data = np.array(
        [[per_class_iou[cls], per_class_acc[cls]] for cls in range(n_classes)] + [[mean_iou, mean_acc]]
    )

    fig, ax = plt.subplots(figsize=(4, 0.5 * len(row_labels) + 1))
    sns.heatmap(
        data, mask=np.isnan(data), annot=True, fmt=".4f", cmap="Blues",
        xticklabels=["IoU", "Accuracy"], yticklabels=row_labels,
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.xaxis.tick_top()
    ax.set_title(f"Per-class metrics  (OA: {overall_acc:.4f})")

    plt.tight_layout()
    plt.savefig(results_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()

    main(model_dir)
