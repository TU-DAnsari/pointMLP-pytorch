import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed on a training server
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os

_CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#aaffc3",
]

def save_plots(history, checkpoint_dir, labels_classes):
    """
    Write three PNG files to checkpoint_dir:
      training_curves.png  – train/val loss + val accuracy on one figure
      iou_curves.png       – val instance IoU + val mean-class IoU
      per_class_iou.png    – one line per semantic class (val)

    Parameters
    ----------
    history : dict
        Keys populated by train():
          "train_loss", "train_acc", "train_ins_iou"   – one float per epoch
          "val_loss",   "val_acc",   "val_ins_iou"      – one float per epoch
          "val_cls_iou"                                 – one float per epoch (mean across classes)
          "val_per_class_iou"  – list of arrays, shape (n_classes,), one per epoch
    checkpoint_dir : str
    labels_classes : list[str]
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.style.use("seaborn-v0_8-darkgrid")
    _TITLE_SIZE  = 11
    _LABEL_SIZE  = 9
    _LEGEND_SIZE = 8

    # ── 1. Loss + Accuracy ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Training curves", fontsize=_TITLE_SIZE, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", color="#2563eb", linewidth=1.8)
    ax.plot(epochs, history["val_loss"],   label="Val loss",   color="#dc2626", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=_LABEL_SIZE)
    ax.set_ylabel("NLL loss", fontsize=_LABEL_SIZE)
    ax.set_title("Loss", fontsize=_LABEL_SIZE)
    ax.legend(fontsize=_LEGEND_SIZE)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train acc", color="#2563eb", linewidth=1.8)
    ax.plot(epochs, history["val_acc"],   label="Val acc",   color="#dc2626", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Accuracy", fontsize=_LABEL_SIZE)
    ax.set_title("Point accuracy", fontsize=_LABEL_SIZE)
    ax.legend(fontsize=_LEGEND_SIZE)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    fig.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. IoU summary ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Validation IoU", fontsize=_TITLE_SIZE, fontweight="bold")

    ax.plot(epochs, history["val_ins_iou"],  label="Instance mIoU", color="#7c3aed", linewidth=2)
    ax.plot(epochs, history["val_cls_iou"],  label="Class mIoU",    color="#059669", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=_LABEL_SIZE)
    ax.set_ylabel("IoU", fontsize=_LABEL_SIZE)
    ax.legend(fontsize=_LEGEND_SIZE)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    fig.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, "iou_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Per-class IoU ─────────────────────────────────────────────────
    if history["val_per_class_iou"]:
        per_class = np.stack(history["val_per_class_iou"], axis=0)  # (epochs, n_classes)
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("Per-class validation IoU", fontsize=_TITLE_SIZE, fontweight="bold")

        for cls_idx, cls_name in enumerate(labels_classes):
            ax.plot(epochs, per_class[:, cls_idx],
                    label=cls_name,
                    color=_CLASS_COLORS[cls_idx % len(_CLASS_COLORS)],
                    linewidth=1.5)

        ax.set_xlabel("Epoch", fontsize=_LABEL_SIZE)
        ax.set_ylabel("IoU", fontsize=_LABEL_SIZE)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=_LEGEND_SIZE, ncol=3, loc="lower right")

        fig.tight_layout()
        fig.savefig(os.path.join(checkpoint_dir, "per_class_iou.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)