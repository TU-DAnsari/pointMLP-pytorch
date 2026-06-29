import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    balanced_accuracy_score,
)

def evaluate(labels, labels_pred, class_names=None):
    labels      = np.array(labels)
    labels_pred = np.array(labels_pred)

    # ── overall metrics ───────────────────────────────────────────────────────
    print("=" * 55)
    print("OVERALL METRICS")
    print("=" * 55)
    print(f"  Accuracy          : {accuracy_score(labels, labels_pred):.4f}")
    print(f"  Balanced accuracy : {balanced_accuracy_score(labels, labels_pred):.4f}")
    print(f"  Cohen's kappa     : {cohen_kappa_score(labels, labels_pred):.4f}")

    # ── per-class report ──────────────────────────────────────────────────────
    print()
    print("PER-CLASS REPORT")
    print("=" * 55)
    print(classification_report(labels, labels_pred, target_names=class_names))

    # ── confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels, labels_pred)
    classes = class_names or np.unique(np.concatenate([labels, labels_pred]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, ax=axes[0],
    )
    axes[0].set_title("Confusion matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # normalised (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        vmin=0, vmax=1, linewidths=0.5, ax=axes[1],
    )
    axes[1].set_title("Confusion matrix (normalised)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.show()

    # ── per-class F1 bar chart ────────────────────────────────────────────────
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(labels, labels_pred, zero_division=0)

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 4))
    ax.bar(x - width, p,  width, label="Precision", color="#378ADD")
    ax.bar(x,         r,  width, label="Recall",    color="#3B6D11")
    ax.bar(x + width, f,  width, label="F1",        color="#534AB7")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()