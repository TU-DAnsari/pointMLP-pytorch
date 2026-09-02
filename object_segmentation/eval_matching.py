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
import open3d as o3d

import models as models
from util.mixed_dataset import MixedOccupancyDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Path to checkpoint dir")

    args = parser.parse_args()

    return args

def points_to_feature_vector(points):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    normals = np.asarray(pcd.normals)

    feature_vector = np.concatenate([points, normals], axis=1)
    return feature_vector

def through_backbone(backbone, points, n_feats):
    points = MixedOccupancyDataset.normlize_unit_sphere(points)
    feature_vector = points_to_feature_vector(points)
    points_tensor = torch.tensor(np.array([points])).float().permute(0, 2, 1)
    features_tensor = torch.tensor(np.array([feature_vector])).float().permute(0, 2, 1)
    
    with torch.no_grad():
        points_out = backbone.encoder(points_tensor, features_tensor)
        features = backbone.seg_head.decode(points_out).permute(0, 2, 1)

    n_feats = None if n_feats <= 0 else n_feats
    return points_tensor.permute(0, 2, 1), features_tensor.permute(0, 2, 1), features[:, :, :n_feats]

def through_occ(model, target_points, target_features, source_points, source_features):
    # target_points_tensor = torch.tensor(np.array([target_points])).float()
    # target_features_tensor = torch.tensor(np.array([target_features])).float()
    # source_points_tensor = torch.tensor(np.array([source_points])).float()
    # source_features_tensor = torch.tensor(np.array([source_features])).float()

    with torch.no_grad():
        # occ_pred = model(target_points_tensor, target_features_tensor, source_points_tensor, source_features_tensor)
        occ_pred = model(target_points, target_features, source_points, source_features)
        occ_pred = occ_pred.squeeze(-1)
        occ_prob = torch.sigmoid(occ_pred)

    return occ_prob


def main(model_dir):
    with open(model_dir / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        args = SimpleNamespace(**cfg)

    device = torch.device("cpu")

    backbone_eval = Path(args.backbone_eval).expanduser().resolve()
        
    with open(backbone_eval / "config.yaml", 'r') as f:
        config_backbone = yaml.safe_load(f)
        args_backbone = SimpleNamespace(**config_backbone)

    label_remap = args_backbone.label_remap
    labels_classes = args_backbone.labels_classes
    n_classes = len(set(label_remap.values())) if label_remap else 13

    assert len(labels_classes) == n_classes

    backbone = models.__dict__[args_backbone.model](n_classes, args_backbone.num_points, args_backbone.n_inputs)
    checkpoint = torch.load(backbone_eval / "best_insiou_model.pth", weights_only=False, map_location=device)
    state_dict = checkpoint["model"]
    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()


    model = models.__dict__[args.model]()
    checkpoint = torch.load(model_dir / "best_iou_model.pth", weights_only=False, map_location=device)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    data_path = Path(args.data_path)
    test_dataset = MixedOccupancyDataset(data_path, 
                                         split="test", 
                                         num_points=args.num_points)

    confusion = np.zeros((2, 2), dtype=np.int64)  # rows/cols: 0 = negative, 1 = positive
    model.eval()

    with torch.no_grad():
        for reference, _, _, _, combined, labels in tqdm(test_dataset):
            reference_xyz_tensor, _, reference_features = through_backbone(backbone, reference)
            combined_xyz_tensor, _, combined_features = through_backbone(backbone, combined)

            occ_prob = through_occ(model,
                                   reference_xyz_tensor,
                                   reference_features,
                                   combined_xyz_tensor,
                                   combined_features)

            pred_flat = (occ_prob.numpy().reshape(-1) > 0.5).astype(np.int64)
            gt_flat = labels.reshape(-1).astype(np.int64)

            confusion += np.bincount(gt_flat * 2 + pred_flat, minlength=4).reshape(2, 2)

    true_positive = confusion[1, 1]
    support_pos = confusion[1].sum()  # ground-truth positive points
    union_pos = confusion[:, 1].sum() + support_pos - true_positive

    positive_iou = float(true_positive / union_pos) if union_pos > 0 else float("nan")
    positive_acc = float(true_positive / support_pos) if support_pos > 0 else float("nan")

    overall_acc = float(np.diag(confusion).sum() / confusion.sum())
    results_dir = model_dir / "results" / backbone_eval.name
    Path.mkdir(results_dir, exist_ok=True)

    save_results(results_dir, positive_iou, positive_acc, overall_acc, confusion)


def save_results(results_dir, positive_iou, positive_acc, overall_acc, confusion):
    class_names = ["negative", "positive"]

    results = {
        "positive_iou": positive_iou if not np.isnan(positive_iou) else None,
        "positive_acc": positive_acc if not np.isnan(positive_acc) else None,
        "overall_acc": overall_acc,
    }
    with open(results_dir / "eval_results.yaml", "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    np.save(results_dir / "confusion_matrix.npy", confusion)

    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=np.float64), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
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

    data = np.array([[positive_iou, positive_acc]])

    fig, ax = plt.subplots(figsize=(4, 1.5))
    sns.heatmap(
        data, mask=np.isnan(data), annot=True, fmt=".4f", cmap="Blues",
        xticklabels=["IoU", "Accuracy"], yticklabels=["Positive"],
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
    )
    ax.xaxis.tick_top()
    ax.set_title(f"Positive-class metrics  (OA: {overall_acc:.4f})")

    plt.tight_layout()
    plt.savefig(results_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()

    main(model_dir)
