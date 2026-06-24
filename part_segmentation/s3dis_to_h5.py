"""
S3DIS → per-area H5 preprocessor
----------------------------------
Reads the original annotation .txt files and writes one H5 file per area.

Output layout (one file per area, e.g. Area_1.h5):
    /points          float32  (N, 6)   – x y z r g b   (rgb kept as 0-255 uint8 range)
    /semantic_labels int32    (N,)     – 0-12 class index, -1 = unknown/clutter
    /instance_labels int32    (N,)     – unique instance id per object, 0-based per area

Semantic class map (13 classes):
    0  ceiling      1  floor       2  wall        3  beam
    4  column       5  window      6  door        7  table
    8  chair        9  sofa       10  bookcase    11  board
    12 clutter

Usage:
    python s3dis_to_h5.py --data_root /path/to/Stanford3dDataset_v1.2_Aligned_Version
                          --output_dir /path/to/output
                          [--areas 1 2 3 4 5 6]
                          [--verbose]
"""

import os
import re
import argparse
import numpy as np
import h5py

# ── Semantic class map ────────────────────────────────────────────────────────

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
UNKNOWN_LABEL = CLASS_MAP["clutter"]   # fallback for unrecognised names


def semantic_label(filename: str) -> int:
    """Infer semantic class index from an annotation filename like 'chair_3.txt'."""
    base = os.path.splitext(os.path.basename(filename))[0]   # e.g. 'chair_3'
    # strip trailing _<digits>
    class_name = re.sub(r"_\d+$", "", base).lower()
    return CLASS_MAP.get(class_name, UNKNOWN_LABEL)


# ── Per-room loader ───────────────────────────────────────────────────────────

def load_room(room_path: str, instance_offset: int):
    """
    Read all annotation files in <room_path>/Annotations/.

    Returns:
        points          float32 (M, 6)
        sem_labels      int32   (M,)
        inst_labels     int32   (M,)   – globally unique within an area
        next_offset     int             – instance_offset + number of instances in room
    """
    ann_dir = os.path.join(room_path, "Annotations")
    if not os.path.isdir(ann_dir):
        print(f"  [WARN] No Annotations dir in {room_path}, skipping.")
        return None, None, None, instance_offset

    ann_files = sorted(
        f for f in os.listdir(ann_dir) if f.endswith(".txt")
    )
    if not ann_files:
        print(f"  [WARN] Empty Annotations dir in {room_path}, skipping.")
        return None, None, None, instance_offset

    room_points, room_sem, room_inst = [], [], []

    for inst_local_idx, fname in enumerate(ann_files):
        fpath = os.path.join(ann_dir, fname)
        try:
            pts = np.loadtxt(fpath, dtype=np.float32)   # (K, 6)
        except Exception as e:
            print(f"  [WARN] Could not read {fpath}: {e}")
            continue

        if pts.ndim == 1:          # single-point file edge-case
            pts = pts[None, :]
        if pts.shape[1] < 6:
            print(f"  [WARN] Unexpected columns in {fpath}: {pts.shape}, skipping.")
            continue

        n = len(pts)
        sem_idx = semantic_label(fname)
        inst_idx = instance_offset + inst_local_idx

        room_points.append(pts[:, :6])
        room_sem.append(np.full(n, sem_idx, dtype=np.int32))
        room_inst.append(np.full(n, inst_idx, dtype=np.int32))

    if not room_points:
        return None, None, None, instance_offset

    next_offset = instance_offset + len(ann_files)
    return (
        np.concatenate(room_points, axis=0),
        np.concatenate(room_sem,    axis=0),
        np.concatenate(room_inst,   axis=0),
        next_offset,
    )


# ── Per-area processor ────────────────────────────────────────────────────────

def process_area(area_path: str, output_path: str, verbose: bool = False):
    """Merge all rooms in an area and write a single H5 file."""
    room_dirs = sorted(
        d for d in os.listdir(area_path)
        if os.path.isdir(os.path.join(area_path, d))
    )

    all_points, all_sem, all_inst = [], [], []
    instance_offset = 0

    for room_name in room_dirs:
        room_path = os.path.join(area_path, room_name)
        if verbose:
            print(f"    Loading {room_name} …")

        pts, sem, inst, instance_offset = load_room(room_path, instance_offset)
        if pts is None:
            continue

        all_points.append(pts)
        all_sem.append(sem)
        all_inst.append(inst)

    if not all_points:
        print(f"  [WARN] No valid rooms found in {area_path}, skipping H5 output.")
        return

    points = np.concatenate(all_points, axis=0)   # (N, 6)
    sem    = np.concatenate(all_sem,    axis=0)   # (N,)
    inst   = np.concatenate(all_inst,   axis=0)   # (N,)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Store datasets with lossless compression
        f.create_dataset("points",
                         data=points,
                         dtype="float32",
                         compression="gzip",
                         compression_opts=4)
        f.create_dataset("semantic_labels",
                         data=sem,
                         dtype="int32",
                         compression="gzip",
                         compression_opts=4)
        f.create_dataset("instance_labels",
                         data=inst,
                         dtype="int32",
                         compression="gzip",
                         compression_opts=4)

        # Metadata
        f.attrs["num_points"]    = len(points)
        f.attrs["num_instances"] = int(inst.max()) + 1
        f.attrs["class_names"]   = CLASS_NAMES

    print(f"  Saved {len(points):,} points, "
          f"{int(inst.max())+1} instances → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert S3DIS annotation .txt files to per-area H5 files."
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Root of S3DIS dataset (contains Area_1 … Area_6 folders)."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where Area_N.h5 files will be written."
    )
    parser.add_argument(
        "--areas", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
        metavar="N",
        help="Which areas to process (default: 1 2 3 4 5 6)."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each room as it is loaded."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for area_idx in args.areas:
        area_name = f"Area_{area_idx}"
        area_path = os.path.join(args.data_root, area_name)

        if not os.path.isdir(area_path):
            print(f"[SKIP] {area_path} not found.")
            continue

        print(f"Processing {area_name} …")
        output_path = os.path.join(args.output_dir, f"{area_name}.h5")
        process_area(area_path, output_path, verbose=args.verbose)

    print("\nDone.")


if __name__ == "__main__":
    main()