"""
SO(3) rotation utilities for point-cloud batches.

Convention throughout this project:
    points tensors are (B, 3, N)   -- channels-first, matching your training loop
    which does points_batch.permute(0, 2, 1) before feeding the model.

All angles are in RADIANS. Helpers accept per-sample angles of shape (B,)
or a single scalar broadcast to the whole batch.
"""

import math
import torch


def _rot_x(a):
    # a: (B,) radians -> (B, 3, 3)
    B = a.shape[0]
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    R = torch.stack([
        o,  z,  z,
        z,  c, -s,
        z,  s,  c,
    ], dim=-1).reshape(B, 3, 3)
    return R


def _rot_y(a):
    B = a.shape[0]
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    R = torch.stack([
        c,  z,  s,
        z,  o,  z,
       -s,  z,  c,
    ], dim=-1).reshape(B, 3, 3)
    return R


def _rot_z(a):
    B = a.shape[0]
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    R = torch.stack([
        c, -s,  z,
        s,  c,  z,
        z,  z,  o,
    ], dim=-1).reshape(B, 3, 3)
    return R


def euler_to_matrix(ax, ay, az):
    """
    Compose a batch rotation matrix from per-axis angles.
    Order: R = Rz @ Ry @ Rx  (apply X, then Y, then Z).

    ax, ay, az : each (B,) radians  OR python float (broadcast).
    returns    : (B, 3, 3)
    """
    # allow scalar broadcast: caller passes (B,) tensors normally
    R = _rot_z(az) @ _rot_y(ay) @ _rot_x(ax)
    return R


def rotate_points(points_bcn, ax, ay, az):
    """
    Rotate a (B, 3, N) point batch by per-sample Euler angles.

    points_bcn : (B, 3, N)
    ax, ay, az : (B,) radians tensors on the same device.
    returns    : (B, 3, N) rotated, same dtype/device.

    Rotation is applied about the batch centroid so translation of the
    scene origin does not interact with the rotation (matters for S3DIS
    blocks, which are not centered at the origin).
    """
    B, C, N = points_bcn.shape
    assert C == 3, f"expected 3 coord channels, got {C}"

    R = euler_to_matrix(ax, ay, az).to(points_bcn.dtype)   # (B, 3, 3)

    # center -> rotate -> uncenter, per sample
    centroid = points_bcn.mean(dim=2, keepdim=True)        # (B, 3, 1)
    centered = points_bcn - centroid                        # (B, 3, N)
    rotated = torch.bmm(R, centered)                        # (B, 3, N)
    return rotated + centroid


def sample_angles(B, device, axes=("x", "y", "z"), generator=None):
    """
    Uniform random angles in [0, 2pi) for the requested axes.
    Returns dict with keys 'x','y','z' each (B,); unused axes are zeros.
    """
    out = {}
    for k in ("x", "y", "z"):
        if k in axes:
            out[k] = torch.rand(B, device=device, generator=generator) * (2 * math.pi)
        else:
            out[k] = torch.zeros(B, device=device)
    return out


def deg2rad(x):
    return x * math.pi / 180.0


def rad2deg(x):
    return x * 180.0 / math.pi
