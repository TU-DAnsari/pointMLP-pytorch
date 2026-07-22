"""
Optional: explicit decoder-FEATURE invariance loss.

NEAS alone makes seg *outputs* rotation-robust. If you want the decoder
FEATURES themselves invariant (your stated end goal), add this term.

Idea:  features(R.X) should equal features(X), up to the point
permutation that the rotation + FPS induces. Since FPS on the rotated
cloud may keep a different point subset, we align by nearest-neighbor
correspondence in the ORIGINAL coordinate frame before comparing.

This needs the encoder+seg-head split you already have:
    enc_out = model.encoder(samp, feat)
    dense   = model.seg_head.decode(enc_out)   # (B, C, N) per-point features

Cost: one extra encode+decode of the rotated batch per step. Toggle it
with a weight; start small (e.g. 0.1) so it doesn't dominate NLL.
"""

import torch
import torch.nn.functional as F

from .rotation import rotate_points


def _decode_features(model, sampling_in, model_in):
    """Return dense per-point decoder features (B, C, N)."""
    enc = model.encoder(sampling_in, model_in)
    return model.seg_head.decode(enc)          # (B, C, N)


def feature_consistency_loss(model, base_sampling, base_model_in,
                             ax, ay, az):
    """
    L_feat = mean || f(X) - align(f(R.X)) ||^2  over points.

    Because your decoder interpolates back to the FULL input resolution
    (PointNetFeaturePropagation upsamples to N original points), the
    point ordering of f(X) and f(R.X) is IDENTICAL -- both are indexed by
    the original input points, just rotated in space. So no re-matching is
    needed here: point i in f(R.X) corresponds to point i in f(X).

    (If you ever compare features at a *sub-sampled* stage instead, you'd
    need the NN-alignment; kept simple here because decode() is full-res.)
    """
    f_base = _decode_features(model, base_sampling, base_model_in)      # (B,C,N)

    rot_sampling = rotate_points(base_sampling, ax, ay, az)
    rot_model_in = rot_sampling if base_model_in is base_sampling \
        else rotate_points(base_model_in, ax, ay, az)
    f_rot = _decode_features(model, rot_sampling, rot_model_in)         # (B,C,N)

    return F.mse_loss(f_rot, f_base)
