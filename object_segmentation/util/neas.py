"""
NEAS: NTK-based Estimate for Angle Selection.

Implements the training-time angle search from
"Rotation-Invariant Point Cloud Segmentation via Neural Tangent
Kernel-based Angle Selection" (Zhang et al., ICASSP 2026),
adapted to a PointMLP segmentation model.

Core idea (paper Eq. 1-2):
    NTK estimate  k(X, X') ~= <G(X), G(X')>
where G(.) is the gradient of the network's *main operations*
(here: Conv1d / Linear weights) w.r.t. a scalar summary of the
forward pass on that batch. We pick the rotation angle that
MAXIMISES k(X, X')  (paper Sec 3.1, "Magnitude of the Gradients":
larger NTK -> larger effective step -> better convergence).

Binary Approximation (Alg. 1) avoids scanning the infinite angle
set: a divide-and-conquer search per axis, I iterations.

NOTE on faithfulness / choices:
  * The paper is vague on what scalar produces G(.). Jacot et al.'s
    NTK uses grad of the network *output*. For segmentation the
    output is per-point log-probs, so we reduce it to a scalar with
    the mean over the predicted-class log-prob (a stable, label-free
    summary). You can swap `_scalar_summary` for the training loss if
    you prefer -- both are defensible readings; this one needs no
    labels, so the angle search stays unsupervised like the paper's.
  * Gradients are taken over a *subset* of parameters (conv/linear
    weights) to keep the dot product cheap, matching "extract the
    gradients of the main operations (e.g. 3D Conv layers)".
"""

import math
import torch

from .rotation import rotate_points, deg2rad


# ----------------------------------------------------------------------
# Selecting which parameters count as "main operations"
# ----------------------------------------------------------------------
def select_ntk_params(model, include=("Conv1d", "Linear")):
    """
    Return a list of (name, param) for weight tensors of the main
    operation layers. We skip biases and normalization params -- the
    paper points the NTK at the conv/linear layers specifically.
    """
    wanted = []
    include = tuple(include)
    for module_name, module in model.named_modules():
        if type(module).__name__ in include:
            w = getattr(module, "weight", None)
            if w is not None and w.requires_grad:
                wanted.append((module_name + ".weight", w))
    return wanted


# ----------------------------------------------------------------------
# Gradient extraction
# ----------------------------------------------------------------------
def _scalar_summary(seg_logprob):
    """
    Reduce model output (B, N, num_classes log-softmax) to a scalar so
    we can take a single backward pass. Uses the max-over-class log-prob
    averaged over all points -- label-free, and reflects how confidently
    the model commits, which is what the NTK gradient direction captures.
    """
    # max over class dim, mean over points & batch
    return seg_logprob.max(dim=-1).values.mean()


def batch_gradient(model, ntk_params, sampling_in, model_in):
    """
    One forward+backward on the given batch; return a flat gradient
    vector over the selected params (detached, on-device).

    sampling_in, model_in : (B, 3, N) tensors already on device.
    """
    model.zero_grad(set_to_none=True)
    out = model(sampling_in, model_in)           # (B, N, num_classes)
    scalar = _scalar_summary(out)
    grads = torch.autograd.grad(
        scalar,
        [p for _, p in ntk_params],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    flat = []
    for g, (_, p) in zip(grads, ntk_params):
        if g is None:
            flat.append(torch.zeros(p.numel(), device=p.device))
        else:
            flat.append(g.reshape(-1))
    return torch.cat(flat)


def ntk_estimate(model, ntk_params, base_sampling, base_model_in,
                 ax, ay, az, base_grad=None):
    """
    k(X, X') ~= <G(X), G(X')>   (paper Eq. 1)

    base_*   : the original (unrotated) batch, (B, 3, N).
    ax,ay,az : (B,) radian angles defining X'.
    base_grad: optionally pass a precomputed G(X) to avoid recompute.

    We rotate BOTH the sampling coords and the model input. In your
    setup use_xyz=False, so model_in IS the coordinates -> rotate the
    same way. If you later feed color/normal features, only the xyz
    channels should be rotated; adapt here.
    """
    if base_grad is None:
        base_grad = batch_gradient(model, ntk_params, base_sampling, base_model_in)

    rot_sampling = rotate_points(base_sampling, ax, ay, az)
    # model_in may or may not be the same tensor as sampling; handle both
    if base_model_in is base_sampling:
        rot_model_in = rot_sampling
    else:
        rot_model_in = rotate_points(base_model_in, ax, ay, az)

    g_rot = batch_gradient(model, ntk_params, rot_sampling, rot_model_in)
    return torch.dot(base_grad, g_rot).item()


# ----------------------------------------------------------------------
# Binary Approximation (Algorithm 1)
# ----------------------------------------------------------------------
def binary_approximation(model, ntk_params, base_sampling, base_model_in,
                         axis, fixed, a0_deg, max_iter, base_grad):
    """
    Alg. 1: divide-and-conquer search on ONE axis.

    axis   : 'x' | 'y' | 'z'  -- the axis being searched.
    fixed  : dict of the other two axes' angles in DEGREES (floats).
    a0_deg : initial angle for this axis (degrees).
    returns: chosen angle for this axis (degrees).

    At each step compare a_l = a/2 vs a_r = (a+180)/2 by NTK and move
    toward the higher one; break if neither improves (paper's 'otherwise').
    We maximise NTK, per the paper's magnitude argument.
    """
    device = base_sampling.device
    B = base_sampling.shape[0]

    def angles_for(a_deg):
        # build (B,) radian tensors for all three axes
        vals = dict(fixed)
        vals[axis] = a_deg
        ax = torch.full((B,), deg2rad(vals["x"]), device=device)
        ay = torch.full((B,), deg2rad(vals["y"]), device=device)
        az = torch.full((B,), deg2rad(vals["z"]), device=device)
        return ax, ay, az

    def k_of(a_deg):
        ax, ay, az = angles_for(a_deg)
        return ntk_estimate(model, ntk_params, base_sampling, base_model_in,
                            ax, ay, az, base_grad=base_grad)

    a = float(a0_deg)
    for _ in range(max_iter):
        a_l = a / 2.0
        a_r = (a + 180.0) / 2.0
        k_l = k_of(a_l)
        k_r = k_of(a_r)
        if k_l > k_r:
            a = a_l
        elif k_r > k_l:
            a = a_r
        else:
            break
    return a % 360.0


@torch.no_grad()
def _noop():
    pass


def search_angles(model, base_sampling, base_model_in,
                  axes=("x", "y", "z"), init_deg=None, max_iter=5,
                  ntk_params=None):
    """
    Full NEAS angle search for one batch (paper Eq. 4):
        a_chosn = (BA(a^x), BA(a^y), BA(a^z))

    Runs BA on each requested axis in turn, holding the others fixed at
    their current best. Returns (ax, ay, az) as (B,) RADIAN tensors ready
    for rotate_points, plus the chosen degrees dict for logging.

    IMPORTANT: this does many forward+backward passes (roughly
    2 * max_iter per axis). It's gradient computation only -- do NOT
    call opt.step() inside. Wrap the *training* step separately.
    Call model.zero_grad() afterwards; the search dirties .grad.
    """
    device = base_sampling.device
    B = base_sampling.shape[0]

    if ntk_params is None:
        ntk_params = select_ntk_params(model)

    if init_deg is None:
        # random init per axis, shared across batch (paper uses a random
        # a_init); a single scalar per axis keeps the search cheap.
        init_deg = {k: float(torch.rand(1).item() * 360.0) for k in ("x", "y", "z")}
    chosen = {"x": 0.0, "y": 0.0, "z": 0.0}
    for k in ("x", "y", "z"):
        chosen[k] = init_deg.get(k, 0.0)

    # G(X) computed once; reused for every candidate (X fixed during search)
    base_grad = batch_gradient(model, ntk_params, base_sampling, base_model_in)

    for axis in axes:
        fixed = {k: chosen[k] for k in ("x", "y", "z") if k != axis}
        chosen[axis] = binary_approximation(
            model, ntk_params, base_sampling, base_model_in,
            axis=axis, fixed=fixed, a0_deg=chosen[axis],
            max_iter=max_iter, base_grad=base_grad,
        )

    model.zero_grad(set_to_none=True)

    ax = torch.full((B,), deg2rad(chosen["x"]), device=device)
    ay = torch.full((B,), deg2rad(chosen["y"]), device=device)
    az = torch.full((B,), deg2rad(chosen["z"]), device=device)
    return (ax, ay, az), chosen
