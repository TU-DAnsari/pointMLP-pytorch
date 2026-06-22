import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat


# ── Unchanged utilities ────────────────────────────────────────────────────────

def get_activation(activation):
    if activation.lower() == 'gelu':        return nn.GELU()
    elif activation.lower() == 'rrelu':     return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':      return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':      return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish': return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':     return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':  return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:                                   return nn.ReLU(inplace=True)


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


# ── Unchanged building blocks ──────────────────────────────────────────────────

class LocalGrouper(nn.Module):
    """Unchanged from original — builds normalised local neighbourhoods."""
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize is not None else None
        if self.normalize not in ["center", "anchor"]:
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta  = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()

        fps_idx    = farthest_point_sample(xyz, self.groups).long()
        new_xyz    = index_points(xyz, fps_idx)           # [B, S, 3]
        new_points = index_points(points, fps_idx)        # [B, S, d]

        idx             = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz     = index_points(xyz, idx)          # [B, S, k, 3]
        grouped_points  = index_points(points, idx)       # [B, S, k, d]

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, S, k, d+3]

        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            else:  # anchor
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1,
                            keepdim=True).unsqueeze(-1).unsqueeze(-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat(
            [grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)],
            dim=-1
        )
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            get_activation(activation),
        )
    def forward(self, x): return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super().__init__()
        self.act  = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, int(channel * res_expansion), kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act,
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(int(channel * res_expansion), channel, kernel_size, bias=bias),
            nn.BatchNorm1d(channel),
        )
    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1,
                 bias=True, activation='relu', use_xyz=True):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer  = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        self.operation = nn.Sequential(*[
            ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                            bias=bias, activation=activation)
            for _ in range(blocks)
        ])

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)
        x = self.transfer(x)
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(b * n, -1)
        return x.reshape(b, n, -1).permute(0, 2, 1)


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        super().__init__()
        self.operation = nn.Sequential(*[
            ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion,
                            bias=bias, activation=activation)
            for _ in range(blocks)
        ])
    def forward(self, x): return self.operation(x)


# ── NEW: Query-point grouper ───────────────────────────────────────────────────

class QueryGrouper(nn.Module):
    """
    Like LocalGrouper but for arbitrary query points that may not be in the
    surface point cloud.

    Given:
      surface_xyz   [B, N, 3]  – the object surface points (encoder anchors)
      surface_feats [B, d, N]  – their learned features (from encoder)
      query_xyz     [B, Q, 3]  – the points we want to classify

    Returns per-query feature vectors [B, d_out, Q] built by gathering the k
    nearest surface neighbours around each query point and running a small MLP.

    This is the key difference from the original decoder: instead of
    interpolating *back* to surface points, we interpolate *to* arbitrary query
    positions — allowing us to score any point in space against the learned
    object geometry.
    """
    def __init__(self, in_channel, out_channel, k=8, bias=True, activation='relu'):
        super().__init__()
        self.k = k
        # in_channel: surface feature dim d
        # we concatenate (surface_feat, relative_xyz) → d + 3
        self.mlp = nn.Sequential(
            ConvBNReLU1D(in_channel + 3, out_channel, bias=bias, activation=activation),
            ConvBNReLURes1D(out_channel, bias=bias, activation=activation),
        )

    def forward(self, surface_xyz, surface_feats, query_xyz):
        """
        surface_xyz   [B, N, 3]
        surface_feats [B, d, N]
        query_xyz     [B, Q, 3]
        Returns       [B, out_channel, Q]
        """
        B, Q, _ = query_xyz.shape
        _, d, N  = surface_feats.shape

        # k nearest surface points for each query point
        idx = knn_point(self.k, surface_xyz, query_xyz)          # [B, Q, k]
        neighbor_xyz   = index_points(surface_xyz, idx)          # [B, Q, k, 3]
        neighbor_feats = index_points(
            surface_feats.permute(0, 2, 1), idx                  # [B, N, d] → [B, Q, k, d]
        )

        # relative positions encode where each query sits w.r.t. its neighbours
        rel_xyz = query_xyz.unsqueeze(2) - neighbor_xyz          # [B, Q, k, 3]

        # concatenate geometric offset with surface feature
        x = torch.cat([neighbor_feats, rel_xyz], dim=-1)         # [B, Q, k, d+3]

        # treat (Q, k) as the "points" dimension for the shared MLP
        x = x.permute(0, 1, 3, 2).reshape(B * Q, d + 3, self.k) # [B*Q, d+3, k]
        x = self.mlp(x)                                          # [B*Q, out_channel, k]
        x = F.adaptive_max_pool1d(x, 1).view(B, Q, -1)          # [B, Q, out_channel]
        return x.permute(0, 2, 1)                                 # [B, out_channel, Q]


# ── NEW: Occupancy head ────────────────────────────────────────────────────────

class OccupancyHead(nn.Module):
    """
    Maps per-point features + optional global context → scalar occupancy in [0,1].
    Output is a raw logit; apply sigmoid externally (or use BCEWithLogitsLoss).
    """
    def __init__(self, in_channel, hidden=128, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channel, hidden, 1, bias=bias),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(hidden, 1, 1, bias=bias),   # single logit per point
        )
    def forward(self, x):
        return self.net(x).squeeze(1)              # [B, Q]


# ── Main model ─────────────────────────────────────────────────────────────────

class PointMLPOccupancy(nn.Module):
    """
    Object-specific occupancy network built on the PointMLP segmentation backbone.

    Two separate inputs at forward time
    ────────────────────────────────────
    surface_pts  [B, 3+d, N]  – the (partial) object surface as seen by the
                                depth sensor; used to build the geometric
                                context via the encoder.  'd' extra dims are
                                optional (normals, colour …).

    query_pts    [B, 3, Q]    – arbitrary points in the same coordinate frame;
                                the model predicts P(inside | geometry) for each.

    Design choices vs. the original segmentation model
    ───────────────────────────────────────────────────
    • The encoder (embedding → local_groupers → pre/pos blocks) is UNCHANGED.
      It still learns hierarchical geometric features from the surface cloud.

    • The decoder is REPLACED.  Instead of propagating features back to surface
      points via PointNetFeaturePropagation, we use QueryGrouper to project the
      learned surface features onto the query points.  This lets us score any
      point in space, not just the input surface points.

    • Global context (GMP over all encoder stages) is kept — it gives each query
      point a sense of the object's overall shape, which helps near the boundary.

    • The classifier head is replaced by OccupancyHead, which outputs a single
      logit per query point.  Use BCEWithLogitsLoss during training.
    """

    def __init__(self,
                 points=1024,
                 input_dim=3,            # 3 for xyz-only surface cloud
                 embed_dim=32,
                 groups=1,
                 res_expansion=1.0,
                 activation="relu",
                 bias=True,
                 use_xyz=True,
                 normalize="anchor",
                 dim_expansion=[2, 2],
                 pre_blocks=[2, 2],
                 pos_blocks=[2, 2],
                 k_neighbors=[24, 24],
                 reducers=[4, 4],
                 query_k=8,              # neighbours used in QueryGrouper
                 gmp_dim=32,
                 occ_hidden=128,
                 **kwargs):

        super().__init__()
        self.stages = len(pre_blocks)
        self.points = points

        # ── Encoder (unchanged from segmentation model) ──────────────────────
        self.embedding = ConvBNReLU1D(input_dim, embed_dim, bias=bias, activation=activation)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list    = nn.ModuleList()
        self.pos_blocks_list    = nn.ModuleList()

        last_channel  = embed_dim
        anchor_points = points
        en_dims       = [last_channel]

        for i in range(len(pre_blocks)):
            out_channel    = last_channel * dim_expansion[i]
            anchor_points  = anchor_points // reducers[i]
            self.local_grouper_list.append(
                LocalGrouper(last_channel, anchor_points, k_neighbors[i], use_xyz, normalize)
            )
            self.pre_blocks_list.append(
                PreExtraction(last_channel, out_channel, pre_blocks[i], groups=groups,
                              res_expansion=res_expansion, bias=bias,
                              activation=activation, use_xyz=use_xyz)
            )
            self.pos_blocks_list.append(
                PosExtraction(out_channel, pos_blocks[i], groups=groups,
                              res_expansion=res_expansion, bias=bias, activation=activation)
            )
            last_channel = out_channel
            en_dims.append(last_channel)

        # ── Global context (unchanged) ───────────────────────────────────────
        self.gmp_map_list = nn.ModuleList([
            ConvBNReLU1D(d, gmp_dim, bias=bias, activation=activation)
            for d in en_dims
        ])
        self.gmp_map_end = ConvBNReLU1D(
            gmp_dim * len(en_dims), gmp_dim, bias=bias, activation=activation
        )

        # ── NEW: query-point decoder ─────────────────────────────────────────
        # We use the *finest* encoder features (en_dims[0] = embed_dim from the
        # first stage input) to stay closest to the surface geometry.
        # en_dims[0] is the embedding dimension before any downsampling.
        # You can also experiment with en_dims[-1] (deepest) or concatenating all.
        query_feat_dim = en_dims[0]   # features from the full-resolution stage
        self.query_grouper = QueryGrouper(
            in_channel=query_feat_dim,
            out_channel=query_feat_dim,
            k=query_k,
            bias=bias,
            activation=activation,
        )

        # final per-query feature = query_feat + global_context
        self.occ_head = OccupancyHead(
            in_channel=query_feat_dim + gmp_dim,
            hidden=occ_hidden,
            bias=bias,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, surface_pts, query_pts):
        """
        surface_pts  [B, input_dim, N]  – object surface points (xyz + optional feats)
        query_pts    [B, 3, Q]          – points to classify

        Returns
        ───────
        logits  [B, Q]   raw logits; apply torch.sigmoid() for probabilities,
                         or pass directly to nn.BCEWithLogitsLoss.
        """
        # ── Encode surface geometry ──────────────────────────────────────────
        # xyz used for spatial operations, x for feature learning
        xyz = surface_pts[:, :3, :].permute(0, 2, 1)   # [B, N, 3]
        x   = self.embedding(surface_pts)               # [B, embed_dim, N]

        xyz_list = [xyz]
        x_list   = [x]

        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
            xyz_list.append(xyz)
            x_list.append(x)

        # ── Global context from all encoder stages ───────────────────────────
        gmp_list = [
            F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1)
            for i in range(len(x_list))
        ]
        global_ctx = self.gmp_map_end(torch.cat(gmp_list, dim=1))  # [B, gmp_dim, 1]

        # ── Project onto query points ────────────────────────────────────────
        # Use the full-resolution surface features (x_list[0]) so the query
        # grouper has the densest possible surface description to sample from.
        surface_xyz_fine   = xyz_list[0]   # [B, N, 3]       full resolution
        surface_feats_fine = x_list[0]     # [B, embed_dim, N]

        query_xyz = query_pts.permute(0, 2, 1)   # [B, Q, 3]
        q_feats   = self.query_grouper(
            surface_xyz_fine, surface_feats_fine, query_xyz
        )                                          # [B, embed_dim, Q]

        # ── Concatenate global context and classify ──────────────────────────
        Q       = q_feats.shape[-1]
        ctx_exp = global_ctx.expand(-1, -1, Q)    # [B, gmp_dim, Q]
        x_out   = torch.cat([q_feats, ctx_exp], dim=1)

        logits = self.occ_head(x_out)              # [B, Q]
        return logits


# ── Training helper ───────────────────────────────────────────────────────────

class OccupancyLoss(nn.Module):
    """
    Weighted BCE loss.

    near_surface_weight > 1.0 upweights query points that are close to the
    surface — these are the hard cases where the boundary must be sharp.
    Pass near_surface_mask [B, Q] bool tensor (True = near surface) built
    from your ground-truth SDF during data generation.
    """
    def __init__(self, near_surface_weight=5.0, pos_weight=1.0):
        super().__init__()
        self.nsw = near_surface_weight
        # pos_weight handles class imbalance: if you sample more outside points
        # than inside, set pos_weight > 1 to upweight the inside class.
        self.register_buffer('pw', torch.tensor([pos_weight]))

    def forward(self, logits, targets, near_surface_mask=None):
        """
        logits             [B, Q]  raw model output
        targets            [B, Q]  float, 1.0 = inside, 0.0 = outside
        near_surface_mask  [B, Q]  bool, optional
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pw.to(logits.device),
            reduction='none',
        )                                                # [B, Q]

        if near_surface_mask is not None:
            weight = torch.ones_like(bce)
            weight[near_surface_mask] = self.nsw
            bce = bce * weight

        return bce.mean()


# ── Convenience constructors ───────────────────────────────────────────────────

def pointMLPOccupancy(num_points=1024, input_dim=3, **kwargs) -> PointMLPOccupancy:
    """Standard model — balanced accuracy / speed."""
    return PointMLPOccupancy(
        points=num_points,
        input_dim=input_dim,
        embed_dim=64,
        dim_expansion=[2, 2, 2],
        pre_blocks=[2, 2, 2],
        pos_blocks=[2, 2, 2],
        k_neighbors=[24, 24, 24],
        reducers=[4, 4, 4],
        query_k=8,
        gmp_dim=64,
        occ_hidden=128,
        **kwargs,
    )


def pointMLPOccupancySmall(num_points=1024, input_dim=3, **kwargs) -> PointMLPOccupancy:
    """Lightweight model for fast iteration / small objects."""
    return PointMLPOccupancy(
        points=num_points,
        input_dim=input_dim,
        embed_dim=32,
        dim_expansion=[2, 2],
        pre_blocks=[2, 2],
        pos_blocks=[2, 2],
        k_neighbors=[24, 24],
        reducers=[4, 4],
        query_k=8,
        gmp_dim=32,
        occ_hidden=64,
        **kwargs,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B, N, Q = 2, 1024, 2048

    # Surface points: xyz only (input_dim=3)
    surface = torch.rand(B, 3, N)

    # Query points: xyz only
    queries = torch.rand(B, 3, Q)

    # Ground-truth occupancy labels (1 = inside, 0 = outside)
    labels  = (torch.rand(B, Q) > 0.5).float()

    # Near-surface mask: points within 0.05 of the boundary
    # (in practice, derive from your SDF at data-generation time)
    near_surface = torch.rand(B, Q) < 0.2   # ~20 % of query points

    model     = pointMLPOccupancySmall(num_points=N, input_dim=3)
    criterion = OccupancyLoss(near_surface_weight=5.0, pos_weight=2.0)

    logits = model(surface, queries)          # [B, Q]
    loss   = criterion(logits, labels, near_surface)
    probs  = torch.sigmoid(logits)

    print(f"logits shape : {logits.shape}")   # [2, 2048]
    print(f"probs  range : [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"loss         : {loss.item():.4f}")