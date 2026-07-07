import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, -1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        fps_idx = farthest_point_sample(xyz, self.groups).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


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
    def __init__(self, in_channel, coord_channel, out_channel, k=8, bias=True, activation='relu'):
        super().__init__()
        self.k = k
        self.coord_channel = coord_channel
        # in_channel: surface feature dim d
        # we concatenate (surface_feat, relative_xyz) → d + 3
        self.mlp = nn.Sequential(
            ConvBNReLU1D(in_channel + coord_channel, out_channel, bias=bias, activation=activation),
            ConvBNReLURes1D(out_channel, bias=bias, activation=activation),
        )

    def forward(self, surface_xyz, surface_feats, query_xyz):
        """
        surface_xyz   [B, N, 3]
        surface_feats [B, d, N]
        query_xyz     [B, Q, 3]
        Returns       [B, out_channel, Q]
        """
        B, Q, C = query_xyz.shape
        _, d, N  = surface_feats.shape

        # k nearest surface points for each query point
        idx = knn_point(self.k, query_xyz, surface_xyz)          # [B, Q, k]
        neighbor_xyz   = index_points(surface_xyz, idx)          # [B, Q, k, 3]
        neighbor_feats = index_points(
            surface_feats.permute(0, 2, 1), idx                  # [B, N, d] → [B, Q, k, d]
        )

        # relative positions encode where each query sits w.r.t. its neighbours
        rel_xyz = query_xyz.unsqueeze(2) - neighbor_xyz          # [B, Q, k, 3]

        # concatenate geometric offset with surface feature
        x = torch.cat([neighbor_feats, rel_xyz], dim=-1)         # [B, Q, k, d+3]

        # treat (Q, k) as the "points" dimension for the shared MLP
        x = x.permute(0, 1, 3, 2).reshape(B * Q, d + C, self.k) # [B*Q, d+3, k]
        x = self.mlp(x)                                          # [B*Q, out_channel, k]
        x = F.adaptive_max_pool1d(x, 1).view(B, Q, -1)          # [B, Q, out_channel]
        return x.permute(0, 2, 1)                                 # [B, out_channel, Q]


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


class PointMLP(nn.Module):
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
                 input_dim=3,
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
                 query_k=8,
                 gmp_dim=32,
                 occ_hidden=128,
                 **kwargs):

        super().__init__()
        self.stages = len(pre_blocks)
        self.points = points

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
        query_feat_dim = en_dims[0]   
        self.query_grouper = QueryGrouper(
            in_channel=query_feat_dim,
            out_channel=query_feat_dim,
            coord_channel=input_dim,
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
        xyz = surface_pts.permute(0, 2, 1)   # [B, N, 3]
        x = self.embedding(surface_pts)               # [B, embed_dim, N]

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
        Q = q_feats.shape[-1]
        ctx_exp = global_ctx.expand(-1, -1, Q)    # [B, gmp_dim, Q]
        x_out = torch.cat([q_feats, ctx_exp], dim=1)

        logits = self.occ_head(x_out)              # [B, Q]
        return logits


def pointMLPOccupancy(num_points=1024, input_dim=3, **kwargs) -> PointMLP:
    """Standard model — balanced accuracy / speed."""
    return PointMLP(
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
        use_xyz=False,
        occ_hidden=128,
        **kwargs,
    )


def pointMLPOccupancySmall(num_points=1024, input_dim=3, **kwargs) -> PointMLP:
    """Lightweight model for fast iteration / small objects."""
    return PointMLP(
        points=num_points,
        input_dim=input_dim,
        embed_dim=32,
        dim_expansion=[2, 2],
        pre_blocks=[2, 2],
        pos_blocks=[2, 2],
        k_neighbors=[16, 16],
        reducers=[2, 2],
        query_k=2,
        gmp_dim=32,
        use_xyz=False,
        occ_hidden=64,
        **kwargs,
    )