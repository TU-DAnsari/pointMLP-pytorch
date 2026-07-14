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


class QueryGrouper(nn.Module):
    def __init__(self, in_channel, out_channel, k=8, bias=True, activation='relu'):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            ConvBNReLU1D(in_channel, int(2*in_channel), bias=bias, activation=activation),
            ConvBNReLU1D(int(2*in_channel), out_channel, bias=bias, activation=activation),
            ConvBNReLURes1D(out_channel, bias=bias, activation=activation),
        )

    def forward(self, target_feats, source_xyz, source_feats, proxy_xyz):
        B, N_PROXY, C = proxy_xyz.shape
        _, N_SOURCE, d  = source_feats.shape

        # idx = knn_point(self.k, proxy_xyz, source_xyz)
        # neighbor_xyz = index_points(source_xyz, idx)
        # neighbor_feats = index_points(source_feats, idx)
        # rel_xyz = proxy_xyz.unsqueeze(2) - neighbor_xyz
        # x = torch.cat([neighbor_feats, rel_xyz], dim=-1)

        idx_xyz = knn_point(self.k, proxy_xyz, source_xyz)
        idx_feats = knn_point(self.k, target_feats, source_feats)
        neighbor_xyz = index_points(source_xyz, idx_xyz)
        neighbor_feats = index_points(source_feats, idx_feats)
        rel_xyz = proxy_xyz.unsqueeze(2) - neighbor_xyz
        rel_feats = target_feats.unsqueeze(2) - neighbor_feats
        x = torch.cat([rel_feats, rel_xyz], dim=-1)

        x = x.permute(0, 1, 3, 2).reshape(B * N_PROXY, d + C, self.k)
        x = self.mlp(x)
        x = F.adaptive_max_pool1d(x, 1).view(B, N_PROXY, -1)
        return x.permute(0, 2, 1)       


class OccupancyHead(nn.Module):
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
    def __init__(self,
                 xyz_dim=3,
                 feature_dim=64,
                 activation="relu",
                 bias=True,
                 query_k=8,
                 occ_hidden=128,
                 **kwargs):

        super().__init__()

        self.query_grouper = QueryGrouper(
            in_channel=xyz_dim+feature_dim,
            out_channel=32,
            k=query_k,
            bias=bias,
            activation=activation,
        )

        self.occ_head = OccupancyHead(
            in_channel=32,
            hidden=occ_hidden,
            bias=bias,
        )

    def forward(self, target_fts, src_pts, src_fts, proxy_pts):
        q_feats = self.query_grouper(
            target_fts.permute(0, 2, 1), 
            src_pts.permute(0, 2, 1), 
            src_fts.permute(0, 2, 1), 
            proxy_pts.permute(0, 2, 1)
        )

        logits = self.occ_head(q_feats)
        return logits


def pointMLPOccupancyHead(xyz_dim=3, feature_dim=64, **kwargs) -> PointMLP:
    """Lightweight model for fast iteration / small objects."""
    return PointMLP(
        xyz_dim=xyz_dim,
        feature_dim=feature_dim,
        activation="relu",
        bias=True,
        query_k=2,
        occ_hidden=64,
        **kwargs,
    )