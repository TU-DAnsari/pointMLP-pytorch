import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

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


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize is not None else None
        if self.normalize not in ["center", "anchor"]:
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()

        fps_idx = farthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
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
            nn.Conv1d(channel, int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(int(channel * res_expansion), channel, kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(channel, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(int(channel * res_expansion), channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1,
                 bias=True, activation='relu', use_xyz=True):
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        return self.operation(x)


class PointMLPEncoder(nn.Module):
    """
    Encodes a point cloud (B, input_dim, N) → latent vector (B, latent_dim).

    Architecture mirrors pointMLPSmall's encoder:
      embedding → [LocalGrouper → PreExtraction → PosExtraction] × stages
      → multi-scale global max-pool → gmp_map_end → latent
    """

    def __init__(self, points=1024, input_dim=6, embed_dim=32, groups=1,
                 res_expansion=1.0, activation="relu", bias=True,
                 use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2], pre_blocks=[2, 2], pos_blocks=[2, 2],
                 k_neighbors=[24, 24], reducers=[4, 4],
                 gmp_dim=32):
        super(PointMLPEncoder, self).__init__()

        self.stages = len(pre_blocks)
        self.points = points
        self.embedding = ConvBNReLU1D(input_dim, embed_dim, bias=bias, activation=activation)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = points
        en_dims = [last_channel]

        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            anchor_points = anchor_points // reducers[i]
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

        # Multi-scale global max-pool (same as original PointMLP)
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))

        # Final projection: concatenated multi-scale features → latent_dim
        self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(en_dims), gmp_dim, bias=bias, activation=activation)
        self.latent_dim = gmp_dim

    def forward(self, sampling_input, model_input):
        """
        Args:
            sampling_input: (B, 3, N)  – XYZ used for FPS/KNN
            model_input:    (B, input_dim, N) – full feature used for embedding
        Returns:
            latent: (B, latent_dim)
        """
        xyz = sampling_input.permute(0, 2, 1)       # (B, N, 3)
        x = self.embedding(model_input)              # (B, embed_dim, N)

        x_list = [x]

        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
            x_list.append(x)

        # Multi-scale global max-pool
        gmp_list = []
        for i, xi in enumerate(x_list):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](xi), 1))  # (B, gmp_dim, 1)

        global_feat = self.gmp_map_end(torch.cat(gmp_list, dim=1))  # (B, gmp_dim, 1)
        latent = global_feat.squeeze(-1)                              # (B, latent_dim)
        return latent


# ---------------------------------------------------------------------------
# Folding Decoder  (latent → reconstructed point cloud)
# ---------------------------------------------------------------------------

class FoldingDecoder(nn.Module):
    """
    Reconstructs N points from a latent vector using two folding steps.

    Step 1: latent + 2-D grid  → intermediate 3-D surface
    Step 2: latent + step-1 output → refined 3-D points

    Args:
        latent_dim: dimension of the input latent vector
        num_points: number of output points to reconstruct
        fold_hidden: hidden channel width inside each folding MLP
        grid_size: side length of the 2-D seed grid (grid_size² ≥ num_points)
    """

    def __init__(self, latent_dim=32, num_points=1024,
                 fold_hidden=128, grid_size=32):
        super(FoldingDecoder, self).__init__()

        assert grid_size * grid_size >= num_points, \
            "grid_size² must be ≥ num_points"

        self.num_points = num_points
        self.latent_dim = latent_dim

        # 2-D grid seed: (grid_size², 2)
        gs = grid_size
        xs = torch.linspace(-0.5, 0.5, gs)
        ys = torch.linspace(-0.5, 0.5, gs)
        grid_y, grid_x = torch.meshgrid(xs, ys, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (gs², 2)
        grid = grid[:num_points]                                       # trim to exact N
        self.register_buffer('grid', grid)                             # (N, 2)

        # Fold 1: (latent_dim + 2) → 3
        fold1_in = latent_dim + 2
        self.fold1 = nn.Sequential(
            nn.Conv1d(fold1_in, fold_hidden, 1),
            nn.BatchNorm1d(fold_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(fold_hidden, fold_hidden, 1),
            nn.BatchNorm1d(fold_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(fold_hidden, 3, 1),
        )

        # Fold 2: (latent_dim + 3) → 3
        fold2_in = latent_dim + 3
        self.fold2 = nn.Sequential(
            nn.Conv1d(fold2_in, fold_hidden, 1),
            nn.BatchNorm1d(fold_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(fold_hidden, fold_hidden, 1),
            nn.BatchNorm1d(fold_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(fold_hidden, 3, 1),
        )

    def forward(self, latent):
        """
        Args:
            latent: (B, latent_dim)
        Returns:
            reconstructed: (B, num_points, 3)
        """
        B = latent.size(0)
        N = self.num_points

        # Tile latent: (B, latent_dim) → (B, latent_dim, N)
        lat = latent.unsqueeze(-1).expand(-1, -1, N)    # (B, latent_dim, N)

        # Tile grid: (N, 2) → (B, 2, N)
        grid = self.grid.unsqueeze(0).expand(B, -1, -1) # (B, N, 2)
        grid = grid.permute(0, 2, 1)                    # (B, 2, N)

        # Fold 1
        x1 = torch.cat([lat, grid], dim=1)              # (B, latent_dim+2, N)
        x1 = self.fold1(x1)                             # (B, 3, N)

        # Fold 2
        x2 = torch.cat([lat, x1], dim=1)               # (B, latent_dim+3, N)
        x2 = self.fold2(x2)                             # (B, 3, N)

        reconstructed = x2.permute(0, 2, 1)             # (B, N, 3)
        return reconstructed


# ---------------------------------------------------------------------------
# Full Autoencoder
# ---------------------------------------------------------------------------

class PointMLPAutoEncoder(nn.Module):
    """
    Point cloud autoencoder built on the pointMLPSmall architecture.

    Encoder: PointMLPEncoder  (multi-scale local feature aggregation → global latent)
    Decoder: FoldingDecoder   (latent → reconstructed point cloud via 2-fold MLP)

    Typical use
    -----------
    model = pointMLPSmallAutoEncoder(num_points=1024, input_dim=6)

    # Forward pass
    xyz   = data[:, :3, :]   # (B, 3, N)
    feat  = data             # (B, 6, N)  (or same as xyz if input_dim=3)
    recon = model(xyz, feat) # (B, N, 3)  reconstructed XYZ

    # Loss (Chamfer distance recommended)
    loss  = chamfer_distance(recon, target_xyz)
    """

    def __init__(self, num_points=1024, input_dim=6, embed_dim=32, groups=1,
                 res_expansion=1.0, activation="relu", bias=True,
                 use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2], pre_blocks=[2, 2], pos_blocks=[2, 2],
                 k_neighbors=[24, 24], reducers=[4, 4],
                 gmp_dim=32, fold_hidden=128, grid_size=32):
        super(PointMLPAutoEncoder, self).__init__()

        self.encoder = PointMLPEncoder(
            points=num_points, input_dim=input_dim, embed_dim=embed_dim,
            groups=groups, res_expansion=res_expansion, activation=activation,
            bias=bias, use_xyz=use_xyz, normalize=normalize,
            dim_expansion=dim_expansion, pre_blocks=pre_blocks,
            pos_blocks=pos_blocks, k_neighbors=k_neighbors,
            reducers=reducers, gmp_dim=gmp_dim
        )

        self.decoder = FoldingDecoder(
            latent_dim=gmp_dim,
            num_points=num_points,
            fold_hidden=fold_hidden,
            grid_size=grid_size
        )

    def encode(self, sampling_input, model_input):
        """Returns the latent vector (B, latent_dim)."""
        return self.encoder(sampling_input, model_input)

    def decode(self, latent):
        """Returns reconstructed point cloud (B, num_points, 3)."""
        return self.decoder(latent)

    def forward(self, sampling_input, model_input):
        """
        Args:
            sampling_input: (B, 3, N)        – XYZ coords for FPS/KNN
            model_input:    (B, input_dim, N) – full per-point features
        Returns:
            reconstructed:  (B, N, 3)         – reconstructed XYZ
        """
        latent = self.encode(sampling_input, model_input)
        reconstructed = self.decode(latent)
        return reconstructed


# ---------------------------------------------------------------------------
# Factory function (mirrors pointMLPSmall signature)
# ---------------------------------------------------------------------------

def pointMLPSmallAutoEncoder(num_points=1024, input_dim=3, **kwargs) -> PointMLPAutoEncoder:
    """
    Drop-in replacement for pointMLPSmall, repurposed as an autoencoder.

    Latent dimension = gmp_dim = 32  (same as the original gmp_dim).
    Decoder uses a 2-step folding MLP with a 32×32 seed grid.
    """
    return PointMLPAutoEncoder(
        num_points=num_points,
        input_dim=input_dim,
        embed_dim=32,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=True,
        use_xyz=False,
        normalize="anchor",
        dim_expansion=[2, 2],
        pre_blocks=[2, 2],
        pos_blocks=[2, 2],
        k_neighbors=[24, 24],
        reducers=[4, 4],
        gmp_dim=32,
        fold_hidden=128,
        grid_size=32,
        **kwargs
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, N = 2, 1024
    xyz  = torch.rand(B, 3, N)   # XYZ only for FPS/KNN
    feat = torch.rand(B, 6, N)   # full 6-dim input features

    model = pointMLPSmallAutoEncoder(num_points=N, input_dim=6)
    print(f"Encoder latent dim : {model.encoder.latent_dim}")
    print(f"Total parameters   : {sum(p.numel() for p in model.parameters()):,}")

    recon = model(xyz, feat)
    print(f"Input shape        : {feat.shape}")
    print(f"Reconstructed shape: {recon.shape}")   # expect (2, 1024, 3)