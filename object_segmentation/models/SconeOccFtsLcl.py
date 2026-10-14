import torch.nn.functional as F
from .Attention import *
from .utils import get_knn_points, get_knn_idx
from pytorch3d.ops import knn_gather, sample_farthest_points


class PCTransformer(nn.Module):
    def __init__(self, seq_len, pts_dim=3, pts_embedding_dim=256, feature_dim=512,
                 concatenate_input=True,
                 n_code=2, n_heads=4, FF=True, gelu=True,
                 dropout=None):
        """
        Transformer unit used to encode local point cloud neighborhoods.

        :param seq_len: (int) Length of input point cloud sequence.
        :param pts_dim: (int) Dimension of input points. Usually, pts_dim=3.
        :param pts_embedding_dim: (int) Dimension of point embeddings.
        :param feature_dim: (int) Dimension of output features.
        :param concatenate_input: (bool) If True, the model concatenates the raw input to their initial embedding.
        :param n_code: (int) Number of Multi-Head Self Attention units.
        :param n_heads: (int) Number of heads in Multi-Head Self Attention units.
        :param FF: (bool) If True, the Transformer encoder applies a Feed Forward unit after
        each Multi-Head Self-Attention unit.
        :param gelu: (bool) If True, the model uses GELU non-linearity. Else, it uses ReLU.
        :param dropout: Dropout module to apply on computed features.
        """
        super(PCTransformer, self).__init__()

        # Parameters
        self.seq_len = seq_len
        self.pts_dim = pts_dim
        self.pts_embedding_dim = pts_embedding_dim

        self.n_code = n_code
        self.n_heads = n_heads
        self.FF = FF
        self.gelu = gelu

        self.feature_dim = feature_dim

        self.dropout = dropout

        # Point embedding
        self.embedding = Embedding(input_dim=pts_dim, output_dim=pts_embedding_dim,
                                   dropout=None, gelu=gelu,
                                   global_feature=False, additional_feature_dim=0,
                                   concatenate_input=concatenate_input,
                                   k_for_knn=0)

        # Point cloud encoder Backbone
        encoders = []
        for i in range(n_code):
            encoders += [Encoder(seq_len=seq_len,
                                 embedding_dim=pts_embedding_dim,
                                 qk_dim=pts_embedding_dim // 4,
                                 n_heads=n_heads,
                                 dropout=dropout,
                                 gelu=gelu,
                                 FF=FF)]

        self.encoders = nn.ModuleList(encoders)
        self.norm = nn.LayerNorm(self.pts_embedding_dim)
        self.linear0 = nn.Linear(self.pts_embedding_dim, feature_dim // 2)

        self.max_pool = nn.functional.max_pool1d  # kernel_size will be <= seq_len
        self.avg_pool = nn.functional.avg_pool1d  # kernel_size will be <= seq_len

    def forward(self, pc, mask=None):
        """
        Forward pass.

        :param pc: (Tensor) Point cloud tensor with shape (n_clouds, seq_len, pts_dim)
        :param mask: (Tensor) Mask tensor with shape (batch_size, seq_len, seq_len). Optional.
        :return: (Tensor) Output features with shape (batch_size, features_dim)
        """
        n_clouds, seq_len = pc.shape[0], pc.shape[1]

        pc_embd = self.embedding(pc)
        for encoder in self.encoders:
            pc_embd = encoder(pc_embd, mask=mask)
        features = self.norm(pc_embd)

        # Linear transformation and pooling operations for downstream tasks
        features = self.linear0(features)

        features = features.transpose(dim0=-1, dim1=-2)
        features = torch.cat((self.max_pool(input=features, kernel_size=seq_len),
                              self.avg_pool(input=features, kernel_size=seq_len)), dim=-2)

        features = features.view(n_clouds, self.feature_dim)

        return features


class SconeOcc(nn.Module):
    def __init__(self, pts_dim=3, pts_embedding_dim=128,
                 concatenate_input=True,
                 n_code=2, n_heads=4, FF=True, gelu=True,
                 n_scale=3, local_feature_dim=256, k_for_knn=16,
                 x_dim=3,
                 output_dim=1,
                 dropout=None):
        """
        Main class for SCONE's occupancy probability prediction module.
        A neural model that predicts a vector field as an implicit function, depending on an input point cloud.
        Transformers with Multi-Scale Neighborhood features (MSN features) are used to encode the point cloud,
        depending on the query point x.

        :param pts_dim: (int) Dimension of points in the input point cloud.
        :param pts_embedding_dim: (int) Dimension of embedded point cloud.
        :param concatenate_input: (bool) If True, concatenates raw input points to the point embeddings
        in the point cloud.
        :param n_code: (int) Number of Multi-Head Self Attention units.
        :param n_heads: (int) Number of heads in Multi-Head Self Attention units.
        :param FF: (bool) If True, the Transformer encoder applies a Feed Forward unit after
        each Multi-Head Self-Attention unit.
        :param gelu: (bool) If True, the model uses GELU non-linearity. Else, it uses ReLU.
        :param n_scale: (int) Number of scales to compute neighborhood features in the point cloud.
        :param local_feature_dim: (int) Dimension of point cloud neighborhood features.
        :param k_for_knn: (int) Number of neighbors to use when computing a neighborhood feature.
        :param x_dim: (int) Dimension of query point x.
        :param output_dim: (int) Dimension of the output vector field.
        :param dropout: Dropout module to apply on computed embeddings.
        """
        super(SconeOcc, self).__init__()

        # Parameters
        self.pts_dim = pts_dim
        self.pts_embedding_dim = pts_embedding_dim

        self.n_code = n_code
        self.n_heads = n_heads
        self.FF = FF
        self.gelu = gelu

        self.n_scale = n_scale

        self.x_dim = x_dim

        self.output_dim = output_dim

        self.dropout = dropout

        self.encoding_dim = pts_embedding_dim

        self.k_for_knn = k_for_knn

        self.local_feature_dim = local_feature_dim
        self.all_feature_size = self.n_scale * self.local_feature_dim

        self.ds_factor = 4                                   # same for both clouds
        self.tau = nn.Parameter(torch.tensor(0.1))           # softmax temperature for matches
        geo_dim = self.n_scale * self.local_feature_dim
        self.seg_head = nn.Sequential(
            nn.Linear(3 * geo_dim + self.k_for_knn, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, self.output_dim))

        # Local point cloud transformers
        local_transformers = []
        for i in range(n_scale):
            local_transformers += [PCTransformer(seq_len=k_for_knn, pts_dim=pts_dim,
                                                 pts_embedding_dim=pts_embedding_dim, feature_dim=local_feature_dim,
                                                 concatenate_input=concatenate_input,
                                                 n_code=n_code, n_heads=n_heads, FF=FF, gelu=gelu,
                                                 dropout=dropout)]
        self.local_transformers = nn.ModuleList(local_transformers)

        # Point cloud feature extraction
        self.max_pool = nn.functional.max_pool1d  # kernel_size will be <= seq_len
        self.avg_pool = nn.functional.avg_pool1d  # kernel_size will be <= seq_len

        # MLP for occupancy probability prediction
        self.linear1 = nn.Linear(self.all_feature_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, output_dim)

        if gelu:
            self.non_linear1 = nn.GELU()
            self.non_linear2 = nn.GELU()
            self.non_linear3 = nn.GELU()
        else:
            self.non_linear1 = nn.ReLU(inplace=False)
            self.non_linear2 = nn.ReLU(inplace=False)
            self.non_linear3 = nn.ReLU(inplace=False)

    def multiscale_geometry(self, cloud):
        """Spatial multi-scale descriptors for every point. cloud: (B, N, 3) -> (B, N, n_scale * D)"""
        B, N, _ = cloud.shape
        ds = cloud
        feats = []
        for s in range(self.n_scale):
            idx = get_knn_idx(cloud, ds, self.k_for_knn)              # spatial neighbors in own cloud
            nbrs = knn_gather(ds, idx) - cloud.unsqueeze(2)           # (B, N, k, 3), centered
            f = self.local_transformers[s](nbrs.reshape(B * N, self.k_for_knn, 3))
            feats.append(f.view(B, N, self.local_feature_dim))

            if s < self.n_scale - 1:                                  # coarser neighbor source
                n_next = max(ds.shape[1] // self.ds_factor, self.k_for_knn)
                ds, _ = sample_farthest_points(ds, K=n_next)
        return torch.cat(feats, dim=-1)

    def forward(self, pc, pc_fts, x, x_fts, mask=None):
        """
        Forward pass.
        :param pc: (Tensor) Input point cloud tensor with shape (n_clouds, seq_len, pts_dim)
        :param pc_fts: (Tensor) Point cloud features used for kNN search.
        :param x: (Tensor) Input query points tensor with shape (n_clouds, n_sample, x_dim)
        :param x_fts: (Tensor) Query point features used for kNN search.
        :param mask: (Tensor) Mask tensor with shape (batch_size, seq_len, seq_len). Optional.
        :return: (Tensor) Output vector field values for each query point in x.
        Has shape (n_clouds, n_sample, output_dim)
        """
        n_clouds, full_seq_len = pc.shape[0], pc.shape[1]
        n_sample = x.shape[1]

        geo_x = self.multiscale_geometry(x)
        geo_pc = self.multiscale_geometry(pc)
        corr_idx = get_knn_idx(x_fts, pc_fts, self.k_for_knn)       
        matched_fts = knn_gather(pc_fts, corr_idx)                    
        sim = F.cosine_similarity(x_fts.unsqueeze(2), matched_fts, dim=-1)

        w = torch.softmax(sim / self.tau, dim=-1).unsqueeze(-1)       
        matched_geo = (w * knn_gather(geo_pc, corr_idx)).sum(dim=2) 

        seg_in = torch.cat([geo_x, matched_geo, geo_x * matched_geo, sim], dim=-1)
        return self.seg_head(seg_in) 


def SconeOccFtsLcl(n_fts=None, **kwargs):
    return SconeOcc(pts_dim=3,
                    pts_embedding_dim=64,
                    concatenate_input=True,
                    n_code=1,
                    n_heads=2,
                    FF=True,
                    gelu=True,
                    n_scale=4,
                    local_feature_dim=128,
                    k_for_knn=8,
                    x_dim=3,
                    output_dim=1,
                    dropout=None)