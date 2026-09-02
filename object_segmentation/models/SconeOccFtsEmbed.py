import torch.nn.functional as F
from .Attention import *
from .utils import get_knn_points, get_knn_idx
from pytorch3d.ops import knn_gather


class XEmbedding(nn.Module):
    def __init__(self, x_dim, x_embedding_dim, dropout=None, gelu=True):
        """
        Neural module for individual 3D point embedding using fully connected layers.

        :param x_dim: (int) Dimension of input 3D points. Usually, x_dim=3.
        :param x_embedding_dim: (int) Dimension of output embeddings.
        :param dropout: Dropout module to apply on computed features.
        :param gelu: (bool) If True, the model uses GELU non-linearity. Else, it uses ReLU.
        """
        super(XEmbedding, self).__init__()

        self.linear1 = nn.Linear(x_dim, x_embedding_dim // 4)
        # self.fc1 = nn.Linear(3*2*self.n_harmonic_functions, self.x_embd_size//2)
        self.linear2 = nn.Linear(x_embedding_dim // 4, x_embedding_dim // 2)
        self.linear3 = nn.Linear(x_embedding_dim // 2, x_embedding_dim)

        if gelu:
            self.non_linear1 = nn.GELU()
            self.non_linear2 = nn.GELU()
            self.non_linear3 = nn.GELU()
        else:
            self.non_linear1 = nn.ReLU(inplace=False)
            self.non_linear2 = nn.ReLU(inplace=False)
            self.non_linear3 = nn.ReLU(inplace=False)

        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        res = self.non_linear1(self.linear1(x))
        res = self.non_linear2(self.linear2(res))
        res = self.non_linear3(self.linear3(res))

        res = res if self.dropout is None else self.dropout(res)

        return res


class PCTransformer(nn.Module):
    def __init__(self, seq_len, pts_dim=3, pts_embedding_dim=256, feature_dim=512,
                 concatenate_input=True,
                 n_code=2, n_heads=4, FF=True, gelu=True,
                 dropout=None,
                 fts_embedding_dim=0):
        """
        Main class for Transformer units dedicated to point cloud global encoding.

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
        :param fts_embedding_dim: (int) Dimension of an additional per-point feature embedding to fuse into
        the point embedding. If 0, no additional feature is fused.
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

        self.fts_embedding_dim = fts_embedding_dim

        # Point embedding
        self.embedding = Embedding(input_dim=pts_dim, output_dim=pts_embedding_dim,
                                   dropout=None, gelu=gelu,
                                   global_feature=False, additional_feature_dim=fts_embedding_dim,
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

    def forward(self, pc, fts=None, mask=None):
        """
        Forward pass.

        :param pc: (Tensor) Point cloud tensor with shape (n_clouds, seq_len, pts_dim)
        :param fts: (Tensor) Additional per-point feature embedding to fuse into the point embedding.
        Tensor with shape (n_clouds, seq_len, fts_embedding_dim). Optional.
        :param mask: (Tensor) Mask tensor with shape (batch_size, seq_len, seq_len). Optional.
        :return: (Tensor) Output features with shape (batch_size, features_dim)
        """
        n_clouds, seq_len = pc.shape[0], pc.shape[1]

        pc_embd = self.embedding(pc, additional_feature=fts)
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
    def __init__(self, seq_len=2048, pts_dim=3, pts_embedding_dim=128,
                 concatenate_input=True,
                 n_code=2, n_heads=4, FF=True, gelu=True,
                 global_feature_dim=512,
                 n_scale=3, local_feature_dim=256, k_for_knn=16,
                 x_dim=3, x_embedding_dim=512,
                 output_dim=1,
                 dropout=None,
                 offset=True,
                 fts_dim=128, fts_embedding_dim=64):
        """
        Main class for SCONE's occupancy probability prediction module.
        A neural model that predicts a vector field as an implicit function, depending on an input point cloud
        and view state harmonic features representing the history of camera poses.
        A Transformer with Multi-Scale Neighborhood features (MSN features) is used to encode the point cloud,
        depending on the query point x.

        :param seq_len: (int) Number of points in the input point cloud.
        :param pts_dim: (int) Dimension of points in the input point cloud.
        :param pts_embedding_dim: (int) Dimension of embedded point cloud.
        :param concatenate_input: (bool) If True, concatenates raw input points to the point embeddings
        in the point cloud.
        :param n_code: (int) Number of Multi-Head Self Attention units.
        :param n_heads: (int) Number of heads in Multi-Head Self Attention units.
        :param FF: (bool) If True, the Transformer encoder applies a Feed Forward unit after
        each Multi-Head Self-Attention unit.
        :param gelu: (bool) If True, the model uses GELU non-linearity. Else, it uses ReLU.
        :param global_feature_dim: (int) Dimension of the point cloud global feature.
        :param n_scale: (int) Number of scales to compute neighborhood features in the point cloud.
        :param local_feature_dim: (int) Dimension of point cloud neighborhood features.
        :param k_for_knn: (int) Number of neighbors to use when computing a neighborhood feature.
        :param x_dim: (int) Dimension of query point x.
        :param x_embedding_dim: (int) Dimension of x embedding.
        :param n_harmonics: (int) Number of harmonics used to compute view_state harmonic features.
        :param output_dim: (int) Dimension of the output vector field.
        :param dropout: Dropout module to apply on computed embeddings.
        :param offset: (bool) If True, the model uses the offset between x and its neighbors rather than
        the coordinates of the neighbors to compute neighborhood features.
        This parameter should always be True, since it leads to far better performances.
        :param fts_dim: (int) Dimension of the input pc_fts / x_fts feature vectors.
        :param fts_embedding_dim: (int) Dimension the input features are projected to before being fused
        with point coordinates in the global and local point transformers, and with the query point x.
        If 0, features are only used for kNN matching and are not fused into the transformers.
        """
        super(SconeOcc, self).__init__()

        # Parameters
        self.seq_len = seq_len
        self.pts_dim = pts_dim
        self.pts_embedding_dim = pts_embedding_dim

        self.n_code = n_code
        self.n_heads = n_heads
        self.FF = FF
        self.gelu = gelu

        self.n_scale = n_scale

        self.x_dim = x_dim
        self.x_embedding_dim = x_embedding_dim

        self.output_dim = output_dim

        self.dropout = dropout

        self.encoding_dim = pts_embedding_dim

        self.k_for_knn = k_for_knn
        self.offset = offset
        if self.offset:
            print("Offset set to True.")

        self.global_feature_dim = global_feature_dim
        self.local_feature_dim = local_feature_dim
        self.all_feature_size = self.x_embedding_dim \
                                + self.n_scale * self.local_feature_dim \
                                + self.global_feature_dim

        # Feature projection: pc_fts / x_fts are projected into a shared, lower-dimensional
        # embedding before being fused with point coordinates, so that the local kNN search
        # can keep operating on the raw (high-dimensional) features while the transformers
        # only need to absorb a small additional input.
        self.fts_dim = fts_dim
        self.fts_embedding_dim = fts_embedding_dim if fts_dim > 0 else 0
        if self.fts_embedding_dim > 0:
            self.fts_projection = nn.Linear(fts_dim, self.fts_embedding_dim)
            self.fts_nonlinear = nn.GELU() if gelu else nn.ReLU(inplace=False)

        # Point cloud transformers
        self.global_transformer = PCTransformer(seq_len=seq_len, pts_dim=pts_dim,
                                                pts_embedding_dim=pts_embedding_dim, feature_dim=global_feature_dim,
                                                concatenate_input=concatenate_input,
                                                n_code=n_code, n_heads=n_heads, FF=FF, gelu=gelu,
                                                dropout=dropout,
                                                fts_embedding_dim=self.fts_embedding_dim)

        local_transformers = []
        for i in range(n_scale):
            local_transformers += [PCTransformer(seq_len=k_for_knn, pts_dim=pts_dim,
                                                 pts_embedding_dim=pts_embedding_dim, feature_dim=local_feature_dim,
                                                 concatenate_input=concatenate_input,
                                                 n_code=n_code, n_heads=n_heads, FF=FF, gelu=gelu,
                                                 dropout=dropout,
                                                 fts_embedding_dim=self.fts_embedding_dim)]
        self.local_transformers = nn.ModuleList(local_transformers)

        # X embedding
        self.x_embedding = XEmbedding(x_dim=x_dim + self.fts_embedding_dim, x_embedding_dim=x_embedding_dim,
                                      dropout=dropout, gelu=gelu)

        # Point cloud feature extraction
        self.max_pool = nn.functional.max_pool1d  # kernel_size will be <= seq_len
        self.avg_pool = nn.functional.avg_pool1d  # kernel_size will be <= seq_len

        # MLP for occupancy probability prediction
        self.linear1 = nn.Linear(self.all_feature_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, output_dim)

        # self.non_linear1 = nn.ReLU(inplace=False)
        # self.non_linear2 = nn.ReLU(inplace=False)
        # self.non_linear3 = nn.ReLU(inplace=False)

        if gelu:
            self.non_linear1 = nn.GELU()
            self.non_linear2 = nn.GELU()
            self.non_linear3 = nn.GELU()
        else:
            self.non_linear1 = nn.ReLU(inplace=False)
            self.non_linear2 = nn.ReLU(inplace=False)
            self.non_linear3 = nn.ReLU(inplace=False)

    def forward(self, pc, pc_fts, x, x_fts, mask=None):
        """
        Forward pass.
        :param pc: (Tensor) Input point cloud tensor with shape (n_clouds, seq_len, pts_dim)
        :param x: (Tensor) Input query points tensor with shape (n_clouds, n_sample, x_dim)
        :param view_harmonics: (Tensor) View state harmonic features.
        Tensor with shape (n_clouds, n_sample, n_harmonics).
        :param mask: (Tensor) Mask tensor with shape (batch_size, seq_len, seq_len). Optional.
        :return: (Tensor) Output vector field values for each query point in x.
        Has shape (n_clouds, n_sample, output_dim)
        """
        n_clouds, full_seq_len = pc.shape[0], pc.shape[1]
        n_sample = x.shape[1]

        # -----Feature projection-----
        # Project pc_fts / x_fts into a shared embedding space, once, so it can be fused into the
        # global point cloud, every local neighborhood, and the query point x below. The kNN search
        # for local correspondences below still uses the raw, unprojected pc_fts / x_fts.
        if self.fts_embedding_dim > 0:
            pc_fts_embd = self.fts_nonlinear(self.fts_projection(pc_fts))
            x_fts_embd = self.fts_nonlinear(self.fts_projection(x_fts))
        else:
            pc_fts_embd = None
            x_fts_embd = None

        # -----Point cloud global encoding-----
        # Down sampling point cloud for global embedding
        global_perm = torch.randperm(pc.shape[1])[:self.seq_len]
        global_down_sampled_pc = pc[:, global_perm]
        global_down_sampled_fts = pc_fts_embd[:, global_perm] if pc_fts_embd is not None else None
        seq_len = global_down_sampled_pc.shape[1]

        global_features = self.global_transformer(global_down_sampled_pc, fts=global_down_sampled_fts)

        # -----Point cloud local encoding-----
        # Computing down sampling factor
        if self.n_scale > 1:
            ds_factor = int(np.power(full_seq_len / (self.k_for_knn * 8), 1./(self.n_scale - 1)))
            if ds_factor == 0:
                # print("Problem: ds_factor=0 encountered. Taking ds_factor=2 as a default value.")
                ds_factor = 2
        else:
            ds_factor = 1

        # kNN computation for local embedding
        down_sampled_pc = pc
        down_sampled_pc_fts = pc_fts
        down_sampled_pc_fts_embd = pc_fts_embd
        local_transformed = []
        for n_transformer in range(self.n_scale):
            local_transformer = self.local_transformers[n_transformer]
            # Get kNN points in down sampled pc, matching in the down sampled feature space
            local_idx = get_knn_idx(x_fts, down_sampled_pc_fts, self.k_for_knn)
            local_pc = knn_gather(down_sampled_pc, local_idx)
            if self.offset:
                local_pc = local_pc - x.view(n_clouds, n_sample, 1, 3)

            if down_sampled_pc_fts_embd is not None:
                local_fts = knn_gather(down_sampled_pc_fts_embd, local_idx)
                local_fts = local_fts.view(-1, self.k_for_knn, self.fts_embedding_dim)
            else:
                local_fts = None

            # Compute features
            local_transformed += [local_transformer(local_pc.view(-1, self.k_for_knn, 3),
                                                     fts=local_fts, mask=mask)]

            # Down sample pc, along with its raw and projected features, for the next scale
            ds_seq_len = down_sampled_pc.shape[1]
            # print("Ds seq len:", ds_seq_len)

            if n_transformer < self.n_scale-1:
                ds_idx = torch.randperm(ds_seq_len)[:ds_seq_len // ds_factor]
                down_sampled_pc = down_sampled_pc[:, ds_idx]
                down_sampled_pc_fts = down_sampled_pc_fts[:, ds_idx]
                if down_sampled_pc_fts_embd is not None:
                    down_sampled_pc_fts_embd = down_sampled_pc_fts_embd[:, ds_idx]
                # print("DS pc:", down_sampled_pc.shape)

        if self.n_scale > 0:
            local_features = torch.cat(local_transformed, dim=-1)
        else:
            local_features = torch.zeros(n_clouds, n_sample, 0, device=pc.get_device())
        local_features = local_features.view(n_clouds, n_sample, self.n_scale * self.local_feature_dim)

        # -----X encoding-----
        x_input = torch.cat((x, x_fts_embd), dim=-1) if x_fts_embd is not None else x
        x_features = self.x_embedding(x_input)

        # -----Occupancy prediction-----
        global_features = global_features.view(n_clouds, 1, self.global_feature_dim).expand(-1, n_sample, -1)
        x_features = x_features.view(n_clouds, n_sample, self.x_embedding_dim)

        res = torch.cat((global_features, local_features, x_features), dim=-1)
        res = self.non_linear1(self.linear1(res))
        res = self.non_linear2(self.linear2(res))
        res = self.linear3(res)

        return res.view(n_clouds, n_sample, self.output_dim)
    
def SconeOccSmallFtsEmbed(n_fts, **kwargs):
    return SconeOcc(seq_len=1024,
                    pts_dim=3, 
                    pts_embedding_dim=64,
                    concatenate_input=True,
                    n_code=1, 
                    n_heads=2, 
                    FF=True, 
                    gelu=True,
                    global_feature_dim=256,
                    n_scale=3, 
                    local_feature_dim=128, 
                    k_for_knn=8,
                    x_dim=3,
                    x_embedding_dim=256,
                    output_dim=1,
                    dropout=None,
                    offset=False,
                    fts_dim=n_fts,
                    fts_embedding_dim=64)