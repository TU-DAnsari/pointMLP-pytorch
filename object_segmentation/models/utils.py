import torch
from pytorch3d.ops import knn_gather

def get_knn_points(X, pc, k):
    """
    Returns the k nearest neighbours of X in pc.
    :param X: Tensor with shape (n_clouds, n_sample, 3)
    :param pc: Tensor with shape (n_clouds, seq_len, 3)
    :param k: integer
    :return: returns a Tensor with shape (n_clouds, n_sample, k, 3)
    """
    dists = torch.cdist(X, pc)

    min_dists, argmin_dists = torch.topk(dists, k=k, dim=-1, largest=False)

    return knn_gather(pc, argmin_dists), min_dists, argmin_dists