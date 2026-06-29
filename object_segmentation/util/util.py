import numpy as np
import torch
import torch.nn.functional as F
import yaml
from types import SimpleNamespace
import argparse

def parse_args():
    # Step 1: get the config file path (and any CLI overrides)
    parser = argparse.ArgumentParser(description='ARKitScenes Scene Segmentation')
    parser.add_argument('--config', type=str, default='cfg/config.yaml',
                        help='Path to YAML config file')
    
    # Optional: allow any key to be overridden from the CLI
    args, overrides = parser.parse_known_args()

    # Step 2: load the YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Step 3: apply CLI overrides (e.g. --lr 0.001 --epochs 100)
    override_parser = argparse.ArgumentParser()
    for key, val in config.items():
        override_parser.add_argument(f'--{key}', type=type(val) if val is not None else str)
    override_args = override_parser.parse_args(overrides)

    for key, val in vars(override_args).items():
        if val is not None:
            config[key] = val

    config['config'] = args.config  # preserve the config path
    return SimpleNamespace(**config)

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1) # gold is the groudtruth label in the dataloader

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)  # the number of feature_dim of the ouput, which is output channels

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


# create a file and write the text into it:
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y


def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred = pred.max(dim=2)[1]    # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):   # sample_idx
        part_ious = []
        for part in range(num_classes):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))

            F = np.sum(target_np[shape_idx] == part)

            if F != 0:
                iou = I / float(U)    #  iou across all points for this class
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # each time append an average iou across all classes of this sample (sample_level!)
    return shape_ious   # [batch_size]

def compute_class_weights(labels, n_classes, device):
    class_counts = np.zeros(n_classes)
    for cls in range(n_classes):
        class_counts[cls] += np.sum(labels == cls)
    
    # inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * n_classes  # normalize
    return torch.tensor(weights, dtype=torch.float32).to(device)
