import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Sampler


def set_seed(seed: int = 42):
    """
    Set random seed for Python, NumPy, and PyTorch (CPU and CUDA) for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDA operations (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Seed set to {seed}")


def check_label_distribution(dataset, node_type_to_idx):
    ys_dist = {}
    total = 0

    for data in tqdm(dataset):
        for y in data.y:
            y_int = int(y[0].item())
            for k, v in node_type_to_idx.items():
                if y_int == v:
                    y_str = k
                    break
            if y_str not in ys_dist:
                ys_dist[y_str] = 0
            ys_dist[y_str] += 1
            total += 1
    sorted_ys_dist = dict(sorted(ys_dist.items(), key=lambda item: item[1], reverse=True))
    for label, count in sorted_ys_dist.items():
        print(f"{label}: {count} ({round((count / total) * 100, 2)}%)")
    # ys_percent = {k: round((v / total) * 100, 2) for k, v in ys_dist.items()}
    # sorted_ys_percent = dict(sorted(ys_percent.items(), key=lambda item: item[1], reverse=True))
    # for label, percent in sorted_ys_percent.items():
    #     print(f"{label}: {percent:.2f}%")

    print(f"Total unique labels: {len(ys_dist)}")


def compute_class_weights(dataset, num_classes=32):
    ys_dist = {}
    total = 0

    for data in dataset:
        for y in data.y:
            y_int = int(y[0].item())
            if y_int not in ys_dist:
                ys_dist[y_int] = 0
            ys_dist[y_int] += 1
            total += 1
    
    class_weights = [total / ys_dist[i] for i in range(num_classes)]
    return torch.tensor(class_weights, dtype=torch.float32)

    
def split_dataset(dataset, val_split=0.2):
    # split the dataset into training and validation sets
    num_samples = len(dataset)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


class SizeAwareBatchSampler(Sampler):
    def __init__(self, dataset, max_nodes_per_batch, shuffle=True):
        self.dataset = dataset
        self.max_nodes_per_batch = max_nodes_per_batch
        self.shuffle = shuffle
        self.graph_sizes = [data.x.size(0) for data in dataset]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch, total_nodes = [], 0
        for idx in indices:
            size = self.graph_sizes[idx]
            if total_nodes + size > self.max_nodes_per_batch and batch:
                yield batch
                batch, total_nodes = [idx], size # hit the node limit, reset batch and total_nodes
            else:
                batch.append(idx)
                total_nodes += size

        if batch:
            yield batch

    def __len__(self):
        # Rough estimate — actual batch count depends on graph sizes
        return (sum(self.graph_sizes) + self.max_nodes_per_batch - 1) // self.max_nodes_per_batch


def get_dynamic_mask_ratio(epoch, total_epochs, min_ratio=0.1, max_ratio=0.4):
    """
    Linearly increase mask ratio from min_ratio to max_ratio over training epochs.
    """
    progress = epoch / total_epochs
    return min_ratio + (max_ratio - min_ratio) * progress


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super().__init__()
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        self.weights = weights / weights.sum() * len(samples_per_class)  # Normalize
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=None)  # Do weighting manually
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma
        class_weight = self.weights.to(logits.device)[targets]
        loss = class_weight * focal * ce
        return loss.mean()
    

def compute_class_counts(dataset, num_classes):
    counts = torch.zeros(num_classes)
    for data in dataset:
        y = data.y[:, 0]
        for c in range(num_classes):
            counts[c] += (y == c).sum()
    return counts


def stratified_mask(labels, mask_ratio=0.2, min_per_class=2):
    """
    Stratified masking to ensure at least `min_per_class` nodes from each class are masked.
    labels: [num_nodes] (tensor of class indices)
    Returns: mask of shape [num_nodes] (bool)
    """
    num_nodes = labels.size(0)
    device = labels.device
    mask = torch.rand(num_nodes, device=device) < mask_ratio

    for cls in labels.unique():
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        if len(cls_idx) == 0:
            continue
        num_masked = mask[cls_idx].sum().item()
        if num_masked < min_per_class:
            extra = cls_idx[torch.randperm(len(cls_idx))[:min(min_per_class, len(cls_idx))]]
            mask[extra] = True

    return mask


def init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)