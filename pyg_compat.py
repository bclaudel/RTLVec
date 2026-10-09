"""Centralize PyG imports and its upstream TorchScript warning workaround.

PyG eagerly scripts SelectOutput/ConnectOutput during import, even for models
that use neither pooling class. Limit the warning filter to that import: later
TorchScript calls and unrelated warnings retain their normal behavior.
Upstream: https://github.com/pyg-team/pytorch_geometric/issues/10697
"""
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=(r"^`torch\.jit\.script` is not supported in Python 3\.14\+ "
                 r"and may break\. Please switch to `torch\.compile` or "
                 r"`torch\.export`\.$"),
        category=FutureWarning,
        module=r"^torch\.jit\._script$",
    )
    import torch_geometric

from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, GraphNorm, JumpingKnowledge
