"""Batch independent graph sequences without allocating cross-graph attention."""
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence


def encode_graph_sequences(encoder, x, batch):
    # PyG batches store each graph's nodes contiguously. Transfer only the small
    # list of group sizes, rather than an N-by-N attention mask.
    groups, counts = torch.unique_consecutive(batch, return_counts=True)
    group_ids, lengths = groups.tolist(), counts.tolist()
    if len(set(group_ids)) != len(group_ids):
        raise ValueError('Graph attention requires contiguous nodes per graph')
    chunks = x.split(lengths)
    buckets = defaultdict(list)
    for index, length in enumerate(lengths):
        buckets[1 << (length - 1).bit_length()].append(index)
    outputs = [None] * len(chunks)
    for indices in buckets.values():
        if len(indices) == 1:
            index = indices[0]
            outputs[index] = encoder(chunks[index].unsqueeze(0)).squeeze(0)
            continue
        padded = pad_sequence([chunks[index] for index in indices], batch_first=True)
        sizes = torch.tensor([lengths[index] for index in indices], device=x.device)
        padding = torch.arange(padded.shape[1], device=x.device)[None, :] >= sizes[:, None]
        encoded = encoder(padded, src_key_padding_mask=padding)
        for row, index in enumerate(indices):
            outputs[index] = encoded[row, :lengths[index]]
    return torch.cat(outputs, dim=0)
