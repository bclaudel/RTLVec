"""Measure actual stage-1 steps across attention layouts and precision modes."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from dataset import load_datasets
from models import CdfgNN
from pyg_compat import Batch
from train import train_joint_epoch, validate_mnm, validate_ep
from utils import ClassBalancedFocalLoss, SizeAwareBatchSampler, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/deepcircuitx/stage1-full')
    parser.add_argument('--output', default='data/deepcircuitx/performance/benchmark.json')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    options = parser.parse_args()
    if options.repeats < 1 or options.warmup < 0:
        parser.error('repeats must be positive and warmup nonnegative')
    if not torch.cuda.is_available():
        parser.error('CUDA is required for this benchmark')
    torch.set_num_threads(1)
    start = time.perf_counter()
    train, _, metadata = load_datasets(options.data_dir)
    load_seconds = time.perf_counter() - start
    torch.set_num_threads(4)
    mixed = next(iter(SizeAwareBatchSampler(train, 4096, shuffle=False, max_graphs=16)))
    ordered = sorted(range(len(train)), key=lambda i: train[i].num_nodes)
    median = train[ordered[len(ordered) // 2]].num_nodes
    typical = sorted(ordered, key=lambda i: abs(train[i].num_nodes - median))[:16]
    cases = {'mixed': mixed, 'typical': typical, 'single_max': [ordered[-1]]}
    selected = {name: [train[index] for index in indices] for name, indices in cases.items()}
    counts = torch.tensor(metadata['statistics']['train']['class_counts'])
    del train
    gc.collect()
    config = dict(gnn_type='gin', gnn_input_dim=97, gnn_hidden_dim=256,
        gnn_output_dim=256, num_gnn_layers=8, gnn_num_heads=4, jk_mode='last',
        gnn_dropout=.2, transformer_hidden_dim=512, transformer_feedforward_dim=2048,
        transformer_num_heads=4, num_transformer_layers=8, transformer_dropout=.2,
        num_classes=65, pos_enc_dim=16, delete_node_width_embedding=False,
        edge_predictor_input_dim=1024)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'dataset_fingerprint': metadata['dataset_fingerprint'],
        'load_seconds': load_seconds, 'repeats': options.repeats,
        'warmup': options.warmup, 'model_config': config,
        'timing': 'synchronized wall time; component CUDA events include host launch gaps',
        'results': []}
    for name, graphs in selected.items():
        sizes = [graph.num_nodes for graph in graphs]
        for layout in ('dense', 'graph'):
            for precision in ('fp32', 'bf16'):
                if precision == 'bf16' and not torch.cuda.is_bf16_supported():
                    continue
                set_seed(42)
                model = CdfgNN(**config, attention_layout=layout).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
                batch = Batch.from_data_list(graphs).to('cuda')
                args = SimpleNamespace(precision=precision, mask_ratio=.2,
                    stratified_masking=True, random_flip_posenc=True)
                mnm_loss = ClassBalancedFocalLoss(counts.cuda())
                ep_loss = torch.nn.CrossEntropyLoss()
                events = {}
                handles = []
                def start_hook(key):
                    def hook(*unused):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        events.setdefault(key, []).append((begin, end))
                    return hook
                def end_hook(key):
                    def hook(*unused):
                        events[key][-1][1].record()
                    return hook
                for component, module in (('gnn', model.gnn), ('transformer', model.transformer)):
                    handles += [module.register_forward_pre_hook(start_hook(component + '_forward')),
                        module.register_forward_hook(end_hook(component + '_forward')),
                        module.register_full_backward_pre_hook(start_hook(component + '_backward')),
                        module.register_full_backward_hook(end_hook(component + '_backward'))]
                def step():
                    return train_joint_epoch(args, model, [batch], optimizer, mnm_loss, ep_loss, torch.device('cuda'))
                for _ in range(options.warmup):
                    step()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                milliseconds, components = [], {}
                for _ in range(options.repeats):
                    events.clear()
                    start = time.perf_counter()
                    metrics = step()
                    torch.cuda.synchronize()
                    milliseconds.append((time.perf_counter() - start) * 1000)
                    for key, pairs in events.items():
                        components.setdefault(key, []).append(sum(begin.elapsed_time(end) for begin, end in pairs))
                peak = torch.cuda.max_memory_allocated() / 2**30
                # Exercise validation autocast as well as training on each mode.
                validation = {'mnm': validate_mnm(args, model, [batch], mnm_loss, torch.device('cuda')),
                              'ep': validate_ep(args, model, [batch], ep_loss, torch.device('cuda'))}
                result = {'case': name, 'graph_sizes': sizes, 'graphs': len(sizes),
                    'nodes': sum(sizes), 'dense_pairs': sum(sizes)**2,
                    'within_graph_pairs': sum(n*n for n in sizes),
                    'attention_layout': layout, 'precision': precision,
                    'step_ms_median': statistics.median(milliseconds),
                    'step_ms_samples': milliseconds, 'peak_allocated_GiB': peak,
                    'component_ms_median': {key: statistics.median(values) for key, values in components.items()},
                    'training_metrics': metrics, 'validation_metrics': validation}
                report['results'].append(result)
                print(json.dumps(result), flush=True)
                for handle in handles:
                    handle.remove()
                del model, optimizer, batch, mnm_loss, ep_loss
                gc.collect()
                torch.cuda.empty_cache()
    destination = Path(options.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Benchmark report: {destination}', flush=True)


if __name__ == '__main__':
    main()
