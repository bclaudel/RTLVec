import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import torch
from pyg_compat import Batch, Data

from checkpoint import load_encoder
from dataset import load_datasets, validate_graph, json_hash
from models import CdfgNN
from prepare_data import validate_manifest, convert_manifest, assemble_dataset, assign_families
from rtl_graph import NODE_TYPES, GRAPH_SPEC, TYPE_IDS, preprocessing_contract, positional_encoding, typed_graph, graphs_equal
from train import generate_negative_edges, main
from utils import stratified_mask, ClassBalancedFocalLoss, SizeAwareBatchSampler

ROOT = Path(__file__).resolve().parents[1]


def graph():
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    g = Data(x=torch.cat([torch.nn.functional.one_hot(labels, len(NODE_TYPES)).float(), torch.ones(6, 1)], 1),
             y=labels[:, None], edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
             edge_attr=torch.zeros(5, 1, dtype=torch.long), pos_enc=torch.randn(6, 16))
    g.edge_ports = [f'A:{i}' for i in range(5)]
    g.node_parameters = [{} for _ in range(6)]
    g.graph_schema = GRAPH_SPEC['name']
    return g


class GraphValidationTests(unittest.TestCase):
    def test_valid_and_invalid_features(self):
        g = graph()
        validate_graph(g)
        g.x[0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'one-hot'):
            validate_graph(g)

    def test_invalid_widths(self):
        for width in (-1, 0, 1.5):
            g = graph()
            g.x[0, -1] = width
            with self.assertRaisesRegex(ValueError, 'widths'):
                validate_graph(g)

    def test_edges_positions_and_finiteness(self):
        for mutate, message in (
            (lambda g: setattr(g, 'pos_enc', torch.zeros(6, 15)), 'pos_enc'),
            (lambda g: g.x.__setitem__((0, 0), float('nan')), 'finite'),
            (lambda g: g.edge_index.__setitem__((0, 0), 99), 'endpoints'),
        ):
            g = graph(); mutate(g)
            with self.assertRaisesRegex(ValueError, message):
                validate_graph(g)

    def test_legacy_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, 'legacy'):
            load_datasets(ROOT / 'data')

    def test_spectral_positions_on_sparse_and_disconnected_graphs(self):
        for n in (6, 300):
            edges = torch.tensor([(i, i + 1) for i in range(n - 1) if i != n // 2]).t()
            first = positional_encoding(n, edges.numpy())
            second = positional_encoding(n, edges.numpy())
            self.assertEqual(tuple(first.shape), (n, 16))
            self.assertTrue(torch.isfinite(first).all())
            torch.testing.assert_close(first, second)
            if n == 6:
                self.assertTrue((first[:, 4:] == 0).all())

    def test_pin_order_and_parameters_affect_deduplication(self):
        left, right = graph(), graph()
        self.assertTrue(graphs_equal(left, right))
        right.edge_ports[0] = 'B:0'
        self.assertFalse(graphs_equal(left, right))
        right = graph()
        right.node_parameters[0] = {'A_SIGNED': '1'}
        self.assertFalse(graphs_equal(left, right))


class ManifestTests(unittest.TestCase):
    def test_example_and_partial_failure(self):
        self.assertEqual(validate_manifest(ROOT / 'examples/projects.json')['accepted'], 4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.v').write_text('module a; endmodule\n')
            valid = dict(design_id='a', project_family='family', sources=['a.v'], top='a', source_provenance='fixture')
            invalid = dict(valid, design_id='b', sources=['missing.v'])
            path = root / 'projects.json'
            path.write_text(json.dumps(dict(schema_version=1, designs=[valid, invalid])))
            report = validate_manifest(path)
            self.assertEqual((report['accepted'], report['rejected']), (1, 1))
            key = report['designs'][0]['source_configuration_hash']
            self.assertEqual(key, validate_manifest(path)['designs'][0]['source_configuration_hash'])
            (root / 'a.v').write_text('module a; wire changed; endmodule\n')
            self.assertNotEqual(key, validate_manifest(path)['designs'][0]['source_configuration_hash'])

    def test_adjacent_header_content_affects_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.v').write_text('module a; endmodule')
            header = root / 'defs.vh'
            header.write_text('`define WIDTH 8')
            design = dict(design_id='a', project_family='a', sources=['a.v'], top='a', source_provenance='test')
            path = root / 'projects.json'
            path.write_text(json.dumps(dict(schema_version=1, designs=[design])))
            key = validate_manifest(path)['designs'][0]['source_configuration_hash']
            header.write_text('`define WIDTH 16')
            self.assertNotEqual(key, validate_manifest(path)['designs'][0]['source_configuration_hash'])

    def test_assignments_persist_when_families_are_added(self):
        initial = assign_families(['a', 'b', 'c'])
        grown = assign_families(['a', 'b', 'c', 'd'], initial)
        self.assertEqual({key: grown[key] for key in initial}, initial)
        self.assertEqual(set(initial.values()), {'train', 'val'})
        with self.assertRaisesRegex(ValueError, 'two'):
            assign_families(['a'])


class SamplingTests(unittest.TestCase):
    def test_complete_graph_rejected_and_empty_requests_supported(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        with self.assertRaisesRegex(ValueError, 'No negative'):
            generate_negative_edges(edges, torch.zeros(2, dtype=torch.long), edges)
        neg, _ = generate_negative_edges(edges[:, :0], torch.zeros(2, dtype=torch.long), edges)
        self.assertEqual(tuple(neg.shape), (2, 0))

    def test_dense_graph_fallback_terminates(self):
        edges = torch.tensor([(u, v) for u in range(5) for v in range(5) if u != v and (u, v) != (4, 3)]).t()
        with patch('train.random.sample', return_value=[0, 1]):
            neg, pos = generate_negative_edges(edges, torch.zeros(5, dtype=torch.long), edges)
        self.assertEqual(neg.shape, pos.shape)
        self.assertTrue(torch.equal(neg, torch.tensor([[4], [3]]).expand_as(neg)))

    def test_negative_edges_stay_inside_graphs(self):
        batch = Batch.from_data_list([graph(), graph()])
        neg, _ = generate_negative_edges(batch.edge_index, batch.batch, batch.edge_index)
        self.assertTrue(torch.equal(batch.batch[neg[0]], batch.batch[neg[1]]))
        adjacency = set(map(tuple, batch.edge_index.t().tolist()))
        self.assertFalse(any(tuple(pair) in adjacency for pair in neg.t().tolist()))

    def test_stratified_mask_guarantees_minimum(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 2])
        for seed in range(30):
            torch.manual_seed(seed)
            mask = stratified_mask(labels, mask_ratio=0.01)
            for cls in labels.unique():
                self.assertGreaterEqual(mask[labels == cls].sum().item(), min(2, (labels == cls).sum().item()))

    def test_missing_classes_do_not_dominate_focal_loss(self):
        loss = ClassBalancedFocalLoss(torch.tensor([10., 0., 20.]))
        self.assertEqual(loss.weights[1].item(), 0)
        self.assertTrue(torch.isfinite(loss(torch.randn(2, 3), torch.tensor([0, 2]))))

    def test_size_aware_batches_respect_node_and_graph_limits(self):
        sampler = SizeAwareBatchSampler([graph()] * 5, 18, shuffle=False, max_graphs=2)
        self.assertEqual(list(sampler), [[0, 1], [2, 3], [4]])


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        designs = json.loads((ROOT / 'examples/projects.json').read_text())['designs']
        for design in designs:
            design['sources'] = [str(ROOT / 'examples' / name) for name in design['sources']]
        duplicate = copy.deepcopy(designs[1])
        duplicate.update(design_id='zzz-copy-of-adder', project_family='copy-of-adder')
        designs.append(duplicate)
        cls.manifest = cls.root / 'projects.json'
        cls.manifest.write_text(json.dumps(dict(schema_version=1, designs=designs)))
        cls.generated = cls.root / 'generated'
        cls.report = convert_manifest(cls.manifest, cls.generated)
        if cls.report['rejected']:
            raise AssertionError(cls.report)
        cls.expanded = cls.root / 'expanded'
        cls.dataset = assemble_dataset(cls.generated, cls.expanded)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def converted_graph(self, design_id):
        record = next(r for r in self.report['designs'] if r['design_id'] == design_id)
        return torch.load(self.generated / record['graph_file'], weights_only=False)

    def test_combinational_and_parameterized_semantics(self):
        g = self.converted_graph('example-adder-v1')
        types = g.y[:, 0].tolist()
        self.assertEqual(types.count(TYPE_IDS['$add']), 1)
        self.assertEqual(types.count(TYPE_IDS['input']), 2)
        self.assertEqual(types.count(TYPE_IDS['output']), 1)
        index = types.index(TYPE_IDS['$add'])
        self.assertEqual(g.x[index, -1].item(), 8)
        pins = [g.edge_ports[i] for i, (_, v) in enumerate(g.edge_index.t().tolist()) if v == index]
        self.assertIn('A:0', pins)
        self.assertIn('B:7', pins)
        self.assertTrue(nx.is_directed_acyclic_graph(typed_graph(g)))

    def test_sequential_feedback_and_control_edges(self):
        counter = self.converted_graph('example-counter-v1')
        self.assertIn(TYPE_IDS['$dff'], counter.y[:, 0])
        self.assertIn(TYPE_IDS['$mux'], counter.y[:, 0])
        self.assertFalse(nx.is_directed_acyclic_graph(typed_graph(counter)))
        self.assertIn(2, counter.edge_attr[:, 0])
        self.assertIn(4, counter.edge_attr[:, 0])
        select = self.converted_graph('example-select-v1')
        self.assertIn(1, select.edge_attr[:, 0])

    def test_multifile_hierarchy_is_flattened(self):
        g = self.converted_graph('example-accumulator-v1')
        self.assertIn(TYPE_IDS['$add'], g.y[:, 0])
        self.assertIn(TYPE_IDS['$dff'], g.y[:, 0])
        validate_graph(g)

    def test_non_utf8_comments_preserve_source_bytes(self):
        root = self.root / 'legacy-encoding'; root.mkdir()
        source = root / 'a.v'
        content = b'// legacy comment: \xbf\nmodule a(input x, output y); assign y=~x; endmodule\n'
        source.write_bytes(content)
        path = root / 'projects.json'
        path.write_text(json.dumps(dict(schema_version=1, designs=[dict(design_id='a',
            project_family='a', source_provenance='test', sources=['a.v'], top='a')])))
        report = convert_manifest(path, root / 'generated')
        self.assertEqual(report['accepted'], 1)
        self.assertEqual(source.read_bytes(), content)

    def test_parallel_conversion_and_explicit_rejection_filtering(self):
        report = convert_manifest(self.manifest, self.root / 'parallel', jobs=2)
        self.assertEqual(report['accepted'], self.report['accepted'])
        for left, right in zip(report['designs'], self.report['designs']):
            self.assertEqual(left['design_id'], right['design_id'])
            self.assertEqual(left['typed_graph_hash'], right['typed_graph_hash'])
        report['designs'].append(dict(design_id='broken', status='rejected', error='syntax error'))
        report['rejected'] = 1
        (self.root / 'parallel/conversion.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'rejected designs'):
            assemble_dataset(self.root / 'parallel', self.root / 'filtered')
        metadata = assemble_dataset(self.root / 'parallel', self.root / 'filtered', allow_rejected=True)
        self.assertEqual(metadata['conversion_summary']['rejected'], 1)
        self.assertEqual(metadata['conversion_summary']['accepted'], 4)
        self.assertFalse(any(d['design_id'] == 'broken' for d in metadata['designs']))
        load_datasets(self.root / 'filtered')

    def test_reference_split_can_be_reused_without_copying_graphs(self):
        destination = self.root / 'reference-splits'
        metadata = assemble_dataset(self.generated, destination, split_from=self.expanded)
        self.assertEqual(metadata['family_splits'], self.dataset['family_splits'])
        self.assertEqual({d['design_id']: d['split'] for d in metadata['designs']},
                         {d['design_id']: d['split'] for d in self.dataset['designs']})
        load_datasets(destination)

    def test_headers_defines_parameters_and_cache_invalidation(self):
        root = self.root / 'headers with spaces'; root.mkdir()
        (root / 'inc').mkdir()
        (root / 'inc/params.vh').write_text('`define BIAS 1\n')
        (root / 'a.v').write_text('`include "params.vh"\nmodule a #(parameter WIDTH=4) '
                                 '(input [WIDTH-1:0] value, output [WIDTH-1:0] result); '
                                 'assign result = value + `BIAS + `EXTRA; endmodule\n')
        design = dict(design_id='a', project_family='a', source_provenance='test', sources=['a.v'],
                      top='a', include_dirs=['inc'], defines={'EXTRA': 2}, parameters={'WIDTH': 12})
        path = root / 'projects.json'
        path.write_text(json.dumps(dict(schema_version=1, designs=[design])))
        report = convert_manifest(path, root / 'generated')
        self.assertEqual(report['rejected'], 0)
        g = torch.load(root / 'generated' / report['designs'][0]['graph_file'], weights_only=False)
        port = next(i for i, kind in enumerate(g.y[:, 0]) if kind == TYPE_IDS['input'])
        self.assertEqual(g.x[port, -1].item(), 12)
        old_key = report['designs'][0]['graph_file']
        (root / 'inc/params.vh').write_text('`define BIAS 3\n')
        design['parameters']['WIDTH'] = 64
        path.write_text(json.dumps(dict(schema_version=1, designs=[design])))
        report = convert_manifest(path, root / 'generated')
        self.assertEqual(report['rejected'], 0)
        self.assertFalse(report['designs'][0]['cached'])
        self.assertNotEqual(old_key, report['designs'][0]['graph_file'])
        g = torch.load(root / 'generated' / report['designs'][0]['graph_file'], weights_only=False)
        port = next(i for i, kind in enumerate(g.y[:, 0]) if kind == TYPE_IDS['input'])
        self.assertEqual(g.x[port, -1].item(), 64)

    def test_syntax_inout_memory_and_node_limit_rejections(self):
        root = self.root / 'unsupported'; root.mkdir()
        cases = {
            'syntax': 'module syntax(',
            'inout': 'module inout_case(inout wire value); endmodule',
            'memory': 'module memory(input clk, input [1:0] addr, input [7:0] d, output [7:0] q); '
                      'reg [7:0] mem[0:3]; always @(posedge clk) mem[addr] <= d; assign q=mem[addr]; endmodule',
        }
        designs = []
        for name, code in cases.items():
            (root / f'{name}.v').write_text(code)
            designs.append(dict(design_id=name, project_family=name, sources=[f'{name}.v'],
                                top='inout_case' if name == 'inout' else name, source_provenance='test'))
        path = root / 'projects.json'
        path.write_text(json.dumps(dict(schema_version=1, designs=designs)))
        report = convert_manifest(path, root / 'generated')
        self.assertEqual(report['rejected'], 3)
        self.assertIn('Inout', report['designs'][1]['error'])
        self.assertIn('Memories', report['designs'][2]['error'])
        report = convert_manifest(self.manifest, root / 'limited', max_nodes=2)
        self.assertEqual(report['accepted'], 0)
        self.assertTrue(all('max_nodes' in r['error'] for r in report['designs']))

    def test_cache_and_repeat_assembly_are_stable(self):
        report = convert_manifest(self.manifest, self.generated)
        self.assertTrue(all(r['cached'] for r in report['designs']))
        dataset = assemble_dataset(self.generated, self.expanded)
        self.assertEqual(dataset['family_splits'], self.dataset['family_splits'])
        self.assertEqual(dataset['dataset_fingerprint'], self.dataset['dataset_fingerprint'])

    def test_deduplication_and_provenance(self):
        self.assertEqual(len(self.dataset['duplicates']), 1)
        self.assertEqual(self.dataset['duplicates'][0]['duplicate_of'], 'example-adder-v1')
        train, val, metadata = load_datasets(self.expanded)
        self.assertEqual(len(train) + len(val), 4)
        train_families = {entry['project_family'] for entry in metadata['designs'] if entry['split'] == 'train'}
        val_families = {entry['project_family'] for entry in metadata['designs'] if entry['split'] == 'val'}
        self.assertFalse(train_families & val_families)

    def test_incremental_assembly_preserves_existing_duplicate_representative(self):
        root = self.root / 'incremental'; root.mkdir()
        output = root / 'expanded'
        initial = assemble_dataset(self.generated, output)
        source = json.loads(self.manifest.read_text())
        added = copy.deepcopy(source['designs'][1])
        added.update(design_id='aaa-copy', project_family='aaa-copy')
        source['designs'].append(added)
        path = root / 'projects.json'; path.write_text(json.dumps(source))
        convert_manifest(path, root / 'generated')
        grown = assemble_dataset(root / 'generated', output)
        self.assertEqual(grown['family_splits'], initial['family_splits'])
        self.assertEqual([r['design_id'] for r in grown['designs']], [r['design_id'] for r in initial['designs']])
        self.assertIn({'design_id': 'aaa-copy', 'duplicate_of': 'example-adder-v1'}, grown['duplicates'])

    def test_failures_are_reported_and_assembly_refuses_them(self):
        root = self.root / 'bad'; root.mkdir()
        (root / 'bad.v').write_text('module bad(input a, output b); missing u(a, b); endmodule')
        design = dict(design_id='bad', project_family='bad', top='bad', sources=['bad.v'], source_provenance='test')
        path = root / 'projects.json'
        path.write_text(json.dumps(dict(schema_version=1, designs=[design])))
        report = convert_manifest(path, root / 'generated')
        self.assertEqual(report['rejected'], 1)
        self.assertTrue((root / 'generated' / report['designs'][0]['log_file']).exists())
        with self.assertRaisesRegex(ValueError, 'rejected'):
            assemble_dataset(root / 'generated', root / 'expanded')
        self.assertFalse((root / 'expanded').exists())
        with patch('prepare_data.run_yosys', side_effect=subprocess.TimeoutExpired('yosys', 1)):
            report = convert_manifest(path, root / 'timeout')
        self.assertIn('timeout', report['designs'][0]['error'])

    def test_training_and_checkpoint_reload(self):
        self.check_training_and_checkpoint_reload('dense')

    def test_graph_training_and_checkpoint_reload(self):
        self.check_training_and_checkpoint_reload('graph')

    def check_training_and_checkpoint_reload(self, layout):
        run_root = self.root / 'runs' / layout
        argv = ['train.py', '--data-dir', str(self.expanded), '--num_epochs', '1', '--batch_size', '2',
                '--output-dir', str(run_root), '--gnn_type', 'gin', '--gnn_hidden_dim', '8',
                '--gnn_output_dim', '8', '--num_gnn_layers', '2', '--transformer_hidden_dim', '8',
                '--transformer_feedforward_dim', '16', '--transformer_num_heads', '2',
                '--num_transformer_layers', '1', '--stratified_masking', '--attention-layout', layout]
        with patch.object(sys, 'argv', argv), patch('train.torch.cuda.is_available', return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        run = next(run_root.iterdir())
        for name in ('args.json', 'dataset.json', 'metrics.jsonl', 'last.pt', 'best.pt'):
            self.assertTrue((run / name).is_file())
        checkpoint = torch.load(run / 'best.pt', weights_only=True)
        self.assertIn('optimizer_state_dict', checkpoint)
        self.assertIn('scheduler_state_dict', checkpoint)
        self.assertEqual(checkpoint['epoch'], 1)
        encoder = load_encoder(run / 'best.pt', preprocessing_fingerprint=self.dataset['preprocessing_fingerprint'])
        reference = CdfgNN(**checkpoint['model_config'])
        reference.load_state_dict(checkpoint['model_state_dict']); reference.eval()
        batch = Batch.from_data_list([self.converted_graph('example-counter-v1')])
        with torch.no_grad():
            embeddings = encoder.encode_nodes(batch.x, batch.edge_index, batch.batch, batch.pos_enc)
            expected = reference.encode_nodes(batch.x, batch.edge_index, batch.batch, batch.pos_enc)
            torch.testing.assert_close(embeddings, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            load_encoder(run / 'best.pt', preprocessing_fingerprint='wrong')


if __name__ == '__main__':
    unittest.main()
