import unittest

import torch

from models import TransformerModel


class GraphAttentionTests(unittest.TestCase):
    def models(self):
        torch.manual_seed(43)
        dense = TransformerModel(8, 8, 16, 2, 2, 65, 16, 0, 'dense')
        graph = TransformerModel(8, 8, 16, 2, 2, 65, 16, 0, 'graph')
        graph.load_state_dict(dense.state_dict())
        return dense, graph

    def test_outputs_and_gradients_match_dense(self):
        dense, graph = self.models()
        batch = torch.repeat_interleave(torch.arange(5), torch.tensor([2, 3, 3, 7, 8]))
        left = torch.randn(len(batch), 8, requires_grad=True)
        right = left.detach().clone().requires_grad_()
        positions = torch.randn(len(batch), 16)
        mask = torch.arange(len(batch)) % 3 == 0
        a, b = dense(left, batch, positions, mask), graph(right, batch, positions, mask)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        weights = torch.randn_like(a)
        (a * weights).sum().backward()
        (b * weights).sum().backward()
        torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=2e-5)
        for p, q in zip(dense.parameters(), graph.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)

    def test_graphs_are_independent(self):
        _, graph = self.models()
        graph.eval()
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        x = torch.randn(6, 8)
        positions = torch.randn(6, 16)
        a = graph(x, batch, positions)
        x[3:] += 100
        b = graph(x, batch, positions)
        torch.testing.assert_close(a[:3], b[:3], atol=0, rtol=0)

    def test_single_graph_matches_dense(self):
        dense, graph = self.models()
        batch = torch.zeros(9, dtype=torch.long)
        x, positions = torch.randn(9, 8), torch.randn(9, 16)
        torch.testing.assert_close(dense(x, batch, positions), graph(x, batch, positions))

    def test_bfloat16_masking_and_gradients(self):
        for layout in ('dense', 'graph'):
            with self.subTest(layout=layout):
                model = self.models()[0 if layout == 'dense' else 1]
                batch = torch.tensor([0, 0, 0, 1, 1, 1])
                x, positions = torch.randn(6, 8), torch.randn(6, 16)
                mask = torch.tensor([True, False, False, False, True, False])
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    output = model(x, batch, positions, mask)
                    loss = output.float().square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(model.mask_embedding.grad).all())

    def test_eval_without_gradients_matches_dense_with_padding(self):
        dense, graph = self.models()
        dense.eval()
        graph.eval()
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        x, positions = torch.randn(7, 8), torch.randn(7, 16)
        with torch.no_grad():
            torch.testing.assert_close(dense(x, batch, positions), graph(x, batch, positions), atol=2e-6, rtol=2e-5)
