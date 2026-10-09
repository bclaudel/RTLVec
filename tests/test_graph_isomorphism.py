import random
import unittest

import networkx as nx

from graph_isomorphism import attributed_graphs_equal


class ExactMatchingTests(unittest.TestCase):
    def test_matches_original_algorithm_and_permutations(self):
        rng = random.Random(41)
        for _ in range(100):
            graphs = []
            for seed in (rng.randrange(100000), rng.randrange(100000)):
                g = nx.gnp_random_graph(8, .18, seed=seed, directed=True)
                nx.set_node_attributes(g, {u: str(rng.randrange(2)) for u in g}, 'label')
                nx.set_edge_attributes(g, {edge: str(rng.randrange(2)) for edge in g.edges}, 'pins')
                graphs.append(g)
            left, right = graphs
            expected = nx.is_isomorphic(left, right,
                node_match=nx.algorithms.isomorphism.categorical_node_match('label', None),
                edge_match=nx.algorithms.isomorphism.categorical_edge_match('pins', None))
            self.assertEqual(attributed_graphs_equal(left, right), expected)
            order = list(left)
            rng.shuffle(order)
            self.assertTrue(attributed_graphs_equal(left, nx.relabel_nodes(left, dict(zip(left, order)))))

    def test_wl_collision_and_symmetric_components(self):
        left = nx.cycle_graph(8, create_using=nx.DiGraph)
        right = nx.disjoint_union(nx.cycle_graph(4, create_using=nx.DiGraph), nx.cycle_graph(4, create_using=nx.DiGraph))
        for g in (left, right):
            nx.set_node_attributes(g, 'signal', 'label')
            nx.set_edge_attributes(g, 'A:0', 'pins')
        self.assertEqual(nx.weisfeiler_lehman_graph_hash(left, node_attr='label', edge_attr='pins'),
                         nx.weisfeiler_lehman_graph_hash(right, node_attr='label', edge_attr='pins'))
        self.assertFalse(attributed_graphs_equal(left, right))
        symmetric = nx.disjoint_union_all([left] * 64)
        renamed = nx.relabel_nodes(symmetric, {u: len(symmetric) - 1 - u for u in symmetric})
        self.assertTrue(attributed_graphs_equal(symmetric, renamed))

    def test_self_loops_and_edge_labels(self):
        left = nx.DiGraph()
        left.add_node(0, label='signal')
        left.add_edge(0, 0, pins='reset:0')
        right = left.copy()
        self.assertTrue(attributed_graphs_equal(left, right))
        right[0][0]['pins'] = 'clock:0'
        self.assertFalse(attributed_graphs_equal(left, right))

    def test_connected_regular_wl_collision_requires_exact_matching(self):
        left = nx.complete_bipartite_graph(3, 3).to_directed()
        right = nx.circular_ladder_graph(3).to_directed()
        for g in (left, right):
            nx.set_node_attributes(g, 'signal', 'label')
            nx.set_edge_attributes(g, 'A:0', 'pins')
        self.assertEqual(nx.weisfeiler_lehman_graph_hash(left, node_attr='label', edge_attr='pins'),
                         nx.weisfeiler_lehman_graph_hash(right, node_attr='label', edge_attr='pins'))
        self.assertFalse(attributed_graphs_equal(left, right))
