"""Release regressions for SNACK 1.x graph and molecule conversions."""
import itertools
import random
import unittest

import networkx as nx
from rdkit import Chem, rdBase

import snack


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.settings = (snack.config.NODE_FIRST, snack.config.ASCENDING)
        snack.config.set()

    def tearDown(self):
        snack.config.set(*self.settings)

    def test_named_attributes_round_trip_across_layouts_and_orderings(self):
        graph = nx.Graph()
        graph.add_node('hub', node_type='hub', size=2, label='3')
        graph.add_node('leaf', color='blue(1)%\n')
        graph.add_node('isolated')
        graph.add_edge('hub', 'leaf', edge_type='knows', weight=3.5)
        for node_first, ascending in itertools.product((False, True), repeat=2):
            snack.config.set(node_first, ascending)
            for ordering in (None, 'bfs', 'cm', 'uniform', 'neighbor'):
                with self.subTest(node_first=node_first, ascending=ascending, ordering=ordering):
                    random.seed(0)
                    text = snack.graph_to_snack(graph, ordering=ordering)
                    restored = snack.snack_to_graph(text)
                    self.assertTrue(nx.is_isomorphic(
                        graph, restored, node_match=lambda a, b: a == b,
                        edge_match=lambda a, b: a == b))

    def test_molecular_identity_across_layouts_and_orderings(self):
        molecules = ('CCO', '[13CH3]C', '[2H][H]', 'C[C@H](N)C(=O)O',
                     'F/C=C/F', 'F/C=C\\F', '[CH2]C', 'C[O]',
                     '[Na+].[Cl-]', '[O-][n+]1ccccc1', 'F[Xe]F')
        for node_first, ascending in itertools.product((False, True), repeat=2):
            snack.config.set(node_first, ascending)
            for ordering, smiles in itertools.product((None, 'bfs', 'cm', 'uniform', 'neighbor'), molecules):
                with self.subTest(smiles=smiles, ordering=ordering,
                                  node_first=node_first, ascending=ascending):
                    random.seed(0)
                    text = snack.smiles_to_snack(smiles, ordering=ordering)
                    self.assertEqual(snack.snack_to_smiles(text, stereo='strict'),
                                     Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))

    def test_invalid_valence_is_rejected_with_explicit_hydrogens(self):
        with rdBase.BlockLogs():
            for text in ('F1-Kr1-F', 'F1-Kr(H0)1-F', 'He(H8)', 'C(+)(H5)'):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    snack.snack_to_mol(text)

    def test_stereo_cleaning_and_strict_validation(self):
        text = 'C1-C(@)1-C2-C'
        self.assertEqual(snack.snack_to_smiles(text), 'CC(C)C')
        with self.assertRaises(ValueError):
            snack.snack_to_smiles(text, stereo='strict')
        self.assertFalse(snack.is_canonical_snack(text))
        self.assertFalse(snack.is_canonical_snack('C(H4)'))
        self.assertTrue(snack.is_canonical_snack('C(H2)1-C'))

    def test_isotopes_break_graph_symmetry(self):
        self.assertEqual(snack.count_automorphisms(snack.smiles_to_graph('CC')), 2)
        for smiles in ('[13CH3]C', '[2H][H]'):
            with self.subTest(smiles=smiles):
                graph = snack.smiles_to_graph(smiles)
                self.assertEqual(snack.count_automorphisms(graph), 1)
                self.assertEqual(snack.count_automorphisms(graph, stereo=True), 1)

    def test_weights_break_graph_symmetry(self):
        graph = nx.path_graph(3)
        self.assertEqual(snack.count_automorphisms(graph), 2)
        nx.set_edge_attributes(graph, {(0, 1): 1, (1, 2): 2}, 'weight')
        self.assertEqual(snack.count_automorphisms(graph), 1)

    def test_isotopes_and_weights_in_stereo_automorphisms(self):
        meso = snack.smiles_to_graph('O[C@@H](C)[C@@H](C)O')
        self.assertEqual(snack.count_automorphisms(meso), 2)
        self.assertEqual(snack.count_automorphisms(meso, stereo=True), 1)
        graph = snack.smiles_to_graph('O[C@H](C)[C@@H](C)O')
        self.assertEqual(snack.count_automorphisms(graph), 2)
        self.assertEqual(snack.count_automorphisms(graph, stereo=True), 2)
        labeled = graph.copy()
        labeled.nodes[2]['isotope'] = 13
        self.assertEqual(snack.count_automorphisms(labeled, stereo=True), 1)
        graph.edges[0, 1]['weight'] = 2
        self.assertEqual(snack.count_automorphisms(graph), 1)
        self.assertEqual(snack.count_automorphisms(graph, stereo=True), 1)

    def test_bundled_dataset_splits_and_notation(self):
        datasets = {
            'community_small': (70, 10, 20), 'ego': (529, 76, 152),
            'ego_small': (140, 20, 40), 'enzymes': (410, 59, 118),
            'grid': (70, 10, 20), 'grid_small': (70, 10, 20),
            'lobster': (69, 20, 11), 'planar': (128, 32, 40), 'sbm': (128, 32, 40),
        }
        for name, sizes in datasets.items():
            with self.subTest(dataset=name):
                splits = snack.load_dataset(name)
                self.assertEqual(tuple(map(len, splits.values())), sizes)
                for texts in splits.values():
                    for text in texts:
                        self.assertEqual(snack.graph_to_snack(snack.snack_to_graph(text)), text)


if __name__ == '__main__':
    unittest.main()
