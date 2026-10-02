"""
Tests for the snack package.

Run from the repository root: python -B -m unittest discover -s tests -v
Tests include bundled data, randomized graph/molecule round trips, strict input
validation, weighted graphs, and molecule conversion without stereochemistry.
"""
import unittest
import random
import networkx as nx
from rdkit import Chem

import csv
import copy
import math
from pathlib import Path
import subprocess
import sys

import snack

PACKAGE_DIR = Path(snack.__file__).resolve().parent

# Submodules for testing internal functions
from snack import graph as snack_graph
from snack import mol as snack_mol


class TestSmilesToGraph(unittest.TestCase):
    """Tests for snack.smiles_to_graph()."""

    def test_methane(self):
        """Test conversion of methane SMILES to graph."""
        G = snack.smiles_to_graph('C')
        self.assertEqual(len(G.nodes()), 1)

    def test_ethane(self):
        """Test conversion of ethane SMILES to graph."""
        G = snack.smiles_to_graph('CC')
        self.assertEqual(len(G.nodes()), 2)
        self.assertEqual(len(G.edges()), 1)

    def test_benzene_kekulized(self):
        """Test conversion of benzene with kekulization."""
        G = snack.smiles_to_graph('c1ccccc1', kekulize=True)
        self.assertEqual(len(G.nodes()), 6)
        self.assertEqual(len(G.edges()), 6)
        edge_types = [G.edges[e]['edge_type'] for e in G.edges()]
        self.assertTrue('-' in edge_types or '=' in edge_types)

    def test_benzene_aromatic(self):
        """Test conversion of benzene without kekulization."""
        G = snack.smiles_to_graph('c1ccccc1', kekulize=False)
        self.assertEqual(len(G.nodes()), 6)
        edge_types = [G.edges[e]['edge_type'] for e in G.edges()]
        self.assertTrue('a' in edge_types)

    def test_use_hydrogens_option(self):
        """Test use_hydrogens option."""
        G_with_h = snack.smiles_to_graph('C', use_hydrogens=True)
        G_without_h = snack.smiles_to_graph('C', use_hydrogens=False)
        self.assertIn('H', G_with_h.nodes[0]['node_type'])
        self.assertNotIn('H', G_without_h.nodes[0]['node_type'])

    def test_use_charge_option(self):
        """Test use_charge option."""
        G_with = snack.smiles_to_graph('[NH4+]', use_charge=True)
        G_without = snack.smiles_to_graph('[NH4+]', use_charge=False)
        self.assertIn('+', G_with.nodes[0]['node_type'])
        self.assertNotIn('+', G_without.nodes[0]['node_type'])


class TestGraphToSmiles(unittest.TestCase):
    """Tests for snack.graph_to_smiles()."""

    def test_simple_graph(self):
        """Test conversion of simple graph to SMILES."""
        G = nx.Graph()
        G.add_node(0, node_type='[CH4]')
        smiles = snack.graph_to_smiles(G)
        self.assertIsInstance(smiles, str)

    def test_ethane_graph(self):
        """Test conversion of ethane graph to SMILES."""
        G = nx.Graph()
        G.add_node(0, node_type='[CH3]')
        G.add_node(1, node_type='[CH3]')
        G.add_edge(0, 1, edge_type='-')
        smiles = snack.graph_to_smiles(G)
        self.assertIsNotNone(smiles)


class TestMolToGraph(unittest.TestCase):
    """Tests for snack.mol_to_graph()."""

    def test_methane(self):
        """Test conversion of methane mol to graph."""
        mol = Chem.MolFromSmiles('C')
        G = snack.mol_to_graph(mol)
        self.assertEqual(len(G.nodes()), 1)

    def test_ethane(self):
        """Test conversion of ethane mol to graph."""
        mol = Chem.MolFromSmiles('CC')
        G = snack.mol_to_graph(mol)
        self.assertEqual(len(G.nodes()), 2)
        self.assertEqual(len(G.edges()), 1)

    def test_node_types(self):
        """Test that node types are set correctly."""
        mol = Chem.MolFromSmiles('CO')
        G = snack.mol_to_graph(mol)
        node_types = [G.nodes[n]['node_type'] for n in G.nodes()]
        self.assertTrue(any('C' in nt for nt in node_types))
        self.assertTrue(any('O' in nt for nt in node_types))

    def test_use_hydrogens_option(self):
        """Test use_hydrogens=False option."""
        mol = Chem.MolFromSmiles('C')
        G = snack.mol_to_graph(mol, use_hydrogens=False)
        self.assertEqual(G.nodes[0]['node_type'], '[C]')


class TestGraphToMol(unittest.TestCase):
    """Tests for snack.graph_to_mol()."""

    def test_simple_graph(self):
        """Test conversion of simple graph to mol."""
        G = nx.Graph()
        G.add_node(0, node_type='[CH4]')
        mol = snack.graph_to_mol(G)
        self.assertIsNotNone(mol)

    def test_ethane_graph(self):
        """Test conversion of ethane graph to mol."""
        G = nx.Graph()
        G.add_node(0, node_type='[CH3]')
        G.add_node(1, node_type='[CH3]')
        G.add_edge(0, 1, edge_type='-')
        mol = snack.graph_to_mol(G)
        self.assertEqual(mol.GetNumAtoms(), 2)
        self.assertEqual(mol.GetNumBonds(), 1)

    def test_double_bond(self):
        """Test graph with double bond."""
        G = nx.Graph()
        G.add_node(0, node_type='[CH2]')
        G.add_node(1, node_type='[CH2]')
        G.add_edge(0, 1, edge_type='=')
        mol = snack.graph_to_mol(G)
        self.assertEqual(mol.GetBondWithIdx(0).GetBondType(), Chem.BondType.DOUBLE)

    def test_triple_bond(self):
        """Test graph with triple bond."""
        G = nx.Graph()
        G.add_node(0, node_type='[CH]')
        G.add_node(1, node_type='[CH]')
        G.add_edge(0, 1, edge_type='#')
        mol = snack.graph_to_mol(G)
        self.assertEqual(mol.GetBondWithIdx(0).GetBondType(), Chem.BondType.TRIPLE)


class TestSmilesToSnack(unittest.TestCase):
    """Tests for snack.smiles_to_snack()."""

    def test_simple_smiles(self):
        """Test conversion of simple SMILES to SNACK."""
        snack_str = snack.smiles_to_snack('CC')
        self.assertIsInstance(snack_str, str)
        self.assertTrue(len(snack_str) > 0)

    def test_with_order_cm(self):
        """Test smiles_to_snack with Cuthill-McKee ordering."""
        snack_str = snack.smiles_to_snack('CCCC', ordering='cm')
        self.assertIsInstance(snack_str, str)

    def test_with_order_bfs(self):
        """Test smiles_to_snack with BFS ordering."""
        snack_str = snack.smiles_to_snack('CCCC', ordering='bfs')
        self.assertIsInstance(snack_str, str)

    def test_default_order_is_bfs_and_none_preserves_input_order(self):
        state = random.getstate()
        try:
            for seed in range(5):
                random.seed(seed)
                default = snack.smiles_to_snack('CC(C)CO')
                random.seed(seed)
                explicit = snack.smiles_to_snack('CC(C)CO', ordering='bfs')
                self.assertEqual(default, explicit)
            graph = snack.smiles_to_graph('CC(C)CO')
            self.assertEqual(snack.smiles_to_snack('CC(C)CO', ordering=None),
                             snack.graph_to_snack(graph, ordering=None))
        finally:
            random.setstate(state)

    def test_with_order_uniform(self):
        """Test smiles_to_snack with uniform random ordering."""
        snack_str = snack.smiles_to_snack('CCCC', ordering='uniform')
        self.assertIsInstance(snack_str, str)

    def test_kekulize_option(self):
        """Test kekulize option."""
        snack_kek = snack.smiles_to_snack('c1ccccc1', kekulize=True)
        snack_arom = snack.smiles_to_snack('c1ccccc1', kekulize=False)
        self.assertIsInstance(snack_kek, str)
        self.assertIsInstance(snack_arom, str)

    def test_all_options(self):
        """Test all options together."""
        snack_str = snack.smiles_to_snack(
            '[NH4+]', ordering=None, kekulize=True,
            use_hydrogens=True, use_charge=True
        )
        self.assertIn('+', snack_str)
        self.assertIn('H', snack_str)


class TestSnackToSmiles(unittest.TestCase):
    """Tests for snack.snack_to_smiles()."""

    def test_simple_snack(self):
        """Test conversion of simple SNACK to SMILES."""
        smiles = snack.snack_to_smiles('[CH3]1-[CH3]')
        self.assertIsInstance(smiles, str)

    def test_roundtrip(self):
        """Test SMILES -> SNACK -> SMILES roundtrip."""
        original = 'CC'
        snack_str = snack.smiles_to_snack(original)
        recovered = snack.snack_to_smiles(snack_str)
        
        mol_orig = Chem.MolFromSmiles(original)
        mol_recovered = Chem.MolFromSmiles(recovered)
        self.assertEqual(Chem.MolToSmiles(mol_orig), Chem.MolToSmiles(mol_recovered))


class TestGraphToTokens(unittest.TestCase):
    """Tests for snack.graph_to_tokens()."""

    def test_simple_graph(self):
        """Test conversion of simple graph to tokens."""
        G = nx.Graph()
        G.add_node(0, node_type='[A]')
        G.add_node(1, node_type='[B]')
        G.add_edge(0, 1, edge_type='-')
        tokens = snack.graph_to_tokens(G)
        self.assertIsInstance(tokens, list)
        self.assertIn('[A]', tokens)
        self.assertIn('[B]', tokens)
        self.assertIn('-', tokens)

    def test_with_order_cm(self):
        """Test graph_to_tokens with Cuthill-McKee ordering."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        for i in G.nodes():
            G.nodes[i]['node_type'] = f'[{i}]'
        for e in G.edges():
            G.edges[e]['edge_type'] = '-'
        tokens = snack.graph_to_tokens(G, ordering='cm')
        self.assertIsInstance(tokens, list)

    def test_with_order_bfs(self):
        """Test graph_to_tokens with BFS ordering."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        for i in G.nodes():
            G.nodes[i]['node_type'] = f'[{i}]'
        for e in G.edges():
            G.edges[e]['edge_type'] = '-'
        tokens = snack.graph_to_tokens(G, ordering='bfs')
        self.assertIsInstance(tokens, list)

    def test_invalid_order_raises(self):
        """Test that invalid ordering raises ValueError."""
        G = nx.Graph()
        G.add_node(0, node_type='[A]')
        with self.assertRaises(ValueError):
            snack.graph_to_tokens(G, ordering='invalid')


class TestTokensToGraph(unittest.TestCase):
    """Tests for snack.tokens_to_graph()."""

    def test_simple_tokens(self):
        """Test conversion of simple tokens to graph."""
        tokens = ['[A]', '1', '-', '[B]']
        G = snack.tokens_to_graph(tokens)
        self.assertEqual(len(G.nodes()), 2)
        self.assertEqual(len(G.edges()), 1)

    def test_edge_types_preserved(self):
        """Test that edge types are preserved."""
        tokens = ['[A]', '1', '-', '[B]']
        G = snack.tokens_to_graph(tokens)
        self.assertEqual(G.edges[0, 1]['edge_type'], '-')

    def test_node_types_preserved(self):
        """Test that node types are preserved."""
        tokens = ['[C]', '1', '-', '[N]']
        G = snack.tokens_to_graph(tokens)
        self.assertEqual(G.nodes[0]['node_type'], '[C]')
        self.assertEqual(G.nodes[1]['node_type'], '[N]')

    def test_linear_chain(self):
        """Test conversion of linear chain."""
        tokens = ['[A]', '1', '-', '[B]', '1', '-', '[C]']
        G = snack.tokens_to_graph(tokens)
        self.assertEqual(len(G.nodes()), 3)
        self.assertEqual(len(G.edges()), 2)

    def test_with_separator(self):
        """Test tokens with separator for non-attributed edges."""
        tokens = ["[v]", "1_", "[v]"]
        G = snack.tokens_to_graph(tokens)
        self.assertEqual(len(G.nodes()), 2)
        self.assertEqual(len(G.edges()), 1)


class TestGraphToSnack(unittest.TestCase):
    """Tests for snack.graph_to_snack()."""

    def test_simple_graph(self):
        """Test conversion of simple graph to SNACK string."""
        G = nx.Graph()
        G.add_node(0, node_type='[A]')
        G.add_node(1, node_type='[B]')
        G.add_edge(0, 1, edge_type='-')
        snack_str = snack.graph_to_snack(G)
        self.assertIsInstance(snack_str, str)
        self.assertIn('[A]', snack_str)
        self.assertIn('[B]', snack_str)

    def test_with_order(self):
        """Test graph_to_snack with ordering."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        for i in G.nodes():
            G.nodes[i]['node_type'] = f'[{i}]'
        for e in G.edges():
            G.edges[e]['edge_type'] = '-'
        snack_str = snack.graph_to_snack(G, ordering='bfs')
        self.assertIsInstance(snack_str, str)


class TestSnackToGraph(unittest.TestCase):
    """Tests for snack.snack_to_graph()."""

    def test_simple_snack(self):
        """Test conversion of simple SNACK string to graph."""
        G = snack.snack_to_graph('[A]1-[B]')
        self.assertEqual(len(G.nodes()), 2)
        self.assertEqual(len(G.edges()), 1)

    def test_roundtrip(self):
        """Test graph -> snack -> graph roundtrip."""
        G_orig = nx.Graph()
        G_orig.add_node(0, node_type='[A]')
        G_orig.add_node(1, node_type='[B]')
        G_orig.add_node(2, node_type='[C]')
        G_orig.add_edge(0, 1, edge_type='-')
        G_orig.add_edge(1, 2, edge_type='=')
        
        snack_str = snack.graph_to_snack(G_orig)
        G_new = snack.snack_to_graph(snack_str)
        
        self.assertEqual(len(G_new.nodes()), len(G_orig.nodes()))
        self.assertEqual(len(G_new.edges()), len(G_orig.edges()))


class TestSplit(unittest.TestCase):
    """Tests for snack.split()."""

    def test_token_granularity(self):
        """Test split with token granularity."""
        tokens = snack.split('[A]1-[B]2=[C]', pattern='token')
        self.assertIn('[A]', tokens)
        self.assertIn('[B]', tokens)
        self.assertIn('[C]', tokens)
        self.assertIn('1', tokens)
        self.assertIn('2', tokens)
        self.assertIn('-', tokens)
        self.assertIn('=', tokens)

    def test_element_granularity(self):
        """Test split with element granularity."""
        elements = snack.split('[A]1-[B]2=[C]', pattern='graph')
        self.assertIn('[A]', elements)
        self.assertIn('[B]', elements)
        self.assertIn('[C]', elements)
        self.assertTrue(any('1-' in e for e in elements))

    def test_block_granularity(self):
        """Test split with block granularity."""
        blocks = snack.split('[A]1-[B]2=[C]', pattern='block')
        self.assertIsInstance(blocks, list)
        self.assertTrue(len(blocks) > 0)

    def test_invalid_granularity_raises(self):
        """Test that invalid granularity raises ValueError."""
        with self.assertRaises(ValueError):
            snack.split('[A]1-[B]', pattern='invalid')


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline using public API."""

    def test_smiles_full_roundtrip(self):
        """Test full roundtrip: SMILES -> graph -> snack -> graph -> mol -> SMILES."""
        test_smiles = ['C', 'CC', 'CCC', 'C=C', 'C#C', 'CCO', 'CCN']
        
        for smiles in test_smiles:
            with self.subTest(smiles=smiles):
                graph1 = snack.smiles_to_graph(smiles)
                snack_str = snack.graph_to_snack(graph1)
                graph2 = snack.snack_to_graph(snack_str)
                final_smiles = snack.graph_to_smiles(graph2)
                
                mol_orig = Chem.MolFromSmiles(smiles)
                mol_final = Chem.MolFromSmiles(final_smiles)
                
                self.assertEqual(mol_orig.GetNumAtoms(), mol_final.GetNumAtoms())
                self.assertEqual(mol_orig.GetNumBonds(), mol_final.GetNumBonds())

    def test_graph_token_roundtrip(self):
        """Test graph -> tokens -> graph roundtrip."""
        G_orig = nx.Graph()
        G_orig.add_node(0, node_type='[A]')
        G_orig.add_node(1, node_type='[B]')
        G_orig.add_node(2, node_type='[C]')
        G_orig.add_edge(0, 1, edge_type='-')
        G_orig.add_edge(1, 2, edge_type='=')
        
        tokens = snack.graph_to_tokens(G_orig)
        G_new = snack.tokens_to_graph(tokens)
        
        self.assertEqual(len(G_new.nodes()), len(G_orig.nodes()))
        self.assertEqual(len(G_new.edges()), len(G_orig.edges()))

    def test_different_orderings_produce_equivalent_graphs(self):
        """Test that different orderings produce equivalent graphs."""
        smiles = 'CCCC'
        
        for order in [None, 'cm', 'bfs', 'uniform']:
            with self.subTest(ordering=order):
                snack_str = snack.smiles_to_snack(smiles, ordering=order)
                smiles_back = snack.snack_to_smiles(snack_str)
                mol = Chem.MolFromSmiles(smiles_back)
                self.assertEqual(mol.GetNumAtoms(), 4)
                self.assertEqual(mol.GetNumBonds(), 3)


# =============================================================================
# Internal Function Tests (using submodules)
# =============================================================================

class TestInternalSetting(unittest.TestCase):
    """Tests for the public config.set() API."""

    def setUp(self):
        snack.config.set()

    def tearDown(self):
        snack.config.set()

    def test_default_values(self):
        """Test that default settings are applied correctly."""
        self.assertFalse(snack_graph.NODE_FIRST)
        self.assertFalse(snack_graph.ASCENDING)
        self.assertEqual(snack_graph.DEFAULT_NODE_TOKEN, 'v')

    def test_node_first(self):
        """Test setting node_first option."""
        snack.config.set(node_first=True)
        self.assertTrue(snack_graph.NODE_FIRST)

    def test_ascending(self):
        """Test setting ascending option."""
        snack.config.set(ascending=True)
        self.assertTrue(snack_graph.ASCENDING)

    def test_default_node_token(self):
        """Default token is classified as an unattributed node."""
        self.assertEqual(snack.token_type(snack_graph.DEFAULT_NODE_TOKEN), 'node_type')

    def test_node_first_affects_output(self):
        """Test that node_first setting affects token order."""
        G = nx.Graph()
        G.add_node(0, node_type='[A]')
        G.add_node(1, node_type='[B]')
        G.add_edge(0, 1, edge_type='-')
        
        snack.config.set(node_first=True)
        tokens = snack.graph_to_tokens(G)
        self.assertEqual(tokens[0], '[A]')


class TestInternalOrderFunctions(unittest.TestCase):
    """Tests for internal graph ordering functions."""

    def setUp(self):
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
        random.seed(42)

    def test_random_bfs_order_returns_all_nodes(self):
        """Test that random_bfs_order returns all nodes."""
        order = snack_graph.random_bfs_order(self.G)
        self.assertEqual(set(order), set(self.G.nodes()))
        self.assertEqual(len(order), len(self.G.nodes()))

    def test_random_bfs_order_type(self):
        """Test that random_bfs_order returns a list of integers."""
        order = snack_graph.random_bfs_order(self.G)
        self.assertIsInstance(order, list)
        for node in order:
            self.assertIsInstance(node, int)

    def test_uniform_random_order_returns_all_nodes(self):
        """Test that uniform_random_order returns all nodes."""
        order = snack_graph.uniform_random_order(self.G)
        self.assertEqual(set(order), set(self.G.nodes()))
        self.assertEqual(len(order), len(self.G.nodes()))

    def test_pseudo_peripheral_node_returns_valid_node(self):
        """Test that pseudo_peripheral_node returns a node in the graph."""
        node = snack_graph.pseudo_peripheral_node(self.G)
        self.assertIn(node, self.G.nodes())
        self.assertIsInstance(node, int)

    def test_random_cuthill_mckee_order_returns_all_nodes(self):
        """Test that random_cuthill_mckee_order returns all nodes."""
        order = snack_graph.random_cuthill_mckee_order(self.G)
        self.assertEqual(set(order), set(self.G.nodes()))
        self.assertEqual(len(order), len(self.G.nodes()))

    def test_random_neighbor_order_returns_all_nodes(self):
        """Neighbor ordering visits each node once."""
        order = snack_graph.random_neighbor_order(self.G)
        self.assertEqual(len(order), len(self.G))
        self.assertEqual(set(order), set(self.G))

    def test_order_funcs_dictionary(self):
        """Test that ORDER_FUNCS dictionary contains expected functions."""
        self.assertIn('cm', snack_graph.ORDER_FUNCS)
        self.assertIn('bfs', snack_graph.ORDER_FUNCS)
        self.assertIn('uniform', snack_graph.ORDER_FUNCS)


class TestInternalAtomToString(unittest.TestCase):
    """Tests for internal atom_to_string function."""

    def test_simple_carbon(self):
        """Test conversion of simple carbon atom."""
        mol = Chem.MolFromSmiles('C')
        atom = mol.GetAtomWithIdx(0)
        result = snack_mol.atom_to_string(atom)
        self.assertIn('C', result)
        self.assertTrue(result.startswith('['))
        self.assertTrue(result.endswith(']'))

    def test_with_hydrogens(self):
        """Test atom with hydrogens."""
        mol = Chem.MolFromSmiles('C')
        atom = mol.GetAtomWithIdx(0)
        result = snack_mol.atom_to_string(atom, use_hydrogens=True)
        self.assertIn('H', result)

    def test_without_hydrogens(self):
        """Test atom without hydrogens."""
        mol = Chem.MolFromSmiles('C')
        atom = mol.GetAtomWithIdx(0)
        result = snack_mol.atom_to_string(atom, use_hydrogens=False)
        self.assertEqual(result, '[C]')

    def test_without_hydrogens_discards_explicit_counts(self):
        for smiles, expected in (('[CH2]', '[C]'), ('[C]', '[C]'),
                                 ('[NH4+]', '[N+]'), ('[13CH2-:7]', '[13C-:7]')):
            with self.subTest(smiles=smiles):
                atom = Chem.MolFromSmiles(smiles).GetAtomWithIdx(0)
                self.assertEqual(snack_mol.atom_to_string(atom, use_hydrogens=False), expected)

    def test_positive_charge(self):
        """Test positively charged atom."""
        mol = Chem.MolFromSmiles('[NH4+]')
        atom = mol.GetAtomWithIdx(0)
        result = snack_mol.atom_to_string(atom, use_charge=True)
        self.assertIn('+', result)

    def test_negative_charge(self):
        """Test negatively charged atom."""
        mol = Chem.MolFromSmiles('[O-]')
        atom = mol.GetAtomWithIdx(0)
        result = snack_mol.atom_to_string(atom, use_charge=True)
        self.assertIn('-', result)

    def test_without_charge(self):
        """Test atom without showing charge."""
        mol = Chem.MolFromSmiles('[NH4+]')
        atom = mol.GetAtomWithIdx(0)
        result = snack_mol.atom_to_string(atom, use_charge=False)
        self.assertNotIn('+', result)


class TestInternalStringToAtom(unittest.TestCase):
    """Tests for internal string_to_atom function."""

    def test_simple_carbon(self):
        """Test conversion of simple carbon string."""
        atom = snack_mol.string_to_atom('[C]')
        self.assertEqual(atom.GetSymbol(), 'C')
        self.assertEqual(atom.GetFormalCharge(), 0)

    def test_carbon_with_hydrogens(self):
        """Test conversion of carbon with hydrogens."""
        atom = snack_mol.string_to_atom('[CH3]')
        self.assertEqual(atom.GetSymbol(), 'C')
        self.assertEqual(atom.GetNumExplicitHs(), 3)

    def test_single_hydrogen(self):
        """Test atom with single hydrogen (H without number)."""
        atom = snack_mol.string_to_atom('[CH]')
        self.assertEqual(atom.GetNumExplicitHs(), 1)

    def test_positive_charge(self):
        """Test atom with positive charge."""
        atom = snack_mol.string_to_atom('[N+]')
        self.assertEqual(atom.GetSymbol(), 'N')
        self.assertEqual(atom.GetFormalCharge(), 1)

    def test_negative_charge(self):
        """Test atom with negative charge."""
        atom = snack_mol.string_to_atom('[O-]')
        self.assertEqual(atom.GetSymbol(), 'O')
        self.assertEqual(atom.GetFormalCharge(), -1)

    def test_double_positive_charge(self):
        """Test atom with double positive charge."""
        atom = snack_mol.string_to_atom('[Ca++]')
        self.assertEqual(atom.GetSymbol(), 'Ca')
        self.assertEqual(atom.GetFormalCharge(), 2)

    def test_numeric_charge(self):
        """Test atom with numeric charge notation."""
        atom = snack_mol.string_to_atom('[Fe+3]')
        self.assertEqual(atom.GetSymbol(), 'Fe')
        self.assertEqual(atom.GetFormalCharge(), 3)

    def test_complex_atom(self):
        """Test complex atom string with hydrogens and charge."""
        atom = snack_mol.string_to_atom('[NH4+]')
        self.assertEqual(atom.GetSymbol(), 'N')
        self.assertEqual(atom.GetNumExplicitHs(), 4)
        self.assertEqual(atom.GetFormalCharge(), 1)

    def test_two_letter_element(self):
        """Test two-letter element symbol."""
        atom = snack_mol.string_to_atom('[Cl]')
        self.assertEqual(atom.GetSymbol(), 'Cl')

    def test_invalid_atom_string_raises(self):
        """Test that invalid atom string raises ValueError."""
        with self.assertRaises(ValueError):
            snack_mol.string_to_atom('[123invalid]')


class TestInternalBondTypeConversions(unittest.TestCase):
    """Tests for internal bond type conversion dictionaries."""

    def test_bondtype_to_symbol_completeness(self):
        """Test that BONDTYPE_TO_SYMBOL contains expected bond types."""
        self.assertIn(Chem.BondType.SINGLE, snack_mol.BONDTYPE_TO_SYMBOL)
        self.assertIn(Chem.BondType.DOUBLE, snack_mol.BONDTYPE_TO_SYMBOL)
        self.assertIn(Chem.BondType.TRIPLE, snack_mol.BONDTYPE_TO_SYMBOL)
        self.assertIn(Chem.BondType.AROMATIC, snack_mol.BONDTYPE_TO_SYMBOL)

    def test_symbol_to_bondtype_completeness(self):
        """Test that SYMBOL_TO_BONDTYPE contains expected symbols."""
        self.assertIn('-', snack_mol.SYMBOL_TO_BONDTYPE)
        self.assertIn('=', snack_mol.SYMBOL_TO_BONDTYPE)
        self.assertIn('#', snack_mol.SYMBOL_TO_BONDTYPE)
        self.assertIn('a', snack_mol.SYMBOL_TO_BONDTYPE)

    def test_bondtype_symbol_roundtrip(self):
        """Test that bondtype -> symbol -> bondtype is consistent."""
        for bt, sym in snack_mol.BONDTYPE_TO_SYMBOL.items():
            self.assertEqual(snack_mol.SYMBOL_TO_BONDTYPE[sym], bt)




class RegressionCase(unittest.TestCase):
    def setUp(self):
        self.saved_config = (snack_graph.NODE_FIRST, snack_graph.ASCENDING)
        self.saved_random = random.getstate()
        snack.config.set()
        random.seed(1729)

    def tearDown(self):
        snack.config.set(*self.saved_config)
        random.setstate(self.saved_random)

    def assert_graph_equal(self, original, recovered):
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            original, recovered,
            node_match=nx.algorithms.isomorphism.categorical_node_match('node_type', None),
            edge_match=nx.algorithms.isomorphism.categorical_edge_match(['edge_type', 'weight'], [None, None]))
        self.assertTrue(matcher.is_isomorphic(),
                        f'{list(original.nodes(data=True))}, {list(original.edges(data=True))}'
                        f' != {list(recovered.nodes(data=True))}, {list(recovered.edges(data=True))}')

    def exact_automorphisms(self, graph):
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            graph, graph,
            node_match=nx.algorithms.isomorphism.categorical_node_match('node_type', None),
            edge_match=nx.algorithms.isomorphism.categorical_edge_match(['edge_type', 'weight'], [None, None]))
        return sum(1 for _ in matcher.isomorphisms_iter())


class TestPackageImport(RegressionCase):
    def test_clean_import(self):
        result = subprocess.run(
            [sys.executable, '-B', '-c',
             'import sys; sys.path.insert(0, sys.argv[1]); import snack; print(snack.__file__)',
             str(PACKAGE_DIR.parent)],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(PACKAGE_DIR / '__init__.py'), result.stdout)


class TestGraphRegression(RegressionCase):
    def test_roundtrip_matrix(self):
        """800 cases: 40 graphs x 5 orderings x 4 formatting combinations."""
        for seed in range(40):
            graph = nx.gnp_random_graph(seed % 11, (seed % 5) / 4, seed=seed)
            rng = random.Random(seed)
            if seed % 2:
                nx.set_node_attributes(graph, {n: rng.choice(['[C]', '[N+]', '[O-]'])
                                               for n in graph}, 'node_type')
            if seed % 3:
                nx.set_edge_attributes(graph, {e: rng.choice(['-', '=', '#', 'a'])
                                               for e in graph.edges}, 'edge_type')
            for node_first in (False, True):
                for ascending in (False, True):
                    snack.config.set(node_first=node_first, ascending=ascending)
                    for ordering in (None, 'bfs', 'cm', 'uniform', 'neighbor'):
                        with self.subTest(seed=seed, node_first=node_first,
                                          ascending=ascending, ordering=ordering):
                            before = copy.deepcopy((dict(graph.nodes(data=True)), dict(((u, v), dict(d))
                                      for u, v, d in graph.edges(data=True))))
                            encoded = snack.graph_to_snack(graph, ordering=ordering)
                            self.assert_graph_equal(graph, snack.snack_to_graph(encoded))
                            self.assertEqual(before, (dict(graph.nodes(data=True)),
                                dict(((u, v), dict(d)) for u, v, d in graph.edges(data=True))))

    def test_bundled_graph_data(self):
        """Check every bundled graph's node count and default serialization round trip."""
        paths = sorted((PACKAGE_DIR / 'data').glob('*.csv'))
        self.assertTrue(paths, 'Bundled CSV fixtures are missing')
        for path in paths:
            with path.open(newline='') as handle:
                for i, row in enumerate(csv.DictReader(handle), 2):
                    with self.subTest(file=path.name, line=i):
                        graph = snack.snack_to_graph(row['snack'])
                        # ego.csv uses n_nodes; the other fixtures use n_node.
                        expected_nodes = row.get('n_node', row.get('n_nodes'))
                        self.assertEqual(len(graph), int(expected_nodes))
                        self.assertEqual(snack.graph_to_snack(graph), row['snack'])

    def test_split_unattributed_graph_pattern(self):
        self.assertEqual(snack.split('v1_v', pattern='graph'), ['v', '1_', 'v'])

    def test_split_unattributed_blocks(self):
        for node_first, encoded, expected in [
            (False, 'v1_v2_1_v', ['v', '1_v', '2_1_v']),
            (True, 'vv1_v2_1_', ['v', 'v1_', 'v2_1_'])]:
            with self.subTest(node_first=node_first):
                snack.config.set(node_first=node_first)
                self.assertEqual(snack.split(encoded, pattern='block'), expected)

    def test_all_exported_bond_symbols_roundtrip(self):
        for symbol in snack_mol.SYMBOL_TO_BONDTYPE:
            with self.subTest(symbol=symbol):
                graph = nx.Graph()
                graph.add_edge(0, 1, edge_type=symbol)
                self.assert_graph_equal(graph, snack.snack_to_graph(snack.graph_to_snack(graph)))

    def test_weight_does_not_remove_connectivity(self):
        for weight in (0, 0.5, 2, -1):
            with self.subTest(weight=weight):
                graph = nx.Graph()
                graph.add_edge(0, 1, weight=weight)
                self.assert_graph_equal(graph, snack.snack_to_graph(snack.graph_to_snack(graph)))

    def test_arbitrary_node_labels_default_order(self):
        for labels in [('a', 'b'), (10, 20)]:
            with self.subTest(labels=labels):
                graph = nx.Graph()
                graph.add_edge(*labels)
                self.assert_graph_equal(graph, snack.snack_to_graph(snack.graph_to_snack(graph)))

    def test_arbitrary_node_labels_explicit_order(self):
        graph = nx.Graph([('a', 'b'), ('b', 'c')])
        for ordering in snack_graph.ORDER_FUNCS:
            with self.subTest(ordering=ordering):
                self.assert_graph_equal(graph, snack.snack_to_graph(
                    snack.graph_to_snack(graph, ordering=ordering)))

    def test_reject_malformed_snack(self):
        for encoded in ('[C]]', 'v0_v', 'v1_', 'v1__v', 'v1-v='):
            with self.subTest(encoded=encoded):
                with self.assertRaises((ValueError, AssertionError)):
                    snack.snack_to_graph(encoded)

    def test_reject_unsupported_self_loop(self):
        graph = nx.Graph([(0, 0)])
        with self.assertRaises((ValueError, TypeError, nx.NetworkXException)):
            snack.graph_to_snack(graph)

    def test_reject_unsupported_directed_graph(self):
        graph = nx.DiGraph([(0, 1)])
        with self.assertRaises((ValueError, TypeError, nx.NetworkXException)):
            snack.graph_to_snack(graph)

    def test_reject_unsupported_parallel_edges(self):
        graph = nx.MultiGraph([(0, 1), (0, 1)])
        with self.assertRaises((ValueError, TypeError, nx.NetworkXException)):
            snack.graph_to_snack(graph)

    def test_filters(self):
        encoded = '[NH4+]1-[Cl-]'
        self.assertEqual(snack.filter(encoded, 'node'), ['[NH4+]', '[Cl-]'])
        self.assertEqual(snack.filter(encoded, 'dist'), ['1'])
        self.assertEqual(snack.filter(encoded, 'edge'), ['-'])

    def test_relabel_nodes_preserves_attributes(self):
        graph = nx.Graph()
        graph.add_node('a', node_type='[N]')
        graph.add_node('b', node_type='[C]')
        graph.add_edge('a', 'b', edge_type='=')
        result = snack.relabel_nodes(graph, ['b', 'a'])
        self.assertEqual(result.nodes[0]['node_type'], '[C]')
        self.assertEqual(result.nodes[1]['node_type'], '[N]')
        self.assertEqual(result.edges[0, 1]['edge_type'], '=')
        self.assertEqual(set(graph), {'a', 'b'})


class TestAutomorphismRegression(RegressionCase):
    def test_isolated_vertices(self):
        self.assertEqual(snack.count_automorphisms(nx.empty_graph(3)), 6)

    def test_untyped_edges_preserve_node_colors(self):
        graph = nx.path_graph(3)
        nx.set_node_attributes(graph, {0: '[N]', 1: '[C]', 2: '[O]'}, 'node_type')
        self.assertEqual(snack.count_automorphisms(graph), self.exact_automorphisms(graph))

    def test_missing_node_type_is_distinct(self):
        graph = nx.path_graph(2)
        graph.nodes[0]['node_type'] = '[C]'
        nx.set_edge_attributes(graph, '-', 'edge_type')
        self.assertEqual(snack.count_automorphisms(graph), 1)

    def test_missing_edge_type_is_distinct(self):
        graph = nx.path_graph(3)
        graph.edges[0, 1]['edge_type'] = '-'
        self.assertEqual(snack.count_automorphisms(graph), 1)

    def test_large_group_order(self):
        graph = nx.complete_graph(15)
        nx.set_node_attributes(graph, '[C]', 'node_type')
        nx.set_edge_attributes(graph, '-', 'edge_type')
        self.assertAlmostEqual(snack.count_automorphisms(graph) / math.factorial(15), 1.0)

    def test_return_type(self):
        self.assertIsInstance(snack.count_automorphisms(nx.path_graph(3)), int)

    def test_small_fully_attributed_graphs_against_networkx(self):
        for seed in range(100):
            rng = random.Random(seed)
            graph = nx.gnp_random_graph(1 + seed % 6, (seed % 5) / 4, seed=seed)
            nx.set_node_attributes(graph, {n: rng.choice(['[C]', '[N]', '[O]'])
                                           for n in graph}, 'node_type')
            nx.set_edge_attributes(graph, {e: rng.choice(['-', '='])
                                           for e in graph.edges}, 'edge_type')
            with self.subTest(seed=seed):
                self.assertEqual(snack.count_automorphisms(graph), self.exact_automorphisms(graph))


class TestMoleculeRegression(RegressionCase):
    def assert_smiles_roundtrip(self, smiles, **kwargs):
        expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        encoded = snack.smiles_to_snack(smiles, **kwargs)
        actual = snack.snack_to_smiles(encoded)
        self.assertEqual(actual, expected, f'{smiles} -> {encoded} -> {actual}')

    def test_common_molecule_matrix(self):
        """1,040 cases: 13 molecules x 5 orders x 4 configs x 2 H x 2 aromatic modes."""
        for smiles in ('', 'C', 'CCO', 'C=C', 'C#N', 'c1ccccc1', 'c1ccncc1',
                       'CC(=O)O', 'C1CCCCC1', '[NH4+]', '[Na+].[Cl-]', 'CC.O', 'ClCBr'):
            for ordering in (None, 'bfs', 'cm', 'uniform', 'neighbor'):
                for node_first in (False, True):
                    for ascending in (False, True):
                        snack.config.set(node_first=node_first, ascending=ascending)
                        for use_hydrogens in (False, True):
                            for kekulize in (False, True):
                                with self.subTest(smiles=smiles, ordering=ordering,
                                                  node_first=node_first, ascending=ascending,
                                                  use_hydrogens=use_hydrogens, kekulize=kekulize):
                                    self.assert_smiles_roundtrip(smiles, ordering=ordering,
                                        use_hydrogens=use_hydrogens, kekulize=kekulize)

    def test_aromatic_nh_roundtrip(self):
        for kekulize in (False, True):
            with self.subTest(kekulize=kekulize):
                self.assert_smiles_roundtrip('c1cc[nH]c1', kekulize=kekulize,
                                            use_hydrogens=True)

    def test_mol_to_snack_ordering(self):
        mol = Chem.MolFromSmiles('CCCO')
        for ordering in (None, 'bfs', 'cm', 'uniform', 'neighbor'):
            with self.subTest(ordering=ordering):
                encoded = snack.mol_to_snack(mol, ordering=ordering)
                self.assertEqual(snack.mol_to_smiles(snack.snack_to_mol(encoded)), 'CCCO')

    def test_radical_and_explicit_hydrogens_preserved(self):
        for smiles in ('[CH2]c1c[nH]c2ccccc12', '[CH3]', '[CH2]', '[CH]', '[C]',
                       '[O]', '[O]C', '[N]=O', '[NH]', '[B]C', '[c]1ccccc1',
                       '[CH2-]', '[O-]', '[Cl]'):
            original = Chem.MolFromSmiles(smiles)
            expected = Chem.MolToSmiles(original)
            radicals = sorted(a.GetNumRadicalElectrons() for a in original.GetAtoms())
            for ordering in (None, 'bfs', 'uniform'):
                for node_first in (False, True):
                    for ascending in (False, True):
                        snack.config.set(node_first, ascending)
                        for hydrogens in (False, True):
                            with self.subTest(smiles=smiles, ordering=ordering,
                                              node_first=node_first, ascending=ascending,
                                              hydrogens=hydrogens):
                                text = snack.smiles_to_snack(smiles, ordering=ordering,
                                    attributed=True, use_hydrogens=hydrogens)
                                self.assertNotIn(';r=', text)
                                if not hydrogens:
                                    for token in snack.split(text):
                                        if snack_graph.token_type(token) == 'node_type':
                                            self.assertIsNone(snack_mol.RE_ATOM_TYPE.fullmatch(
                                                token[1:-1])['hcount'])
                                    continue
                                restored = snack.snack_to_mol(text)
                                self.assertEqual(Chem.MolToSmiles(restored), expected)
                                self.assertEqual(sorted(a.GetNumRadicalElectrons()
                                                        for a in restored.GetAtoms()), radicals)

    def test_discarding_hydrogen_counts_can_change_radicals(self):
        self.assertEqual(snack.snack_to_smiles('[CH3]'), '[CH3]')
        self.assertEqual(snack.snack_to_smiles('[C]'), 'C')
        self.assertEqual(snack.smiles_to_snack('[C]', use_hydrogens=True), '[CH0]')

    def test_requested_radical_without_hydrogens(self):
        smiles = '[CH2]c1c[nH]c2ccccc12'
        original = Chem.MolFromSmiles(smiles)
        for node_first in (False, True):
            for ascending in (False, True):
                snack.config.set(node_first, ascending)
                text = snack.smiles_to_snack(smiles, ordering='bfs', attributed=True,
                                            use_hydrogens=False)
                graph = snack.snack_to_graph(text)
                self.assertEqual({data['node_type'] for _, data in graph.nodes(data=True)},
                                 {'[C]', '[N]'})
                restored = snack.snack_to_mol(text)
                self.assertEqual(restored.GetNumAtoms(), original.GetNumAtoms())
                self.assertEqual(restored.GetNumBonds(), original.GetNumBonds())
                self.assertEqual(Chem.MolToSmiles(restored), 'Cc1c[nH]c2ccccc12')
                self.assertNotEqual(Chem.MolToSmiles(restored), Chem.MolToSmiles(original))
        for convert, value in ((snack.smiles_to_graph, smiles), (snack.mol_to_graph, original)):
            graph = convert(value, use_hydrogens=False)
            self.assertEqual({data['node_type'] for _, data in graph.nodes(data=True)},
                             {'[C]', '[N]'})

    def test_isotope_preserved(self):
        self.assert_smiles_roundtrip('[13CH4]')

    def test_tetrahedral_stereochemistry_omitted(self):
        for hydrogens in (False, True):
            text = snack.smiles_to_snack('N[C@@H](C)C(=O)O', use_hydrogens=hydrogens)
            self.assertNotIn('@', text)
            self.assertEqual(snack.snack_to_smiles(text), 'CC(N)C(=O)O')

    def test_double_bond_stereochemistry_omitted(self):
        for smiles in ('F/C=C/F', 'F/C=C\\F'):
            text = snack.smiles_to_snack(smiles)
            self.assertEqual(snack.snack_to_smiles(text), 'FC=CF')

    def test_wildcard_atom_roundtrip(self):
        self.assert_smiles_roundtrip('*C')

    def test_invalid_smiles_raises_value_error(self):
        from rdkit import rdBase
        with rdBase.BlockLogs():
            for kekulize in (False, True):
                with self.subTest(kekulize=kekulize):
                    with self.assertRaises(ValueError):
                        snack.smiles_to_mol('invalid_smiles', kekulize=kekulize)

    def test_invalid_valence_rejected(self):
        from rdkit import rdBase
        graph = nx.star_graph(5)
        nx.set_node_attributes(graph, '[C]', 'node_type')
        nx.set_edge_attributes(graph, '-', 'edge_type')
        with rdBase.BlockLogs(), self.assertRaises(ValueError):
            snack.graph_to_mol(graph)

    def test_atom_charge_roundtrip(self):
        for text, charge in [('[N+]', 1), ('[O-]', -1), ('[Fe+3]', 3),
                             ('[Ca++]', 2), ('[O--]', -2), ('[N-3]', -3)]:
            with self.subTest(text=text):
                atom = snack_mol.string_to_atom(text)
                self.assertEqual(atom.GetFormalCharge(), charge)

    def test_canonicalize_and_randomize(self):
        expected = snack_mol.canonicalize_smiles('OCC')
        self.assertEqual(expected, 'CCO')
        for _ in range(10):
            self.assertEqual(snack_mol.canonicalize_smiles(snack_mol.randomize_smiles('OCC')),
                             expected)
        from rdkit import rdBase
        with rdBase.BlockLogs():
            self.assertIsNone(snack_mol.canonicalize_smiles('invalid_smiles'))
            self.assertIsNone(snack_mol.randomize_smiles('invalid_smiles'))




class TestExtendedNotation(RegressionCase):
    def test_edge_labels_are_literal_under_reordering(self):
        for symbol in ('=cis', '=trans', '=unknown', 'dative', 'dative-reverse'):
            graph = nx.Graph()
            graph.add_nodes_from([(0, {'node_type': '[C]'}), (1, {'node_type': '[N]'})])
            graph.add_edge(0, 1, edge_type=symbol, weight=3)
            for ordering in (None, 'bfs', 'cm', 'uniform', 'neighbor'):
                for node_first in (False, True):
                    for ascending in (False, True):
                        snack.config.set(node_first, ascending)
                        with self.subTest(symbol=symbol, ordering=ordering,
                                          node_first=node_first, ascending=ascending):
                            encoded = snack.graph_to_snack(graph, ordering=ordering)
                            self.assert_graph_equal(graph, snack.snack_to_graph(encoded))

    def test_complex_edge_spelling(self):
        for symbol in snack_mol.SYMBOL_TO_BONDTYPE:
            with self.subTest(symbol=symbol):
                graph = nx.Graph()
                graph.add_nodes_from([(0, {'node_type': '[C]'}), (1, {'node_type': '[C]'})])
                graph.add_edge(0, 1, edge_type=symbol)
                edge = symbol if symbol in {'-', '=', '#', 'a', 'u'} else f'({symbol})'
                encoded = f'[C]1{edge}[C]'
                self.assertEqual(snack.graph_to_snack(graph), encoded)
                self.assertEqual(snack.split(encoded), ['[C]', '1', edge, '[C]'])
                self.assertEqual(snack.filter(encoded, 'edge'), [edge])
                self.assert_graph_equal(graph, snack.snack_to_graph(encoded))

    def test_weight_spelling_and_roundtrip(self):
        for weight in (0, 1, 3, -2, 0.5, 1e-20, 1e20, 10**100):
            for node_first in (False, True):
                with self.subTest(weight=weight, node_first=node_first):
                    snack.config.set(node_first=node_first)
                    graph = nx.Graph()
                    graph.add_edge('x', 'y', weight=weight)
                    expected = f'vv1(weight={weight})' if node_first else f'v1(weight={weight})v'
                    self.assertEqual(snack.graph_to_snack(graph), expected)
                    recovered = snack.snack_to_graph(expected)
                    self.assertEqual(recovered.edges[0, 1], {'weight': weight})
                    self.assert_graph_equal(graph, recovered)

    def test_combined_type_and_weight(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, edge_type='dative', weight=3)
        encoded = snack.graph_to_snack(graph)
        self.assertEqual(encoded, 'v1(type=dative;weight=3)v')
        self.assertEqual(snack.snack_to_graph(encoded).edges[0, 1],
                         {'edge_type': 'dative', 'weight': 3})

    def test_escaped_edge_types(self):
        for symbol in ('weight=3', 'type=dative;weight=3', 'a(b)', 'a[b]', 'bond type',
                       'bond%type', 'λ', 'node/v'):
            for weight in (None, 2):
                with self.subTest(symbol=symbol, weight=weight):
                    attributes = {'edge_type': symbol}
                    if weight is not None:
                        attributes['weight'] = weight
                    graph = nx.Graph()
                    graph.add_edge(0, 1, **attributes)
                    self.assert_graph_equal(graph, snack.snack_to_graph(snack.graph_to_snack(graph)))

    def test_unattributed_option_discards_attributes_only(self):
        graph = nx.Graph()
        graph.add_nodes_from([(0, {'node_type': '[C]'}), (1, {'node_type': '[N]'})])
        graph.add_edge(0, 1, weight=0, edge_type='dative')
        before = copy.deepcopy(graph)
        self.assertEqual(snack.graph_to_snack(graph, attributed=False), 'v1_v')
        self.assertEqual(snack.graph_to_tokens(graph, attributed=False), ['v', '1_', 'v'])
        self.assertEqual(list(snack.snack_to_graph('v1_v').edges(data=True)), [(0, 1, {})])
        self.assertEqual(dict(graph.nodes(data=True)), dict(before.nodes(data=True)))
        self.assertEqual(dict(graph.edges), dict(before.edges))
        self.assertEqual(snack.smiles_to_snack('CC', attributed=False), 'v1_v')
        self.assertEqual(snack.mol_to_snack(Chem.MolFromSmiles('CC'), attributed=False), 'v1_v')

    def test_mixed_weighted_graph_matrix(self):
        for seed in range(20):
            graph = nx.gnp_random_graph(8, 0.4, seed=seed)
            for i, edge in enumerate(graph.edges):
                if i % 3:
                    graph.edges[edge]['weight'] = (i - 3) / 2
                if i % 2:
                    graph.edges[edge]['edge_type'] = 'complex3'
            for node_first in (False, True):
                for ascending in (False, True):
                    snack.config.set(node_first, ascending)
                    for ordering in (None, 'bfs', 'cm', 'uniform', 'neighbor'):
                        with self.subTest(seed=seed, node_first=node_first,
                                          ascending=ascending, ordering=ordering):
                            encoded = snack.graph_to_snack(graph, ordering)
                            self.assert_graph_equal(graph, snack.snack_to_graph(encoded))
                            for pattern in ('token', 'graph', 'block'):
                                self.assertEqual(''.join(snack.split(encoded, pattern)), encoded)

    def test_invalid_weights(self):
        for weight in (float('nan'), float('inf'), -float('inf'), True, '3', None, 3j):
            with self.subTest(weight=weight):
                graph = nx.Graph()
                graph.add_edge(0, 1, weight=weight)
                with self.assertRaisesRegex(ValueError, 'weights'):
                    snack.graph_to_snack(graph)
                # Explicitly dropping attributes also drops unsupported weights.
                self.assertEqual(snack.graph_to_snack(graph, attributed=False), 'v1_v')

    def test_invalid_grammar(self):
        malformed = ('[C]]', '[C', '[]', 'v0_v', 'v01_v', 'v1_', 'v1__v', 'v1-v=',
                     'v1_1_v', 'v2_v', 'v1v', 'v1-v2', 'v1(dative', 'v1()v',
                     'v1dativev', 'v1(weight=nan)v', 'v1(weight=inf)v',
                     'v1(weight=1e999)v', 'v1(type=dative)v', 'v1(bad%escape)v',
                     'v1(outer(inner))v', 'v 1_v')
        for encoded in malformed:
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                snack.snack_to_graph(encoded)
        snack.config.set(node_first=True)
        for encoded in ('1_v', 'v1_', 'vv1_1_', 'vv2_', 'vv1'):
            with self.subTest(node_first=True, encoded=encoded), self.assertRaises(ValueError):
                snack.snack_to_graph(encoded)

    def test_validation_survives_python_optimization(self):
        source = '''import snack
for s in ("v0_v", "v2_v", "v1_", "v1-v=", "[C]]"):
    try: snack.snack_to_graph(s)
    except ValueError: continue
    raise RuntimeError("accepted malformed SNACK: " + s)
'''
        result = subprocess.run([sys.executable, '-B', '-O', '-c', source],
                                cwd=PACKAGE_DIR.parent, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_large_automorphism_integer(self):
        # 25! cannot be represented exactly as a double-precision number.
        self.assertEqual(snack.count_automorphisms(nx.complete_graph(25)), math.factorial(25))
        self.assertEqual(snack.count_automorphisms(nx.empty_graph(25)), math.factorial(25))

    def test_weight_sensitive_automorphisms(self):
        graph = nx.path_graph(3)
        graph.edges[0, 1]['weight'] = 1
        graph.edges[1, 2]['weight'] = 2
        self.assertEqual(snack.count_automorphisms(graph), 1)

    def test_partial_attributes_automorphism_matrix(self):
        for seed in range(75):
            rng = random.Random(seed)
            graph = nx.gnp_random_graph(seed % 7, (seed % 5) / 4, seed=seed)
            for node in graph:
                if rng.random() < 0.7:
                    graph.nodes[node]['node_type'] = rng.choice(['[C]', '[N]'])
            for edge in graph.edges:
                if rng.random() < 0.7:
                    graph.edges[edge]['edge_type'] = rng.choice(['-', '=', 'dative'])
                if rng.random() < 0.4:
                    graph.edges[edge]['weight'] = rng.choice([0, 1, 2])
            with self.subTest(seed=seed):
                self.assertEqual(snack.count_automorphisms(graph), self.exact_automorphisms(graph))

class TestMoleculeNotation(RegressionCase):
    def assert_molecule_equal(self, expected, recovered):
        self.assertEqual(Chem.MolToSmiles(expected), Chem.MolToSmiles(recovered))

    @staticmethod
    def without_stereo(mol):
        result = Chem.Mol(mol)
        Chem.RemoveStereochemistry(result)
        return result

    def test_stereochemistry_omitted_matrix(self):
        smiles_cases = ('N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O',
                        'C[C@H](O)[C@@H](C)O', 'N[C@@](F)(Cl)Br',
                        'C[C@H]1CCCCO1', 'F[C@H]1CCC[C@@H](Cl)C1',
                        'F/C=C/F', 'F/C=C\\F', 'F/C(Cl)=C(/Br)I',
                        'C/C=C/C=C\\C', 'C[C@H](O)/C=C/[C@@H](F)Cl',
                        '[2H][C@](F)(Cl)Br', '[NH3+][C@@H](C)C(=O)[O-]')
        for smiles in smiles_cases:
            original = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(original, smiles)
            expected = self.without_stereo(original)
            for seed in range(3):
                order = list(range(original.GetNumAtoms()))
                random.Random(seed).shuffle(order)
                mol = Chem.RenumberAtoms(original, order)
                for ordering in (None, 'bfs', 'cm', 'uniform', 'neighbor'):
                    for node_first in (False, True):
                        for ascending in (False, True):
                            snack.config.set(node_first, ascending)
                            with self.subTest(smiles=smiles, seed=seed, ordering=ordering,
                                              node_first=node_first, ascending=ascending):
                                encoded = snack.mol_to_snack(mol, ordering=ordering)
                                self.assertNotIn('@', encoded)
                                self.assert_molecule_equal(expected, snack.snack_to_mol(encoded))
                                graph = snack.mol_to_graph(mol)
                                self.assert_molecule_equal(expected, snack.graph_to_mol(graph))
                                self.assert_molecule_equal(original, mol)

    def test_stereoisomers_have_the_same_notation(self):
        for left_smiles, right_smiles in (('F/C=C/F', 'F/C=C\\F'),
                                         ('N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O')):
            for hydrogens in (False, True):
                for seed in range(5):
                    random.seed(seed)
                    left = snack.smiles_to_snack(left_smiles, use_hydrogens=hydrogens)
                    random.seed(seed)
                    right = snack.smiles_to_snack(right_smiles, use_hydrogens=hydrogens)
                    self.assertEqual(left, right)

    def test_molecule_identity_without_stereo_survives_relabeling(self):
        for smiles in ('N[C@@H](C)C(=O)O', 'F/C(Cl)=C(/Br)I'):
            expected = Chem.MolToSmiles(self.without_stereo(Chem.MolFromSmiles(smiles)))
            graph = snack.smiles_to_graph(smiles)
            graph = snack.relabel_nodes(graph, list(reversed(list(graph))))
            self.assertEqual(snack.graph_to_smiles(graph), expected)
            encoded = snack.graph_to_snack(graph)
            self.assertEqual(snack.snack_to_smiles(encoded), expected)

    def test_unsupported_atom_modifiers_rejected(self):
        for text in ('[C@H]', '[C@@H]', '[CH3;r=1]', '[CH0;r=4]'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    snack_mol.string_to_atom(text)
                with self.assertRaises(ValueError):
                    snack.snack_to_mol(text)

    def test_directional_dative_bonds_rejected(self):
        for smiles in ('[NH3]->[Cu+2]', '[Cu+2]<-[NH3]', 'N->[Pt+2]<-N'):
            original = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(original)
            for convert, value in ((snack.mol_to_graph, original), (snack.mol_to_snack, original),
                                   (snack.smiles_to_graph, smiles), (snack.smiles_to_snack, smiles)):
                with self.subTest(smiles=smiles, convert=convert.__name__):
                    with self.assertRaisesRegex(ValueError, 'Unsupported bond type'):
                        convert(value)
        for bond_type in (Chem.BondType.DATIVE, Chem.BondType.DATIVEONE,
                          Chem.BondType.DATIVEL, Chem.BondType.DATIVER):
            mol = Chem.MolFromSmiles('[NH3]->[Cu+2]')
            mol.GetBondWithIdx(0).SetBondType(bond_type)
            with self.subTest(bond_type=bond_type):
                with self.assertRaisesRegex(ValueError, 'Unsupported bond type'):
                    snack.mol_to_graph(mol)

    def test_order_dependent_molecular_edge_types_rejected(self):
        symbols = ['=cis', '=trans', '=unknown']
        for base in ('dative', 'dativeone', 'dativel', 'dativer'):
            symbols.extend((base, base + '-reverse'))
        for symbol in symbols:
            text = f'[C]1({symbol})[C]'
            with self.subTest(symbol=symbol):
                graph = snack.snack_to_graph(text)
                with self.assertRaisesRegex(ValueError, 'Unsupported molecular edge type'):
                    snack.graph_to_mol(graph)
                with self.assertRaisesRegex(ValueError, 'Unsupported molecular edge type'):
                    snack.snack_to_smiles(text)

    def test_unspecified_double_bond_stereo_omitted(self):
        mol = Chem.MolFromSmiles('FC=CF')
        bond = mol.GetBondBetweenAtoms(1, 2)
        bond.SetStereo(Chem.BondStereo.STEREOANY)
        text = snack.mol_to_snack(mol)
        self.assertEqual(text, '[F]1-[C]1=[C]1-[F]')
        self.assertEqual(snack.snack_to_smiles(text), 'FC=CF')
        self.assertEqual(bond.GetStereo(), Chem.BondStereo.STEREOANY)

    def test_radicals_isotopes_wildcards_and_hydrogens(self):
        for smiles in ('[CH3]', '[CH2]', '[C]', '[O]', '[H]', '[13CH4]', '[2H]O[2H]',
                       '*C', '*', '[NH4+]', '[C:1](F)(F)(F)F', 'c1cc[nH]c1'):
            for use_hydrogens in (False, True):
                for kekulize in (False, True):
                    with self.subTest(smiles=smiles, hydrogens=use_hydrogens, kekulize=kekulize):
                        original = Chem.MolFromSmiles(smiles)
                        encoded = snack.smiles_to_snack(smiles, use_hydrogens=use_hydrogens,
                                                        kekulize=kekulize)
                        if use_hydrogens:
                            self.assert_molecule_equal(original, snack.snack_to_mol(encoded))
                        else:
                            for token in snack.split(encoded):
                                if snack_graph.token_type(token) == 'node_type':
                                    self.assertIsNone(snack_mol.RE_ATOM_TYPE.fullmatch(
                                        token[1:-1])['hcount'])

    def test_invalid_molecule_and_atom_inputs(self):
        from rdkit import rdBase
        with rdBase.BlockLogs():
            for text in ('C', '[C]]', '[[C]', '[Qq]', '[C@SP1]', '[13]', '[C;r=-1]'):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    snack_mol.string_to_atom(text)
            for kekulize in (False, True):
                for invalid in ('bad_smiles', 'C(C)(C)(C)(C)C', None):
                    with self.subTest(invalid=invalid, kekulize=kekulize), self.assertRaises(ValueError):
                        snack.smiles_to_graph(invalid, kekulize=kekulize)

    def test_other_atomic_stereo_is_omitted(self):
        for smiles in ('Cl[Pt@SP1](Cl)(N)N', 'F[C@H](Cl)Br |o1:1|'):
            original = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(original)
            encoded = snack.mol_to_snack(original, use_hydrogens=True)
            self.assertNotIn('@', encoded)
            self.assert_molecule_equal(self.without_stereo(original), snack.snack_to_mol(encoded))


class TestDatasetLoader(unittest.TestCase):
    def test_all_bundled_splits_match_legacy(self):
        import hashlib
        import json
        # Frozen against generic.utils.train_val_test_split, before migration.
        # Fingerprints cover every string, duplicate, split and list position.
        expected = LEGACY_DATASET_SPLITS
        for name, (counts, digest) in expected.items():
            with self.subTest(dataset=name):
                data = snack.load_dataset(name)
                self.assertIsInstance(data, dict)
                self.assertEqual(list(data), ['train', 'val', 'test'])
                self.assertEqual(tuple(map(len, data.values())), counts)
                actual = hashlib.sha256(json.dumps(data, sort_keys=True,
                                                   separators=(',', ':')).encode()).hexdigest()
                self.assertEqual(actual, digest)
                with (PACKAGE_DIR / 'data' / f'{name}.csv').open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                from collections import Counter
                self.assertEqual(Counter(s for split in data.values() for s in split),
                                 Counter(row['snack'] for row in rows))

    def test_returns_independent_lists(self):
        first = snack.load_dataset('planar')
        original = first['train'][:]
        first['train'].clear()
        self.assertEqual(snack.load_dataset('planar')['train'], original)

    def test_unknown_and_invalid_names(self):
        for name in ('../planar', '/tmp/planar', 'planar.csv', '', None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                snack.load_dataset(name)
        with self.assertRaises(FileNotFoundError):
            snack.load_dataset('missing_dataset')

    def test_split_index_preserves_order_and_duplicate_strings(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'data' / 'example.csv').write_text(
                'snack,split,split_index\n'
                'v,train,1\nvv,val,0\nv1_v,train,0\nv,train,2\n')
            with patch('snack.dataset.files', return_value=root):
                self.assertEqual(snack.load_dataset('example'),
                                 {'train': ['v1_v', 'v', 'v'], 'val': ['vv'], 'test': []})

    def test_malformed_split_metadata_is_rejected(self):
        import tempfile
        from unittest.mock import patch
        cases = (
            'snack,n_node\nv,1\n',
            'snack,split,split_index\nv,validation,0\n',
            'snack,split,split_index\nv,train,no_index\n',
            'snack,split,split_index\nv,train,-1\n',
            'snack,split,split_index\nv,train,1\n',
            'snack,split,split_index\nv,train,0\nvv,train,0\n',
            'snack,split,split_index\nv,train\n',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            for content in cases:
                with self.subTest(content=content):
                    (root / 'data' / 'example.csv').write_text(content)
                    with patch('snack.dataset.files', return_value=root), self.assertRaises(ValueError):
                        snack.load_dataset('example')

    def test_relocated_zip_resources_without_training_dependencies(self):
        import tempfile
        import zipfile
        # Exercise an actual relocated package/archive, with an unrelated CWD.
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'snack.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                for path in list(PACKAGE_DIR.glob('*.py')) + list((PACKAGE_DIR / 'data').glob('*.csv')):
                    if path.name != 'test.py':
                        bundle.write(path, 'snack/' + path.relative_to(PACKAGE_DIR).as_posix())
            code = '''import importlib.abc
import sys
class RejectTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'pandas', 'sklearn'}:
            raise ImportError('Training dependency forbidden: ' + fullname)
sys.meta_path.insert(0, RejectTrainingDependencies())
sys.path.insert(0, sys.argv[1])
import snack
assert snack.__file__.startswith(sys.argv[1]), snack.__file__
data = snack.load_dataset('planar')
assert {key: len(value) for key, value in data.items()} == {'train': 128, 'val': 32, 'test': 40}
assert not {'torch', 'pandas', 'sklearn'} & sys.modules.keys()
'''
            result = subprocess.run([sys.executable, '-B', '-c', code, str(archive)],
                                    cwd=directory, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)


LEGACY_DATASET_SPLITS = {
    'community_small': ((70, 10, 20),
        '014c72e1389cd366027490cd178cb75a03d6783ce75d5ca40b2a79bc0e4c673b'),
    'ego': ((529, 76, 152),
        'b3521842398fa60b40808ec336ebbe1c1920ccf2638794c876c6b60415fc46dc'),
    'ego_small': ((140, 20, 40),
        'b3669018cf30ca35428ae64422ae148b84a61dbd4c4241747e3a344f2a37cff5'),
    'enzymes': ((410, 59, 118),
        '3cf18efafee33c45ab6aa865c7f6839507044063de9255b56a1fb9ac2f57950f'),
    'grid': ((70, 10, 20),
        'aa8169b317aa32fc2d2cb3483eeebb1a819baa44d56a517d2dd99d68707d995e'),
    'grid_small': ((70, 10, 20),
        '9a3474f0cc81aca1b1c6b1f15aad7502d8ec24dfc82e12f4a6571dbb9638de92'),
    'lobster': ((69, 20, 11),
        '8a1a21e9c24609961c9a61d6a7b916ed3f61af3a4062e26237cfa3fae41b4f17'),
    'planar': ((128, 32, 40),
        'f6e521350f677b0ab2539dcdca157161292985b7ef5a7871a312d5fd2cbd98d2'),
    'sbm': ((128, 32, 40),
        'bd53e5ce6dbf7530090ede991898a93db4fbf6839abb1468f98675c1bd8fc744'),
}


if __name__ == '__main__':
    unittest.main()
