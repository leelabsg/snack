"""Stereochemistry ({@} {@@} {c} {t}) and use_hydrogens='auto' tests."""
import itertools
import random
import unittest

import networkx as nx
from rdkit import Chem, rdBase

import snack
from snack import SnackTokenizer
from snack import graph as snack_graph
from snack.mol import _stereo_perception

EDGE_CASES = [
    'C[C@H](N)C(=O)O', 'C[C@@H](N)C(=O)O',          # enantiomers
    'F/C=C/F', 'F/C=C\\F',                           # trans / cis
    'C[S@](=O)CC', 'C[P@](=O)(O)CC',                 # 3-coordinate centers
    'C[N@+]1(CC)CCCC1',                              # charged center
    'F[C@](Cl)(Br)I',                                # four explicit ligands
    'O[C@@H](C)[C@H](C)O', 'O[C@@H](C)[C@@H](C)O',   # meso / chiral
    'O[C@H]1CC[C@@H](O)CC1',                         # ring cis/trans
    'CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O',               # several centers
    'C[C@H](O)/C=C/[C@@H](F)Cl', 'C/C=C/C=C\\C',     # mixed, conjugated
    'C[C@H](O)CC(C)O', 'CC(F)=CC',                   # partially / un-specified
    'C1=CCCCC1', 'C1=CCCCCCCC/C=C/1',                # small vs large ring
    'O=C1/C(=C\\c2cccs2)COC/C1=C\\c1cccs1',          # conjugated exocyclic
    'C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1',  # ZINC
]
ORDERINGS = ('bfs', 'uniform', 'cm', 'neighbor')
CONFIGS = list(itertools.product((False, True), repeat=2))


def reference(smiles):
    with _stereo_perception():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


class StereoCase(unittest.TestCase):
    def setUp(self):
        self.state = random.getstate()
        snack.config.set()

    def tearDown(self):
        random.setstate(self.state)
        snack.config.set()


class TestStereoNotation(StereoCase):
    def test_golden_strings(self):
        golden = {
            'C[C@H](N)C(=O)O': '[C]1-[C]{@}1-[N]2-[C]1=[O]2-[O]',
            'C[C@@H](N)C(=O)O': '[C]1-[C]{@@}1-[N]2-[C]1=[O]2-[O]',
            'F/C=C/F': '[F]1-[C]1={t}[C]1-[F]',
            'F/C=C\\F': '[F]1-[C]1={c}[C]1-[F]',
            'C[S@](=O)CC': '[C]1-[S]{@}1=[O]2-[C]1-[C]',
        }
        for smiles, expected in golden.items():
            with self.subTest(smiles=smiles):
                self.assertEqual(snack.smiles_to_snack(smiles, ordering=None, stereo=True), expected)
                self.assertEqual(snack.snack_to_smiles(expected, stereo='strict'), reference(smiles))

    def test_stereo_is_opt_in(self):
        for smiles in EDGE_CASES:
            with self.subTest(smiles=smiles):
                random.seed(0)
                text = snack.smiles_to_snack(smiles)
                self.assertNotIn('{', text)
                self.assertNotIn('tetra', str(snack.smiles_to_graph(smiles).nodes(data=True)))

    def test_round_trip(self):
        for smiles in EDGE_CASES:
            for kekulize in (True, False):
                with self.subTest(smiles=smiles, kekulize=kekulize):
                    random.seed(0)
                    text = snack.smiles_to_snack(smiles, kekulize=kekulize, stereo=True)
                    self.assertEqual(snack.snack_to_smiles(text, stereo='strict'), reference(smiles))
                    self.assertTrue(snack.is_canonical_snack(text))

    def test_ordering_invariance(self):
        for smiles in EDGE_CASES:
            expected = reference(smiles)
            with self.subTest(smiles=smiles):
                decoded = set()
                for (node_first, ascending), ordering, seed in itertools.product(
                        CONFIGS, ORDERINGS, range(8)):
                    snack.config.set(node_first, ascending)
                    random.seed(seed)
                    text = snack.smiles_to_snack(smiles, ordering=ordering, stereo=True)
                    decoded.add(snack.snack_to_smiles(text, stereo='strict'))
                self.assertEqual(decoded, {expected})

    def test_annotations_do_not_depend_on_layout(self):
        # Neighbor order is defined by node index, so the same node order gives
        # the same annotations in every node_first/ascending layout.
        for smiles in EDGE_CASES:
            graph = snack.smiles_to_graph(smiles, stereo=True)
            for seed in range(5):
                random.seed(seed)
                order = snack_graph.graph_order(graph, 'uniform')
                values = set()
                for node_first, ascending in CONFIGS:
                    snack.config.set(node_first, ascending)
                    values.add(tuple(snack.filter(snack.graph_to_snack(graph, order), 'stereo')))
                with self.subTest(smiles=smiles, seed=seed):
                    self.assertEqual(len(values), 1)

    def test_swapping_two_ligands_gives_the_mirror_image(self):
        graph = snack.smiles_to_graph('C[C@H](N)C(=O)O', stereo=True)
        ligands, sign = graph.nodes[1]['tetra']
        mirrored = graph.copy()
        mirrored.nodes[1]['tetra'] = ((ligands[1], ligands[0]) + tuple(ligands[2:]), sign)
        self.assertEqual(snack.graph_to_smiles(mirrored), reference('C[C@@H](N)C(=O)O'))
        graph = snack.smiles_to_graph('F/C=C/F', stereo=True)
        self.assertEqual(graph.edges[1, 2]['cis_trans'], ((1, 0), (2, 3), 't'))
        graph.edges[1, 2]['cis_trans'] = ((1, 0), (2, 3), 'c')
        self.assertEqual(snack.graph_to_smiles(graph), reference('F/C=C\\F'))

    def test_perm_parity(self):
        parity = snack_graph.perm_parity
        self.assertEqual(parity('abcd', 'abcd'), 1)
        self.assertEqual(parity('abcd', 'bacd'), -1)
        self.assertEqual(parity('abcd', 'badc'), 1)
        self.assertEqual(parity([1, None, 2, 3], [None, 1, 2, 3]), -1)
        for bad in (('abc', 'abd'), ('aab', 'aba'), ('abc', 'ab')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parity(*bad)

    def test_decoding_modes(self):
        # A center with two methyl groups is not stereogenic.
        text = '[C]1-[C]{@}1-[C]2-[C]'
        self.assertEqual(snack.snack_to_smiles(text), 'CC(C)C')
        self.assertFalse(snack.is_canonical_snack(text))
        with self.assertRaisesRegex(ValueError, 'non-stereogenic'):
            snack.snack_to_smiles(text, stereo='strict')
        # Structurally impossible annotations: strict raises, clean drops.
        for text in ('[C]{@}1-[C]', '[C]1={c}[C]'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    snack.snack_to_mol(text, stereo='strict')
                self.assertNotIn('@', snack.snack_to_smiles(text))
        with self.assertRaises(ValueError):
            snack.snack_to_mol('[C]', stereo='lenient')

    def test_rdkit_perception_setting_is_pinned_and_restored(self):
        results = {}
        original = Chem.GetUseLegacyStereoPerception()
        try:
            for legacy in (True, False):
                Chem.SetUseLegacyStereoPerception(legacy)
                outputs = []
                for smiles in EDGE_CASES:
                    random.seed(0)
                    text = snack.smiles_to_snack(smiles, stereo=True)
                    outputs.append((text, snack.snack_to_smiles(text)))
                self.assertEqual(Chem.GetUseLegacyStereoPerception(), legacy)
                results[legacy] = outputs
        finally:
            Chem.SetUseLegacyStereoPerception(original)
        self.assertEqual(results[True], results[False])

    def test_stereo_aware_automorphisms(self):
        # (S,R) meso loses the half-swap symmetry; (S,S) keeps it.
        expected = {'O[C@@H](C)[C@@H](C)O': 1, 'O[C@@H](C)[C@H](C)O': 2, 'OC(C)C(C)O': 2}
        for smiles, count in expected.items():
            with self.subTest(smiles=smiles):
                graph = snack.smiles_to_graph(smiles, stereo=True)
                self.assertEqual(snack.count_automorphisms(graph), 2)
                self.assertEqual(snack.count_automorphisms(graph, stereo=True), count)
        trans = snack.smiles_to_graph('F/C=C/F', stereo=True)
        self.assertEqual(snack.count_automorphisms(trans, stereo=True), 2)
        # Decoded graphs carry the same annotations.
        meso = snack.snack_to_graph(snack.smiles_to_snack('O[C@@H](C)[C@@H](C)O', stereo=True))
        self.assertEqual(snack.count_automorphisms(meso, stereo=True), 1)
        with self.assertRaisesRegex(ValueError, 'max_automorphisms'):
            snack.count_automorphisms(snack.smiles_to_graph('O[C@@H](C)[C@H](C)O', stereo=True),
                                      stereo=True, max_automorphisms=1)

    def test_automorphisms_match_rdkit_chiral_matches(self):
        # RDKit's chiral substructure matching covers tetrahedral stereo only.
        for smiles in ('O[C@H]1CC[C@@H](O)CC1', 'O[C@H]1CC[C@H](O)CC1', 'C[C@H]1CC[C@@H](C)CC1',
                       'C[C@H](O)CC(C)O', 'CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O', 'C[C@@H](F)[C@H](C)F'):
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                graph = snack.mol_to_graph(mol, stereo=True)
                self.assertEqual(snack.count_automorphisms(graph, stereo=True),
                                 len(mol.GetSubstructMatches(mol, uniquify=False, useChirality=True)))


class TestGraphStereo(StereoCase):
    def graph(self):
        graph = nx.Graph()
        for node, label in zip('xabc', ('[C]', '[F]', '[Cl]', '[Br]')):
            graph.add_node(node, node_type=label)
        for node in 'abc':
            graph.add_edge('x', node, edge_type='-')
        graph.nodes['x']['tetra'] = (('a', None, 'b', 'c'), '@')
        return graph

    def test_relabel_and_explicit_orderings(self):
        graph = self.graph()
        relabeled = snack.relabel_nodes(graph, ['b', 'x', 'c', 'a'])
        self.assertEqual(relabeled.nodes[1]['tetra'], ((3, None, 0, 2), '@'))
        self.assertEqual(graph.nodes['x']['tetra'], (('a', None, 'b', 'c'), '@'))
        smiles = snack.graph_to_smiles(graph)
        for order in itertools.permutations('xabc'):
            with self.subTest(order=order):
                text = snack.graph_to_snack(graph, list(order))
                self.assertEqual(snack.snack_to_smiles(text), smiles)
                self.assertEqual(snack.graph_to_smiles(snack.relabel_nodes(graph, list(order))), smiles)
        with self.assertRaises(ValueError):
            snack.graph_to_snack(graph, ['x', 'a', 'b'])

    def test_attribute_validation(self):
        for value in ((('a', 'b', 'c'), '@'), (('a', 'b', 'c', 'c'), '@'), (('a', None, 'b', 'c'), 'R'), 'bad'):
            graph = self.graph()
            graph.nodes['x']['tetra'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Invalid tetra'):
                snack.graph_to_snack(graph)
        graph = self.graph()
        graph.edges['x', 'a']['cis_trans'] = (('x', 'b'), ('a', 'c'), 'c')
        with self.assertRaisesRegex(ValueError, 'Invalid cis_trans'):
            snack.graph_to_snack(graph)
        self.assertNotIn('{', snack.graph_to_snack(self.graph(), attributed=False))

    def test_grammar(self):
        text = '[F]1-[C]{@}1-[Cl]2-[Br]'
        self.assertEqual(snack.split(text), ['[F]', '1', '-', '[C]', '{@}', '1', '-', '[Cl]', '2', '-', '[Br]'])
        self.assertEqual(snack.split(text, 'graph'), ['[F]', '1-', '[C]{@}', '1-', '[Cl]', '2-', '[Br]'])
        self.assertEqual(snack.split(text, 'block'), ['[F]', '1-[C]{@}', '1-[Cl]', '2-[Br]'])
        self.assertEqual(snack.filter('[F]1-[C]1={t}[C]1-[F]', 'stereo'), ['{t}'])
        self.assertEqual(snack.token_type('{@@}'), 'stereo')
        for bad in ('{@}[C]', '[C]{@}{@@}', '[C]{c}', '[C]1-{@}[C]', '[C]{x}', '[C]1{t}[C]'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                snack.snack_to_graph(bad)
        snack.config.set(node_first=True)
        self.assertEqual(snack.split('[F][C]{@}1-[Cl]2-[Br]2-', 'block'),
                         ['[F]', '[C]{@}1-', '[Cl]2-', '[Br]2-'])


class TestAutoHydrogens(StereoCase):
    def test_auto_hydrogens_only_where_needed(self):
        cases = {('[CH3]', True): '[CH3]', ('C', True): '[C]', ('C[O]', True): '[C]1-[OH0]',
                 ('CCO', True): '[C]1-[C]1-[O]', ('C[NH3+]', True): '[C]1-[N+]',
                 ('c1cc[nH]c1', False): '[C]1a[C]1a[C]1a[NH]4a1a[C]'}
        for (smiles, kekulize), expected in cases.items():
            with self.subTest(smiles=smiles, kekulize=kekulize):
                text = snack.smiles_to_snack(smiles, ordering=None, kekulize=kekulize, use_hydrogens='auto')
                self.assertEqual(text, expected)
                self.assertEqual(snack.snack_to_smiles(text), reference(smiles))
                self.assertTrue(snack.is_canonical_snack(text))
        self.assertFalse(snack.is_canonical_snack('[CH4]'))
        self.assertFalse(snack.is_canonical_snack('[C]1-[OH]'))

    def test_auto_matches_default_without_radicals(self):
        for smiles in EDGE_CASES:
            with self.subTest(smiles=smiles):
                self.assertEqual(snack.smiles_to_snack(smiles, ordering=None, use_hydrogens='auto'),
                                 snack.smiles_to_snack(smiles, ordering=None))


class TestStereoTokenizer(StereoCase):
    def allowed(self, tokenizer, text):
        return {tokenizer.tokens[i] for i in tokenizer.allowed_next_ids(tokenizer.encode(text)[:-1])}

    def test_vocabulary_ids_are_stable(self):
        for preset in ('zinc250k', 'moses', 'guacamol'):
            with self.subTest(preset=preset):
                base, stereo = SnackTokenizer(preset), SnackTokenizer(preset, stereo=True)
                self.assertEqual(stereo.tokens[:len(base)], base.tokens)
                self.assertEqual(stereo.tokens[len(base):], ('{@}', '{@@}', '{c}', '{t}'))
                self.assertEqual(snack.basic_vocabulary(preset, stereo=True)[-4:],
                                 ['{@}', '{@@}', '{c}', '{t}'])
        with self.assertRaises(ValueError):
            SnackTokenizer('graph', stereo=True)
        with self.assertRaises(ValueError):
            SnackTokenizer('moses', hydrogens=True)
        with self.assertRaises(ValueError):
            SnackTokenizer('moses').encode('[C]{@}')

    def test_masks(self):
        tokenizer = SnackTokenizer('zinc250k', stereo=True)
        self.assertTrue({'{@}', '{@@}'} <= self.allowed(tokenizer, '[C]'))
        self.assertTrue({'{@}', '{@@}'}.isdisjoint(self.allowed(tokenizer, '[F]')))
        self.assertTrue({'{@}', '{@@}'}.isdisjoint(self.allowed(tokenizer, '[O]')))
        self.assertTrue({'{c}', '{t}'} <= self.allowed(tokenizer, '[C]1='))
        self.assertTrue({'{c}', '{t}'}.isdisjoint(self.allowed(tokenizer, '[C]1-')))
        self.assertTrue(set(tokenizer.stereo_types).isdisjoint(self.allowed(tokenizer, '[C]{@}')))
        self.assertTrue(set(tokenizer.stereo_types).isdisjoint(self.allowed(tokenizer, '[C]1')))

    def test_encoded_data_passes_masks(self):
        for node_first, ascending in CONFIGS:
            snack.config.set(node_first, ascending)
            tokenizer = SnackTokenizer('zinc250k', ordering='bfs', stereo=True)
            for smiles, seed in itertools.product(EDGE_CASES, range(3)):
                random.seed(seed)
                text = snack.smiles_to_snack(smiles, stereo=True)
                with self.subTest(smiles=smiles, node_first=node_first, ascending=ascending):
                    ids = tokenizer.encode(text, validate=True)
                    self.assertEqual(tokenizer.decode(ids), text)
                    masks = tokenizer.get_logit_mask(ids)
                    self.assertTrue(all(masks[i][target] for i, target in enumerate(ids[1:])))
        tokenizer = SnackTokenizer('guacamol', ordering='bfs', stereo=True, hydrogens='auto')
        tokenizer.encode(snack.smiles_to_snack('[CH2]C[C@H](F)Cl', stereo=True, use_hydrogens='auto'),
                         validate=True)

    def test_incremental_and_prefix_masks_agree(self):
        tokenizer = SnackTokenizer('moses', ordering='bfs', stereo=True)
        ids = tokenizer.encode('[C]1-[C]{@}1-[N]2-[C]1={t}[C]1-[F]', validate=True)
        state = tokenizer.new_state()
        for i, index in enumerate(ids):
            tokenizer.step(state, index)
            self.assertEqual(tokenizer.allowed_token_ids(state), tokenizer.allowed_next_ids(ids[:i + 1]))

    def generate(self, tokenizer, seed):
        rng = random.Random(seed)
        state, ids = tokenizer.new_state(), []
        for _ in range(200):
            allowed = tokenizer.allowed_token_ids(state)
            self.assertTrue(allowed, 'Generation reached a dead end')
            stereo = [i for i in allowed if tokenizer.tokens[i] in tokenizer.stereo_types]
            # Prefer stereo tokens to exercise annotations that decoding drops.
            index = rng.choice(stereo) if stereo and rng.random() < 0.7 else rng.choice(allowed)
            tokenizer.step(state, index)
            ids.append(index)
            if index == tokenizer.eos_id:
                return tokenizer.decode(ids)
        self.fail('Generation did not terminate')

    def test_generation_is_always_valid(self):
        with rdBase.BlockLogs():
            for (node_first, ascending), bfs in itertools.product(CONFIGS, (False, True)):
                snack.config.set(node_first, ascending)
                tokenizer = SnackTokenizer('zinc250k', max_nodes=7, ordering='bfs' if bfs else None,
                                           stereo=True, node_first=node_first, ascending=ascending)
                for seed in range(15):
                    with self.subTest(node_first=node_first, ascending=ascending, bfs=bfs, seed=seed):
                        mol = snack.snack_to_mol(self.generate(tokenizer, seed))
                        self.assertIsNotNone(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))


if __name__ == '__main__':
    unittest.main()
