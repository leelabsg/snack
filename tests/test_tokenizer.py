"""Regression tests for the torch-free downstream tokenizer."""
import itertools
from pathlib import Path
import random
import subprocess
import sys
import unittest

import networkx as nx
from rdkit import Chem, rdBase

import snack
from snack.tokenizer import SnackTokenizer, basic_vocabulary, capacity, elements, attribute_tokens, UNLIMITED

ORGANIC = ('B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I')
MOLECULES = ['C[C@H](N)C(=O)O', 'F/C=C/F', 'F/C=C\\F', '[CH2]C', 'C[N+](C)(C)C', '[O-]C(=O)C',
             'c1cc[nH]c1', '[Cu]I', 'O=[PH3]', 'C[S@](=O)CC', '[Fe+3]', 'C[Si](C)(C)C',
             '[H][H]', '[NH4+]', 'CS(C)(=O)=O', '[O][O]', 'O=[N+]([O-])c1ccccc1', 'F[B-](F)(F)F',
             'C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1', 'C[C@H](O)/C=C/[C@@H](F)Cl', '[C]',
             'CCCC[Sn](CCCC)(CCCC)c1nc2ccccc2nc1C', '[O-][n+]1ccccc1', 'C[CH-]C', '[BH4-]', 'F[Kr]F']


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.config = (snack.config.NODE_FIRST, snack.config.ASCENDING)
        snack.config.set()

    def tearDown(self):
        snack.config.set(*self.config)

    @staticmethod
    def allowed(tokenizer, text):
        prefix = [tokenizer.bos_id] + tokenizer.encode(text, add_special_tokens=False)
        return {tokenizer.tokens[i] for i in tokenizer.allowed_next_ids(prefix)}

    def attributes_allowed(self, tokenizer, text):
        return {token for token in self.allowed(tokenizer, text) if token.startswith('(')}

    # ----- vocabulary ----------------------------------------------------------
    def test_default_vocabulary_is_general(self):
        isotope_tokens = set(attribute_tokens(('isotope',)))
        self.assertEqual(len(isotope_tokens), 294)
        tokenizer = SnackTokenizer()
        self.assertEqual(tuple(basic_vocabulary()), tokenizer.tokens)
        self.assertEqual(len(tokenizer), 575)
        self.assertEqual((tokenizer.max_nodes, tokenizer.max_distance), (128, 127))
        # The 1,177-token vocabulary of the MOSES runs, before the isotope kind
        # and the smaller node limit, is still reachable.
        before = SnackTokenizer(max_nodes=1024,
                                attributes=('charge', 'hydrogens', 'chirality', 'cis_trans'))
        self.assertEqual(len(before), 1177)
        self.assertEqual(set(tokenizer.tokens) - set(before.tokens) - isotope_tokens, set())
        self.assertEqual(tokenizer.tokens[:3], ('[pad]', '[bos]', '[eos]'))
        self.assertEqual(len(elements()), 119)
        self.assertEqual(elements()[:3], ('H', 'He', 'Li'))
        self.assertEqual(elements()[-2:], ('Og', '*'))
        for token in ('H', 'C', 'Cl', 'Og', '*', '-', '=', '#', '(@)', '(@@)', '(H0)', '(H)', '(H8)', '(+)',
                      '(+2)', '(+8)', '(-)', '(-8)', '(c)', '(t)', '1', str(tokenizer.max_distance)):
            self.assertIn(token, tokenizer.token_to_id)
        self.assertEqual(tokenizer.max_nodes, 128)
        for token in ('(+9)', '(-9)', '(H9)', '(.)', '(.2)', '0', '128', '1024', 'o', '[C]', 'c',
                      '(0)', '(295)', '(13C)', 'u'):
            self.assertNotIn(token, tokenizer.token_to_id)
        self.assertEqual(tokenizer.node_types, elements())
        self.assertEqual(tokenizer.edge_types, ('-', '=', '#'))
        # Attribute tokens are the last block, grouped by kind in the canonical order.
        kinds = [snack.graph.parse_attribute(token)[0] for token in tokenizer.attribute_types]
        self.assertEqual(kinds, sorted(kinds, key=list(snack.graph.ATTRIBUTES).index))
        self.assertEqual(tuple(dict.fromkeys(kinds)),
                         ('isotope', 'charge', 'hydrogens', 'chirality', 'cis_trans'))
        self.assertEqual(tokenizer.token_to_id['C'], 8)
        self.assertEqual(tokenizer.token_to_id['1'], 3 + 119)

    def test_isotope_masks(self):
        tokenizer = SnackTokenizer(max_nodes=4)
        # A mass number may follow any node, and must come before the others.
        self.assertTrue({'(1)', '(2)', '(13)', '(294)'} <= self.attributes_allowed(tokenizer, 'C'))
        self.assertTrue({'(13)'} <= self.attributes_allowed(tokenizer, 'F'))
        self.assertTrue({'(13)'}.isdisjoint(self.attributes_allowed(tokenizer, 'C(+)')))
        self.assertTrue({'(13)'}.isdisjoint(self.attributes_allowed(tokenizer, 'C(13)')))
        for text in ('C(+)(13)', 'C(H3)(13)', 'C(@)(13)', 'C(13)(13)', 'C(0)', 'C1-(13)C'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokenizer.encode(text, validate=True)
        tokenizer.encode('C(13)(+)(H3)(@)', validate=True)
        # An isotope leaves the valence untouched: the same hydrogens are allowed.
        self.assertEqual({t for t in self.attributes_allowed(tokenizer, 'C(13)')
                          if t.startswith('(H')},
                         {t for t in self.attributes_allowed(tokenizer, 'C') if t.startswith('(H')})
        self.assertEqual(snack.snack_to_smiles('C(13)(+)(H3)'), '[13CH3+]')

    def test_dataset_presets(self):
        from snack.tokenizer import DATASETS
        sizes = {'qm9': 20, 'moses': 38}
        for preset, size in sizes.items():
            with self.subTest(preset=preset):
                tokenizer = SnackTokenizer(preset)
                self.assertEqual(len(tokenizer), size)
                self.assertEqual(tokenizer.node_types, DATASETS[preset]['node_types'])
                self.assertEqual(tokenizer.edge_types, ('-', '=', '#'))
                self.assertTrue(tokenizer.molecular)
                self.assertTrue(tokenizer.valence)
                self.assertEqual(tokenizer.max_distance, tokenizer.max_nodes - 1)
        self.assertEqual(SnackTokenizer('qm9').max_nodes, 9)
        self.assertEqual(SnackTokenizer('moses').max_nodes, 26)
        # MOSES needs no attributes; QM9 needs single charges and nothing else.
        self.assertEqual(SnackTokenizer('moses').attribute_types, ())
        self.assertEqual(SnackTokenizer('qm9').attribute_types, ('(-)', '(+)'))
        # Overrides still win over the preset.
        self.assertEqual(SnackTokenizer('qm9', max_nodes=12).max_nodes, 12)
        self.assertEqual(SnackTokenizer('qm9', attributes=()).attribute_types, ())
        with self.assertRaises(ValueError):
            SnackTokenizer('zinc250k')

    def test_presets_encode_their_datasets(self):
        qm9 = SnackTokenizer('qm9', ordering='bfs')
        moses = SnackTokenizer('moses', ordering='bfs')
        for tokenizer, molecules in ((qm9, ['CC(=O)NC', 'O=[N+]([O-])C', 'FC1CC1O', 'CC#N']),
                                     (moses, ['CC(=O)Oc1ccccc1C(=O)O', 'Brc1ccc(S(N)(=O)=O)cc1',
                                              'CN1CCN(c2ccccc2)CC1'])):
            for smiles in molecules:
                with self.subTest(smiles=smiles):
                    text = snack.smiles_to_snack(smiles, ordering='bfs')
                    ids = tokenizer.encode(text, validate=True)
                    self.assertEqual(tokenizer.decode(ids), text)
                    self.assertEqual(snack.snack_to_smiles(tokenizer.decode(ids)),
                                     Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))

    def test_aromatic_bonds_are_opt_in(self):
        text = snack.smiles_to_snack('c1ccccc1', ordering=None, kekulize=False)
        # The kekule bonds are the default, so an aromatic string is refused
        # outright rather than encoded into a vocabulary whose mask forbids it.
        default = SnackTokenizer(max_nodes=8)
        self.assertNotIn('a', default.token_to_id)
        with self.assertRaisesRegex(ValueError, 'outside this vocabulary'):
            default.encode(text)
        self.assertNotIn('a', self.allowed(default, 'C1'))
        self.assertNotIn('a', self.allowed(SnackTokenizer('moses'), 'C1'))
        # Asking for it puts it in the vocabulary and in generation together.
        aromatic = SnackTokenizer(max_nodes=8, edge_types=('-', '=', '#', 'a'))
        self.assertEqual(len(aromatic), len(default) + 1)
        self.assertEqual(aromatic.decode(aromatic.encode(text)), text)
        self.assertIn('a', self.allowed(aromatic, 'C1'))

    def test_attribute_selection_and_graph_preset(self):
        charged = SnackTokenizer(attributes=('charge',))
        self.assertTrue(all(snack.graph.parse_attribute(t)[0] == 'charge' for t in charged.attribute_types))
        self.assertEqual(attribute_tokens(('cis_trans',)), ['(c)', '(t)'])
        self.assertEqual(attribute_tokens(('hydrogens',), max_hydrogens=2), ['(H0)', '(H)', '(H2)'])
        self.assertEqual(attribute_tokens(('chirality',)), ['(@)', '(@@)'])
        self.assertEqual(attribute_tokens(('isotope',), max_isotope=3), ['(1)', '(2)', '(3)'])
        graph = SnackTokenizer('graph', max_nodes=3)
        self.assertEqual(graph.tokens, ('[pad]', '[bos]', '[eos]', 'o', '1', '2', '-'))
        with self.assertRaises(ValueError):
            SnackTokenizer('graph', attributes=('charge',))
        for kwargs in ({'max_nodes': 0}, {'max_nodes': True}, {'max_nodes': 3, 'max_distance': 3},
                       {'ordering': 'cm'}, {'preset': 'zinc250k'}, {'preset': 'graph', 'valence': True},
                       {'attributes': ('mass',)}, {'attributes': ('radical',)}, {'edge_types': ['u']}, {'edge_types': ['a', 'a']},
                       {'node_types': ['C+']}, {'node_types': ['[C]']}, {'node_types': ['Xx']},
                       {'allowed_node_types': ['Xx']}, {'allowed_node_types': ['[C]']}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SnackTokenizer(**kwargs)

    def test_exact_roundtrip_and_special_tokens(self):
        for preset, text in (('graph', 'o1-o2-1-o'), ('mol', 'C1-C(@)1-N(+)(H)2-C1=(t)C1-O(-)'),
                             ('mol', 'Fe(+3)'), ('mol', 'Na(+)Cl(-)'), ('mol', 'Cl1-C1-Br'), ('mol', '*1-C')):
            with self.subTest(text=text):
                tokenizer = SnackTokenizer(preset)
                ids = tokenizer.encode(text)
                self.assertTrue(all(isinstance(index, int) for index in ids))
                self.assertEqual(ids[0], tokenizer.bos_id)
                self.assertEqual(ids[-1], tokenizer.eos_id)
                self.assertEqual(tokenizer.decode(ids + [tokenizer.pad_id] * 3), text)
                self.assertEqual(tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)), text)
                self.assertEqual(tokenizer(text), {'input_ids': ids})
        tokenizer = SnackTokenizer()
        for ids in ([999999], [-1], [True], [1.0], [tokenizer.eos_id, tokenizer.bos_id],
                    [tokenizer.pad_id, tokenizer.token_to_id['C']]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                tokenizer.decode(ids)
        for text in ('o', 'C{@}', '[C]', 'c1-C', 'C(.)', 'Xx'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokenizer.encode(text)
        with self.assertRaises(ValueError):
            SnackTokenizer('graph').encode('C')

    def test_capacity_follows_rdkit(self):
        self.assertEqual(capacity('C'), 4)
        self.assertEqual(capacity('C', 1), 3)
        self.assertEqual(capacity('C', -1), 3)
        self.assertEqual(capacity('N', 1), 4)
        self.assertEqual(capacity('O', -1), 1)
        self.assertEqual(capacity('B', -1), 4)
        self.assertEqual(capacity('S'), 6)
        self.assertEqual(capacity('He'), 0)
        self.assertIsNone(capacity('He', -1))
        self.assertEqual(capacity('Fe'), UNLIMITED)
        self.assertEqual(capacity('*'), UNLIMITED)
        # With a given H count RDKit ignores noble-gas limits and relaxes some charged states.
        self.assertEqual(capacity('He', 0, lenient=True), UNLIMITED)
        self.assertEqual(capacity('N', 0, lenient=True), 3)
        self.assertEqual(capacity('C', 1, lenient=True), 5)
        self.assertEqual(capacity('Sn', 2, lenient=True), 6)

    # ----- graph preset grammar ------------------------------------------------
    def test_shifted_masks_and_padding(self):
        tokenizer = SnackTokenizer('graph', max_nodes=3, ordering='bfs')
        ids = tokenizer.encode('o1-o1-o', validate=True)
        masks = tokenizer.get_logit_mask(ids)
        self.assertEqual(len(masks), len(ids))
        self.assertTrue(all(len(row) == len(tokenizer) for row in masks))
        for position, target in enumerate(ids[1:]):
            self.assertTrue(masks[position][target])
        self.assertEqual(tokenizer.allowed_next_ids(ids), [tokenizer.pad_id])
        self.assertEqual(tokenizer.next_token_mask(ids), masks[-1])

    def test_grammar_and_connectedness(self):
        tokenizer = SnackTokenizer('graph', max_nodes=4)
        self.assertEqual(tokenizer.allowed_next_ids([]), [tokenizer.bos_id])
        self.assertEqual(self.allowed(tokenizer, ''), {'o'})
        self.assertNotIn('o', self.allowed(tokenizer, 'o'))
        self.assertNotIn('[eos]', self.allowed(tokenizer, 'o1-'))
        self.assertEqual(self.allowed(tokenizer, 'o1-'), {'o'})
        for text in ('v0_v', 'o2-o', 'o1-1-o', 'oo', 'o1-o1-2-o', 'o1-'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokenizer.encode(text, validate=True)
        disconnected = SnackTokenizer('graph', max_nodes=4, allow_disconnected=True)
        disconnected.encode('oo2-1-oo', validate=True)

    def test_bfs_parent_must_not_move_backwards(self):
        unconstrained = SnackTokenizer('graph', max_nodes=4)
        bfs = SnackTokenizer('graph', max_nodes=4, ordering='bfs')
        unconstrained.encode('o1-o1-o3-o', validate=True)
        with self.assertRaises(ValueError):
            bfs.encode('o1-o1-o3-o', validate=True)
        self.assertNotIn('3', self.allowed(bfs, 'o1-o1-o'))
        self.assertIn('2', self.allowed(bfs, 'o1-o1-o'))

    def test_bfs_matches_independent_exhaustive_oracle(self):
        # Every labeled simple graph through four nodes, across all four layouts.
        for size in range(1, 5):
            edges = list(itertools.combinations(range(size), 2))
            for bits in range(1 << len(edges)):
                graph = nx.empty_graph(size)
                graph.add_edges_from(edge for i, edge in enumerate(edges) if bits & (1 << i))
                expected_order, visited = [], set()
                for node in graph:
                    if node not in visited:
                        order = list(nx.bfs_tree(graph, node, sort_neighbors=sorted))
                        expected_order.extend(order)
                        visited.update(order)
                valid_bfs = expected_order == list(graph)
                for node_first, ascending in itertools.product((False, True), repeat=2):
                    snack.config.set(node_first, ascending)
                    tokenizer = SnackTokenizer('graph', max_nodes=size, ordering='bfs', allow_disconnected=True)
                    text = snack.graph_to_snack(graph)
                    with self.subTest(size=size, bits=bits, node_first=node_first, ascending=ascending):
                        if valid_bfs:
                            tokenizer.encode(text, validate=True)
                        else:
                            with self.assertRaises(ValueError):
                                tokenizer.encode(text, validate=True)

    def test_disconnected_bfs_cannot_reopen_previous_component(self):
        tokenizer = SnackTokenizer('graph', max_nodes=6, ordering='bfs', allow_disconnected=True)
        tokenizer.encode('o1-oo1-o', validate=True)
        with self.assertRaises(ValueError):
            tokenizer.encode('o1-oo2-o', validate=True)

    def test_node_limit_includes_all_components(self):
        tokenizer = SnackTokenizer('graph', max_nodes=3, allow_disconnected=True, ordering='bfs')
        self.assertEqual(self.allowed(tokenizer, 'ooo'), {'[eos]'})
        tokenizer = SnackTokenizer('graph', max_nodes=3)
        self.assertEqual(self.allowed(tokenizer, 'o1-o1-o'), {'[eos]'})
        tokenizer = SnackTokenizer('graph', max_nodes=1)
        self.assertEqual(self.allowed(tokenizer, 'o'), {'[eos]'})

    def test_node_first_finishes_last_node_edges_before_eos(self):
        tokenizer = SnackTokenizer('graph', max_nodes=2, node_first=True)
        self.assertEqual(self.allowed(tokenizer, 'oo'), {'1'})
        self.assertEqual(self.allowed(tokenizer, 'oo1'), {'-'})
        self.assertEqual(self.allowed(tokenizer, 'oo1-'), {'[eos]'})

    def test_settings_are_captured_without_global_mutation(self):
        tokenizer = SnackTokenizer('graph', node_first=False, ascending=False, max_nodes=3)
        before = self.allowed(tokenizer, 'o1-o')
        snack.config.set(True, True)
        self.assertEqual(self.allowed(tokenizer, 'o1-o'), before)
        self.assertTrue(snack.config.NODE_FIRST)
        self.assertTrue(snack.config.ASCENDING)

    # ----- molecular masks -----------------------------------------------------
    def test_typed_distance_requires_a_bond(self):
        tokenizer = SnackTokenizer(max_nodes=4)
        self.assertEqual(self.allowed(tokenizer, 'C1'), {'-', '=', '#'})
        self.assertEqual(self.allowed(tokenizer, 'F1'), {'-'})
        self.assertEqual(self.allowed(tokenizer, 'O1'), {'-', '='})
        self.assertNotIn('[eos]', self.allowed(tokenizer, 'C1-'))
        for text in ('C1C', 'C1_C', 'C-C', 'F1=C'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokenizer.encode(text, validate=True)

    def test_two_letter_elements_tokenize_greedily(self):
        tokenizer = SnackTokenizer(max_nodes=4)
        self.assertEqual(snack.split('Cl1-C1-Br'), ['Cl', '1', '-', 'C', '1', '-', 'Br'])
        self.assertEqual(snack.split('Co1-O'), ['Co', '1', '-', 'O'])
        ids = tokenizer.encode('Cl1-C1-Br', validate=True)
        self.assertEqual([tokenizer.tokens[i] for i in ids[1:-1]], ['Cl', '1', '-', 'C', '1', '-', 'Br'])
        self.assertEqual(snack.snack_to_smiles(tokenizer.decode(ids)), 'ClCBr')

    def test_charge_extends_valence(self):
        tokenizer = SnackTokenizer(max_nodes=6)
        # Fourth single bond into atom zero violates N but is allowed for N(+).
        self.assertNotIn('4', self.allowed(tokenizer, 'N1-C2-C3-C'))
        self.assertIn('4', self.allowed(tokenizer, 'N(+)1-C2-C3-C'))
        self.assertNotIn('2', self.allowed(tokenizer, 'F1-C'))
        self.assertNotIn('2', self.allowed(tokenizer, 'O(-)1-C'))
        # Four bonds already placed on the pending atom: it must support valence 4.
        pending = 'C1-C1-C3-2-1='
        allowed = self.allowed(tokenizer, pending)
        self.assertIn('C', allowed)
        self.assertIn('N', allowed)    # fixable by (+)
        self.assertIn('Fe', allowed)
        self.assertIn('F', allowed)    # RDKit accepts four bonds on F(+3)
        self.assertIn('He', allowed)   # accepted once its H count is given
        # Without charge or hydrogens tokens the element alone must carry the bonds.
        neutral = SnackTokenizer(max_nodes=6, attributes=('chirality',))
        allowed = self.allowed(neutral, pending)
        self.assertIn('C', allowed)
        self.assertNotIn('N', allowed)
        self.assertNotIn('F', allowed)
        self.assertNotIn('He', allowed)
        # After an over-bonded neutral N only a charge token can follow.
        allowed = self.allowed(tokenizer, pending + 'N')
        self.assertNotIn('[eos]', allowed)
        self.assertNotIn('C', allowed)
        self.assertFalse(any(token.isdigit() for token in allowed))
        self.assertTrue(all(snack.graph.parse_attribute(token)[0] in ('charge', 'isotope')
                            for token in allowed))
        # A mass number does not repair the valence, so a charge is still required.
        after_isotope = self.allowed(tokenizer, pending + 'N(13)')
        self.assertNotIn('[eos]', after_isotope)
        self.assertTrue(all(snack.graph.parse_attribute(token)[0] == 'charge'
                            for token in after_isotope))
        self.assertIn('(+)', allowed)
        self.assertNotIn('(-)', allowed)
        after_fix = self.allowed(tokenizer, pending + 'N(+)')
        self.assertEqual({t for t in after_fix if not t.isdigit()}, {'(H0)', '(@)', '(@@)', '[eos]'})
        with self.assertRaises(ValueError):
            tokenizer.encode(pending + 'N', validate=True)
        tokenizer.encode(pending + 'N(+)', validate=True)

    def test_attribute_order_and_capacity(self):
        # Without the isotope kind, whose tokens are allowed after every node
        # and would swamp these sets; test_isotope_masks covers it separately.
        tokenizer = SnackTokenizer(max_nodes=4,
                                   attributes=('charge', 'hydrogens', 'chirality', 'cis_trans'))
        after_c = self.attributes_allowed(tokenizer, 'C')
        self.assertEqual(after_c, {'(+)', '(+2)', '(+3)', '(+4)', '(+5)', '(+6)', '(+7)', '(+8)',
                                   '(-)', '(-2)', '(-3)', '(-4)', '(H0)', '(H)', '(H2)', '(H3)', '(H4)',
                                   '(@)', '(@@)'})
        # Each attribute kind at most once, in order charge, hydrogens, chirality.
        # RDKit allows up to five hydrogens on a C+ with a given H count.
        self.assertEqual(self.attributes_allowed(tokenizer, 'C(+)'),
                         {'(H0)', '(H)', '(H2)', '(H3)', '(H4)', '(H5)', '(@)', '(@@)'})
        self.assertEqual(self.attributes_allowed(tokenizer, 'C(+)(H3)'), {'(@)', '(@@)'})
        self.assertEqual(self.attributes_allowed(tokenizer, 'C(+)(H3)(@)'), set())
        self.assertIn('[eos]', self.allowed(tokenizer, 'C(+)(H3)(@)'))
        self.assertEqual(self.attributes_allowed(tokenizer, 'C(H3)'), {'(@)', '(@@)'})
        self.assertEqual(self.attributes_allowed(tokenizer, 'C(@)'), set())
        for text in ('C(@)(+)', 'C(H)(+)', 'C(@)(H)', 'C(+)(+)', 'C(H)(H)', 'C(@)(@@)', 'C(.)',
                     'C(H9)', 'F(@)', 'C1-(+)C', 'C(H4)(@)(+)', 'C(+9)'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokenizer.encode(text, validate=True)
        tokenizer.encode('C(-)(H2)(@)', validate=True)
        self.assertEqual(snack.snack_to_smiles('C(+)(H3)'), '[CH3+]')
        fluorine = self.attributes_allowed(tokenizer, 'F')
        self.assertTrue({'(+)', '(-)', '(H)', '(H0)'} <= fluorine)
        self.assertTrue({'(@)', '(@@)', '(H2)', '(-2)'}.isdisjoint(fluorine))
        # Helium has no valence to spare unless its H count is given, which RDKit never limits.
        helium = self.attributes_allowed(tokenizer, 'He')
        self.assertTrue({'(+)', '(-)', '(H0)', '(H8)'} <= helium)   # RDKit accepts [He-] with a given H count
        self.assertNotIn('(@)', helium)
        # A complete node: only EOS or the next node's first edge may follow.
        self.assertEqual(self.allowed(tokenizer, 'He(H0)'), {'1', '[eos]'})
        self.assertEqual(snack.snack_to_smiles('He(H0)'), '[He]')

    def test_cis_trans_only_after_double_bonds(self):
        tokenizer = SnackTokenizer(max_nodes=4)
        self.assertTrue({'(c)', '(t)'} <= self.allowed(tokenizer, 'C1='))
        self.assertEqual(self.attributes_allowed(tokenizer, 'C1-'), set())
        self.assertEqual(self.attributes_allowed(tokenizer, 'C1=(t)'), set())
        self.assertEqual(self.attributes_allowed(tokenizer, 'C1'), set())
        for text in ('C1-(c)C', 'C1=(c)(t)C', 'C(t)'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokenizer.encode(text, validate=True)

    def test_node_first_keeps_room_for_the_backward_edge(self):
        tokenizer = SnackTokenizer(max_nodes=3, node_first=True)
        # F after C must bond back, so it may not spend its only valence on a hydrogen;
        # (H0) is fine, and so is a charge that leaves room.
        after_f = self.allowed(tokenizer, 'CF')
        self.assertIn('(H0)', after_f)
        self.assertNotIn('(H)', after_f)
        self.assertIn('(+)', after_f)
        self.assertNotIn('(-)', after_f)   # F(-) has no valence left for the bond
        self.assertNotIn('[eos]', after_f)
        self.assertIn('(H)', self.allowed(tokenizer, 'F'))   # the first node needs no backward edge
        loose = SnackTokenizer(max_nodes=3, node_first=True, allow_disconnected=True)
        self.assertIn('(H)', self.allowed(loose, 'CF'))
        self.assertEqual(self.allowed(tokenizer, 'CF(H0)'), {'1'})
        self.assertEqual(self.allowed(tokenizer, 'CF1'), {'-'})
        self.assertEqual(self.allowed(tokenizer, 'C(H3)F1-'), {'[eos]'})
        self.assertEqual(self.allowed(tokenizer, 'C(H4)'), {'(@)', '(@@)', '[eos]'})
        self.assertNotIn('F', self.allowed(tokenizer, 'C(H4)(@)'))

    def test_whitelists_restrict_generation_but_not_encoding(self):
        tokenizer = SnackTokenizer(allowed_node_types=['C', 'O'], allowed_edge_types=['-'])
        self.assertEqual(self.allowed(tokenizer, ''), {'C', 'O'})
        self.assertEqual(self.allowed(tokenizer, 'C1'), {'-'})
        self.assertEqual(tokenizer.decode(tokenizer.encode('N')), 'N')
        with self.assertRaises(ValueError):
            tokenizer.encode('N', validate=True)
        with self.assertRaises(ValueError):
            tokenizer.encode('C1=C', validate=True)
        custom = SnackTokenizer(node_types=['C', 'N'], edge_types=['-', '='], attributes=('charge',))
        self.assertEqual(custom.node_types, ('C', 'N'))
        self.assertNotIn('(@)', custom.token_to_id)
        self.assertNotIn('Cl', custom.token_to_id)
        custom.encode('C1-N(+)', validate=True)
        with self.assertRaises(ValueError):
            custom.encode('C1#N', validate=True)
        with self.assertRaises(ValueError):
            custom.encode('Cl1-C')

    def test_valence_false_only_checks_grammar(self):
        tokenizer = SnackTokenizer(max_nodes=4, valence=False)
        tokenizer.encode('F1#F', validate=True)
        self.assertEqual(self.allowed(tokenizer, 'F1'), {'-', '=', '#'})

    def test_encoded_molecules_pass_masks(self):
        for node_first, ascending in itertools.product((False, True), repeat=2):
            snack.config.set(node_first, ascending)
            tokenizer = SnackTokenizer(ordering='bfs', node_first=node_first, ascending=ascending)
            for smiles, seed in itertools.product(MOLECULES, range(2)):
                random.seed(seed)
                text = snack.smiles_to_snack(smiles)
                with self.subTest(smiles=smiles, node_first=node_first, ascending=ascending, seed=seed):
                    ids = tokenizer.encode(text, validate=True)
                    self.assertEqual(tokenizer.decode(ids), text)
                    masks = tokenizer.get_logit_mask(ids)
                    self.assertTrue(all(masks[i][target] for i, target in enumerate(ids[1:])))
                    self.assertEqual(snack.snack_to_smiles(tokenizer.decode(ids)),
                                     Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
        # Salts need allow_disconnected; a connected tokenizer rejects them.
        with self.assertRaises(ValueError):
            SnackTokenizer(ordering='bfs').encode('Na(+)Cl(-)', validate=True)
        SnackTokenizer(ordering='bfs', allow_disconnected=True).encode('Na(+)Cl(-)', validate=True)

    def test_incremental_and_prefix_masks_agree(self):
        tokenizer = SnackTokenizer(ordering='bfs')
        ids = tokenizer.encode('C1-C(@)1-N(+)(H)2-C1=(t)C1-O(-)', validate=True)
        state = tokenizer.new_state()
        for i, index in enumerate(ids):
            tokenizer.step(state, index)
            self.assertEqual(tokenizer.allowed_token_ids(state), tokenizer.allowed_next_ids(ids[:i + 1]))
        with self.assertRaises(ValueError):
            tokenizer.step(tokenizer.new_state(), tokenizer.token_to_id['C'])

    def generate(self, tokenizer, seed, prefer=()):
        rng = random.Random(seed)
        state, ids = tokenizer.new_state(), []
        for _ in range(300):
            allowed = tokenizer.allowed_token_ids(state)
            self.assertTrue(allowed, f'Dead end: {tokenizer.decode(ids)}')
            preferred = [i for i in allowed if tokenizer.tokens[i] in prefer]
            index = rng.choice(preferred) if preferred and rng.random() < 0.6 else rng.choice(allowed)
            tokenizer.step(state, index)
            ids.append(index)
            if index == tokenizer.eos_id:
                return tokenizer.decode(ids)
        self.fail('Generation did not terminate')

    def test_random_generation_is_valid(self):
        with rdBase.BlockLogs():
            for preset in ('graph', 'mol'):
                for node_first, ascending, disconnected, bfs in itertools.product((False, True), repeat=4):
                    snack.config.set(node_first, ascending)
                    tokenizer = SnackTokenizer(preset, max_nodes=6, node_first=node_first, ascending=ascending,
                                               allow_disconnected=disconnected, ordering='bfs' if bfs else None)
                    for seed in range(8):
                        with self.subTest(preset=preset, node_first=node_first, ascending=ascending,
                                          disconnected=disconnected, bfs=bfs, seed=seed):
                            text = self.generate(tokenizer, seed, prefer=tokenizer.attribute_types)
                            graph = snack.snack_to_graph(text, strict=False)
                            self.assertLessEqual(len(graph), 6)
                            if not disconnected:
                                self.assertTrue(nx.is_connected(graph))
                            if preset == 'mol':
                                # Every generated string decodes and sanitizes.
                                mol = snack.snack_to_mol(text)
                                self.assertEqual(mol.GetNumAtoms(), len(graph))

    def test_organic_generation_reparses_as_smiles(self):
        # With common elements and small charges RDKit can always re-read its own SMILES.
        with rdBase.BlockLogs():
            for node_first, ascending in itertools.product((False, True), repeat=2):
                snack.config.set(node_first, ascending)
                tokenizer = SnackTokenizer(max_nodes=8, ordering='bfs', node_first=node_first,
                                           ascending=ascending, allowed_node_types=ORGANIC)
                for seed in range(25):
                    with self.subTest(node_first=node_first, ascending=ascending, seed=seed):
                        text = self.generate(tokenizer, seed, prefer=('(@)', '(@@)', '(c)', '(t)', '(+)', '(-)'))
                        smiles = snack.snack_to_smiles(text)
                        self.assertIsNotNone(Chem.MolFromSmiles(smiles), (text, smiles))

    def test_no_training_library_import(self):
        code = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'pandas', 'sklearn', 'numpy'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from snack.tokenizer import SnackTokenizer
for preset, text in [('graph', 'o1-o'), ('mol', 'C1-O(-)'), ('mol', 'C1=(t)C'), ('mol', 'Cl1-C(H2)')]:
    t = SnackTokenizer(preset, ordering='bfs')
    ids = t.encode(text, validate=True)
    assert t.decode(ids) == text
    assert t.get_logit_mask(ids)
assert not {'torch', 'pandas', 'sklearn'} & sys.modules.keys()
'''
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                cwd=Path(snack.__file__).resolve().parent.parent,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
