"""Regression tests for the torch-free downstream tokenizer."""
import itertools
from pathlib import Path
import random
import subprocess
import sys
import unittest

import networkx as nx
from rdkit import Chem

import snack
from snack.tokenizer import SnackTokenizer, basic_vocabulary


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

    def test_vocabulary_and_exact_roundtrip(self):
        examples = [('graph', 'v1_v2_1_v'), ('moses', '[C]1-[N]1=[O]'),
                    ('zinc250k', '[NH2+]1-[C]1-[O-]'),
                    ('guacamol', '[SiH2]1-[Se]1-[B-]')]
        for preset, text in examples:
            with self.subTest(preset=preset):
                tokenizer = SnackTokenizer(preset)
                self.assertEqual(tuple(basic_vocabulary(preset)), tokenizer.tokens)
                ids = tokenizer.encode(text)
                self.assertTrue(all(isinstance(index, int) for index in ids))
                self.assertEqual(tokenizer.decode(ids + [tokenizer.pad_id] * 3), text)
                self.assertEqual(tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)), text)
        self.assertIn('[P+]', basic_vocabulary('zinc250k'))
        self.assertNotIn('[P+]', basic_vocabulary('moses'))

    def test_guacamol_vocabulary_and_limits(self):
        tokenizer = SnackTokenizer('guacamol')
        self.assertEqual(tokenizer.max_nodes, 88)
        self.assertEqual(tokenizer.max_distance, 87)
        for token in ('[B]', '[B-]', '[C+]', '[Si]', '[Si-]', '[Se]', '[Se+]', '[Se-]',
                      '[Cl-]', '[Br+2]', '[Cl+3]', '[I+3]', '[P-]', '[IH2]', '[SiH-]'):
            self.assertIn(token, tokenizer.token_to_id)
        self.assertIn('87', tokenizer.token_to_id)
        self.assertNotIn('88', tokenizer.token_to_id)
        limited = SnackTokenizer('guacamol', max_nodes=2, ordering='bfs')
        self.assertEqual(self.allowed(limited, '[B]1-[Si]'), {'[eos]'})
        self.assertNotIn('[IH2]', basic_vocabulary('zinc250k'))

    def test_guacamol_rare_atoms_roundtrip_with_valence_masks(self):
        # Includes actual GuacaMol motifs that the old valence table excluded:
        # valence-6 P-, valence-5 Si-, and neutral hypervalent iodine.
        examples = ['B(O)(O)O', '[B-](F)(F)(F)F', 'C[Si](C)(C)C', 'C[Se]C',
                    'C[Si](C)(C)c1ccccc1', 'c1cc[se]c1', 'c1ccc2c(c1)[IH]c1ccccc1-2',
                    'C[Se+](C)C', 'N#C[Se-]', 'F[P-](F)(F)(F)(F)F',
                    'F[Si-](F)(C)(C)C', 'C[IH2](O)O', 'O[I+2](O)O',
                    'O[I+3](O)(O)O', 'C[I+]C', 'O[Cl+2](O)O',
                    'O[Cl+3](O)(O)O', 'O[Br+2](O)O', '[O-][Cl+]O',
                    'C[F+]C', 'C[CH2+]', '[F-]', '[Cl-]', '[Br-]', '[I-]']
        for node_first, ascending in itertools.product((False, True), repeat=2):
            snack.config.set(node_first, ascending)
            base = SnackTokenizer('guacamol')
            tokenizer = SnackTokenizer('guacamol', ordering='bfs', allowed_node_types=base.node_types)
            for smiles in examples:
                with self.subTest(smiles=smiles, node_first=node_first, ascending=ascending):
                    mol = snack.smiles_to_mol(smiles)
                    text = snack.mol_to_snack(mol, ordering='bfs', use_hydrogens=True)
                    ids = tokenizer.encode(text, validate=True)
                    self.assertEqual(tokenizer.decode(ids), text)
                    restored = snack.snack_to_mol(tokenizer.decode(ids))
                    # Input conversion kekulizes; restore aromaticity before
                    # comparing canonical SMILES with the sanitized output.
                    Chem.SanitizeMol(mol)
                    self.assertEqual(Chem.MolToSmiles(restored), Chem.MolToSmiles(mol))

    def test_guacamol_halide_ions_and_bond_capacity(self):
        tokenizer = SnackTokenizer('guacamol', max_nodes=8)
        for ion in ('[F-]', '[Cl-]', '[Br-]', '[I-]'):
            self.assertEqual(self.allowed(tokenizer, ion), {'[eos]'})
            self.assertNotIn(ion, self.allowed(tokenizer, '[C]1-'))
            with self.assertRaises(ValueError):
                tokenizer.encode(ion + '1-[C]', validate=True)
        self.assertNotIn('4', self.allowed(tokenizer, '[B]1-[C]2-[C]3-[C]'))
        self.assertIn('4', self.allowed(tokenizer, '[B-]1-[C]2-[C]3-[C]'))
        self.assertEqual(self.allowed(tokenizer, '[Si]1-[C]2-[C]3-[C]4'), {'-'})
        zinc = SnackTokenizer('zinc250k')
        self.assertNotIn('2', self.allowed(zinc, '[I]1-[C]'))
        self.assertIn('2', self.allowed(tokenizer, '[I]1-[C]'))
        disconnected = SnackTokenizer('guacamol', allow_disconnected=True, ordering='bfs')
        disconnected.encode('[C][Cl-]', validate=True)

    def test_bad_inputs(self):
        for kwargs in ({'max_nodes': 0}, {'max_nodes': True}, {'max_nodes': 3, 'max_distance': 3},
                       {'ordering': 'cm'}, {'preset': 'unknown'}, {'valence': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SnackTokenizer(**kwargs)
        tokenizer = SnackTokenizer()
        for ids in ([999999], [-1], [True], [1.0], [tokenizer.eos_id, tokenizer.bos_id],
                    [tokenizer.pad_id, tokenizer.token_to_id['v']]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                tokenizer.decode(ids)
        with self.assertRaises(ValueError):
            tokenizer.encode('[C]')
        with self.assertRaises(ValueError):
            tokenizer.encode('v?')

    def test_shifted_masks_and_padding(self):
        tokenizer = SnackTokenizer(max_nodes=3, ordering='bfs')
        ids = tokenizer.encode('v1_v1_v', validate=True)
        masks = tokenizer.get_logit_mask(ids)
        self.assertEqual(len(masks), len(ids))
        self.assertTrue(all(len(row) == len(tokenizer) for row in masks))
        for position, target in enumerate(ids[1:]):
            self.assertTrue(masks[position][target])
        self.assertEqual(tokenizer.allowed_next_ids(ids), [tokenizer.pad_id])
        self.assertEqual(tokenizer.next_token_mask(ids), masks[-1])
        self.assertEqual(tokenizer('v1_v'), {'input_ids': tokenizer.encode('v1_v')})

    def test_grammar_and_connectedness(self):
        tokenizer = SnackTokenizer(max_nodes=4)
        self.assertEqual(tokenizer.allowed_next_ids([]), [tokenizer.bos_id])
        self.assertEqual(self.allowed(tokenizer, ''), {'v'})
        self.assertNotIn('v', self.allowed(tokenizer, 'v'))
        self.assertNotIn('[eos]', self.allowed(tokenizer, 'v1_'))
        self.assertEqual(self.allowed(tokenizer, 'v1_'), {'v'})
        for text in ('v0_v', 'v2_v', 'v1_1_v', 'vv', 'v1_v1_2_v', 'v1_'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokenizer.encode(text, validate=True)
        disconnected = SnackTokenizer(max_nodes=4, allow_disconnected=True)
        disconnected.encode('vv2_1_vv', validate=True)

    def test_bfs_parent_must_not_move_backwards(self):
        unconstrained = SnackTokenizer(max_nodes=4)
        bfs = SnackTokenizer(max_nodes=4, ordering='bfs')
        unconstrained.encode('v1_v1_v3_v', validate=True)
        with self.assertRaises(ValueError):
            bfs.encode('v1_v1_v3_v', validate=True)
        self.assertNotIn('3_', self.allowed(bfs, 'v1_v1_v'))
        self.assertIn('2_', self.allowed(bfs, 'v1_v1_v'))

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
                    tokenizer = SnackTokenizer(max_nodes=size, ordering='bfs', allow_disconnected=True)
                    text = snack.graph_to_snack(graph)
                    with self.subTest(size=size, bits=bits, node_first=node_first, ascending=ascending):
                        if valid_bfs:
                            tokenizer.encode(text, validate=True)
                        else:
                            with self.assertRaises(ValueError):
                                tokenizer.encode(text, validate=True)

    def test_disconnected_bfs_cannot_reopen_previous_component(self):
        tokenizer = SnackTokenizer(max_nodes=6, ordering='bfs', allow_disconnected=True)
        tokenizer.encode('v1_vv1_v', validate=True)
        with self.assertRaises(ValueError):
            tokenizer.encode('v1_vv2_v', validate=True)

    def test_node_limit_includes_all_components(self):
        tokenizer = SnackTokenizer(max_nodes=3, allow_disconnected=True, ordering='bfs')
        self.assertEqual(self.allowed(tokenizer, 'vvv'), {'[eos]'})
        tokenizer = SnackTokenizer(max_nodes=3)
        self.assertEqual(self.allowed(tokenizer, 'v1_v1_v'), {'[eos]'})
        tokenizer = SnackTokenizer(max_nodes=1)
        self.assertEqual(self.allowed(tokenizer, 'v'), {'[eos]'})

    def test_node_first_finishes_last_node_edges_before_eos(self):
        tokenizer = SnackTokenizer(max_nodes=2, node_first=True)
        self.assertEqual(self.allowed(tokenizer, 'vv'), {'1_'})
        self.assertEqual(self.allowed(tokenizer, 'vv1_'), {'[eos]'})

    def test_settings_are_captured_without_global_mutation(self):
        tokenizer = SnackTokenizer(node_first=False, ascending=False, max_nodes=3)
        before = self.allowed(tokenizer, 'v1_v')
        snack.config.set(True, True)
        self.assertEqual(self.allowed(tokenizer, 'v1_v'), before)
        self.assertTrue(snack.config.NODE_FIRST)
        self.assertTrue(snack.config.ASCENDING)

    def test_typed_distance_requires_a_bond(self):
        tokenizer = SnackTokenizer('moses', max_nodes=4)
        self.assertEqual(self.allowed(tokenizer, '[C]1'), {'-', '=', '#'})
        self.assertEqual(self.allowed(tokenizer, '[F]1'), {'-'})
        self.assertNotIn('[eos]', self.allowed(tokenizer, '[C]1-'))
        for text in ('[C]1[C]', '[C]1_ [C]', '[C]-[C]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokenizer.encode(text, validate=True)

    def test_endpoint_valences_and_charge(self):
        tokenizer = SnackTokenizer('zinc250k', max_nodes=6)
        self.assertEqual(self.allowed(tokenizer, '[O]1'), {'-', '='})
        self.assertNotIn('2', self.allowed(tokenizer, '[F]1-[C]'))
        with self.assertRaises(ValueError):
            tokenizer.encode('[F]1=[C]', validate=True)
        # Fourth single bond into atom zero violates N but is allowed for N+.
        neutral = '[N]1-[C]2-[C]3-[C]'
        charged = '[N+]1-[C]2-[C]3-[C]'
        self.assertNotIn('4', self.allowed(tokenizer, neutral))
        self.assertIn('4', self.allowed(tokenizer, charged))
        # Four bonds already placed on the pending atom: it must support valence 4.
        pending = '[C]1-[C]1-[C]3-2-1='
        allowed = self.allowed(tokenizer, pending)
        self.assertIn('[C]', allowed)
        self.assertIn('[N+]', allowed)
        self.assertNotIn('[N]', allowed)
        self.assertNotIn('[O]', allowed)

    def test_whitelists_restrict_generation_but_not_encoding(self):
        tokenizer = SnackTokenizer('zinc250k', allowed_node_types=['[C]', '[O]'],
                                   allowed_edge_types=['-'])
        self.assertEqual(self.allowed(tokenizer, ''), {'[C]', '[O]'})
        self.assertEqual(self.allowed(tokenizer, '[C]1'), {'-'})
        self.assertEqual(tokenizer.decode(tokenizer.encode('[N]')), '[N]')
        with self.assertRaises(ValueError):
            tokenizer.encode('[N]', validate=True)

    def test_unsupported_chemistry_is_explicit(self):
        for kwargs in ({'edge_types': ['a']}, {'edge_types': ['(dative)']},
                       {'node_types': ['[C@H]']}, {'node_types': ['[13C]']},
                       {'node_types': ['[C;r=1]']}, {'node_types': ['[*]']}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SnackTokenizer('zinc250k', **kwargs)
        encoder = SnackTokenizer('zinc250k', node_types=['[C]'],
                                 edge_types=['(contact)'], valence=False)
        text = '[C]1(contact)[C]'
        self.assertEqual(encoder.decode(encoder.encode(text)), text)

    def test_stereochemical_inputs_convert_to_basic_vocabulary(self):
        for preset in ('zinc250k', 'moses', 'guacamol'):
            tokenizer = SnackTokenizer(preset, ordering='bfs')
            for smiles in ('F/C=C/F', 'F/C=C\\F', 'C[C@H](O)/C=C/[C@@H](F)Cl'):
                with self.subTest(preset=preset, smiles=smiles):
                    text = snack.smiles_to_snack(smiles)
                    ids = tokenizer.encode(text, validate=True)
                    expected = Chem.MolFromSmiles(smiles)
                    Chem.RemoveStereochemistry(expected)
                    self.assertEqual(snack.snack_to_smiles(tokenizer.decode(ids)),
                                     Chem.MolToSmiles(expected))

    def test_explicit_hydrogens_allow_radicals_but_limit_bond_capacity(self):
        tokenizer = SnackTokenizer('moses', node_types=['[NH]', '[C]'], max_nodes=3)
        self.assertIn('[eos]', self.allowed(tokenizer, '[NH]'))
        self.assertEqual(self.allowed(tokenizer, '[NH]1'), {'-', '='})
        tokenizer.encode('[NH]1=[C]', validate=True)
        tokenizer.encode('[NH]1-[C]', validate=True)
        with self.assertRaises(ValueError):
            tokenizer.encode('[NH]1#[C]', validate=True)
        for text, radicals in (('[NH]', 2), ('[NH]1-[C]', 1), ('[NH]1=[C]', 0)):
            mol = snack.snack_to_mol(tokenizer.decode(tokenizer.encode(text, validate=True)))
            self.assertEqual(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()), radicals)
        # Presets encode H-constrained atoms but default generation uses implicit H.
        preset = SnackTokenizer('moses')
        self.assertIn('[NH]', preset.node_types)
        self.assertNotIn('[NH]', self.allowed(preset, ''))

    def test_radical_corpus_motifs_use_existing_hydrogen_tokens(self):
        examples = ('[CH2]c1c[nH]c2ccccc12', '[CH]=CCc1c[nH]c2ccc(F)cc12',
                    '[N]=O', 'C[O]', '[B]C', '[C]', '[CH2-]')
        for node_first, ascending, hydrogens in itertools.product((False, True), repeat=3):
            snack.config.set(node_first, ascending)
            base = SnackTokenizer('guacamol')
            tokenizer = SnackTokenizer('guacamol', ordering='bfs', allowed_node_types=base.node_types)
            for smiles in examples:
                with self.subTest(smiles=smiles, node_first=node_first,
                                  ascending=ascending, hydrogens=hydrogens):
                    text = snack.smiles_to_snack(smiles, ordering='bfs', use_hydrogens=hydrogens)
                    self.assertNotIn(';r=', text)
                    ids = tokenizer.encode(text, validate=True)
                    self.assertEqual(tokenizer.decode(ids), text)
                    masks = tokenizer.get_logit_mask(ids)
                    self.assertTrue(all(masks[i][target] for i, target in enumerate(ids[1:])))
                    restored = snack.snack_to_mol(tokenizer.decode(ids))
                    original = Chem.MolFromSmiles(smiles)
                    if hydrogens:
                        self.assertEqual(Chem.MolToSmiles(restored), Chem.MolToSmiles(original))
                        self.assertEqual(sum(a.GetNumRadicalElectrons() for a in restored.GetAtoms()),
                                         sum(a.GetNumRadicalElectrons() for a in original.GetAtoms()))
                    else:
                        # Hydrogen-free output uses the default generation vocabulary.
                        base.encode(text, validate=True)

    def test_radical_at_node_limit_can_terminate_but_not_overbond(self):
        tokenizer = SnackTokenizer('guacamol', max_nodes=1, node_types=['[CH2]'])
        self.assertEqual(self.allowed(tokenizer, '[CH2]'), {'[eos]'})
        for node_first in (False, True):
            tokenizer = SnackTokenizer('guacamol', max_nodes=2, node_types=['[CH2]', '[C]'],
                                       node_first=node_first)
            text = '[CH2][C]1' if node_first else '[CH2]1'
            self.assertEqual(self.allowed(tokenizer, text), {'-', '='})
        with self.assertRaises(ValueError):
            SnackTokenizer('guacamol', node_types=['[CH5]'])

    def test_incremental_and_prefix_masks_agree(self):
        tokenizer = SnackTokenizer('moses', ordering='bfs')
        ids = tokenizer.encode('[C]1=[C]2-1-[C]', validate=True)
        state = tokenizer.new_state()
        for i, index in enumerate(ids):
            tokenizer.step(state, index)
            self.assertEqual(tokenizer.allowed_token_ids(state), tokenizer.allowed_next_ids(ids[:i + 1]))

    def test_random_generated_outputs_are_valid(self):
        for preset in ('graph', 'zinc250k', 'moses', 'guacamol'):
            for node_first, ascending, disconnected, bfs in itertools.product((False, True), repeat=4):
                tokenizer = SnackTokenizer(preset, max_nodes=6, node_first=node_first, ascending=ascending,
                                           allow_disconnected=disconnected, ordering='bfs' if bfs else None)
                snack.config.set(node_first, ascending)
                for seed in range(10):
                    with self.subTest(preset=preset, node_first=node_first, ascending=ascending,
                                      disconnected=disconnected, bfs=bfs, seed=seed):
                        rng = random.Random(seed)
                        state = tokenizer.new_state()
                        ids = []
                        for _ in range(100):
                            allowed = tokenizer.allowed_token_ids(state)
                            self.assertTrue(allowed, f'Dead end: {tokenizer.decode(ids)}')
                            index = rng.choice(allowed)
                            tokenizer.step(state, index)
                            ids.append(index)
                            if index == tokenizer.eos_id:
                                break
                        else:
                            self.fail('Generation did not terminate')
                        graph = snack.snack_to_graph(tokenizer.decode(ids))
                        self.assertLessEqual(len(graph), 6)
                        if not disconnected:
                            self.assertTrue(nx.is_connected(graph))
                        if preset != 'graph':
                            mol = snack.graph_to_mol(graph)
                            self.assertFalse(any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
                            self.assertIsNotNone(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

    def test_no_training_library_import(self):
        code = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'pandas', 'sklearn'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from snack.tokenizer import SnackTokenizer
for preset, text in [('graph','v1_v'), ('moses','[C]1-[O]'), ('guacamol','[B]1-[Si]')]:
    t = SnackTokenizer(preset)
    ids = t.encode(text, validate=True)
    assert t.decode(ids) == text
    assert t.get_logit_mask(ids)
assert not {'torch', 'pandas', 'sklearn'} & sys.modules.keys()
'''
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                cwd=Path(snack.__file__).resolve().parent.parent,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
