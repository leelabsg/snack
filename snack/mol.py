import re
import random
from contextlib import contextmanager
from rdkit import Chem
import networkx as nx
from snack.graph import (graph_to_snack, snack_to_graph, _validate_graph, perm_parity,
                         _check_tetra, _check_cis_trans, _FLIP)


SINGLE_SYMBOL = '-'
DOUBLE_SYMBOL = '='
TRIPLE_SYMBOL = '#'
AROMATIC_SYMBOL = 'a'
UNSPECIFIED_SYMBOL = 'u'
QUADRUPLE_SYMBOL = 'quad'
QUINTUPLE_SYMBOL = 'quint'
HEXTUPLE_SYMBOL = 'hex'
ONEANDAHALF_SYMBOL = 'onehalf'
TWOANDAHALF_SYMBOL = 'twohalf'
THREEANDAHALF_SYMBOL = 'threehalf'
FOURANDAHALF_SYMBOL = 'fourhalf'
FIVEANDAHALF_SYMBOL = 'fivehalf'
IONIC_SYMBOL = 'ionic'
HYDROGEN_SYMBOL = 'hyd'
THREECENTER_SYMBOL = 'threecenter'
OTHER_SYMBOL = 'oth'
ZERO_SYMBOL = 'zero'

BONDTYPE_TO_SYMBOL = {
    Chem.BondType.SINGLE: SINGLE_SYMBOL, 
    Chem.BondType.DOUBLE: DOUBLE_SYMBOL, 
    Chem.BondType.TRIPLE: TRIPLE_SYMBOL,
    Chem.BondType.AROMATIC: AROMATIC_SYMBOL,
    Chem.BondType.UNSPECIFIED: UNSPECIFIED_SYMBOL,
    Chem.BondType.QUADRUPLE: QUADRUPLE_SYMBOL,
    Chem.BondType.QUINTUPLE: QUINTUPLE_SYMBOL,
    Chem.BondType.HEXTUPLE: HEXTUPLE_SYMBOL,
    Chem.BondType.ONEANDAHALF: ONEANDAHALF_SYMBOL,
    Chem.BondType.TWOANDAHALF: TWOANDAHALF_SYMBOL,
    Chem.BondType.THREEANDAHALF: THREEANDAHALF_SYMBOL,
    Chem.BondType.FOURANDAHALF: FOURANDAHALF_SYMBOL,
    Chem.BondType.FIVEANDAHALF: FIVEANDAHALF_SYMBOL,
    Chem.BondType.IONIC: IONIC_SYMBOL,
    Chem.BondType.HYDROGEN: HYDROGEN_SYMBOL,
    Chem.BondType.THREECENTER: THREECENTER_SYMBOL,
    Chem.BondType.OTHER: OTHER_SYMBOL,
    Chem.BondType.ZERO: ZERO_SYMBOL,

}
SYMBOL_TO_BONDTYPE = {
    SINGLE_SYMBOL: Chem.BondType.SINGLE, 
    DOUBLE_SYMBOL: Chem.BondType.DOUBLE, 
    TRIPLE_SYMBOL: Chem.BondType.TRIPLE,
    AROMATIC_SYMBOL: Chem.BondType.AROMATIC,
    UNSPECIFIED_SYMBOL: Chem.BondType.UNSPECIFIED,
    QUADRUPLE_SYMBOL: Chem.BondType.QUADRUPLE,
    QUINTUPLE_SYMBOL: Chem.BondType.QUINTUPLE,
    HEXTUPLE_SYMBOL: Chem.BondType.HEXTUPLE,
    ONEANDAHALF_SYMBOL: Chem.BondType.ONEANDAHALF,
    TWOANDAHALF_SYMBOL: Chem.BondType.TWOANDAHALF,
    THREEANDAHALF_SYMBOL: Chem.BondType.THREEANDAHALF,
    FOURANDAHALF_SYMBOL: Chem.BondType.FOURANDAHALF,
    FIVEANDAHALF_SYMBOL: Chem.BondType.FIVEANDAHALF,
    IONIC_SYMBOL: Chem.BondType.IONIC,
    HYDROGEN_SYMBOL: Chem.BondType.HYDROGEN,
    THREECENTER_SYMBOL: Chem.BondType.THREECENTER,
    OTHER_SYMBOL: Chem.BondType.OTHER,
    ZERO_SYMBOL: Chem.BondType.ZERO,
}

# H/H0 disables implicit hydrogens. Radical counts are inferred from hydrogen
# counts, charge and bonding during molecule sanitization.
RE_ATOM_TYPE = re.compile(
    r'^(?P<isotope>[1-9][0-9]*)?(?P<element>[A-Z][a-z]?|\*)'
    r'(?:H(?P<hcount>[0-9]*))?'
    r'(?P<charge>\+\+|--|[+-][0-9]*)?'
    r'(?::(?P<map>[0-9]+))?$')


def atom_to_string(atom, use_hydrogens=False, use_charge=True):
    """Encode an atom with optional hydrogen counts and charge.

    use_hydrogens=False omits all H counts, including explicit H and H0.
    This can change the molecule's identity on decoding. True writes every count.
    Radical counts are inferred during molecule sanitization, not serialized.
    Atomic stereochemistry is omitted, including @/@@ markers.
    """
    isotope = atom.GetIsotope()
    symbol = (str(isotope) if isotope else '') + atom.GetSymbol()
    if use_hydrogens:
        num_h = atom.GetTotalNumHs()
        symbol += 'H' if num_h == 1 else f'H{num_h}'
    charge = atom.GetFormalCharge()
    if use_charge and charge:
        symbol += ('+' if charge > 0 else '-') + (str(abs(charge)) if abs(charge) != 1 else '')
    if atom.GetAtomMapNum():
        symbol += f':{atom.GetAtomMapNum()}'
    return f'[{symbol}]'


def string_to_atom(atom_type: str) -> Chem.Atom:
    """Decode a bracketed SNACK atom; H/H0 disables implicit H inference."""
    if not isinstance(atom_type, str) or not atom_type.startswith('[') or not atom_type.endswith(']'):
        raise ValueError(f'Invalid atom type string: {atom_type!r}')
    match = RE_ATOM_TYPE.fullmatch(atom_type[1:-1])
    if match is None:
        raise ValueError(f'Invalid atom type string: {atom_type!r}')
    try:
        atom = Chem.Atom(match['element'])
        if match['isotope']:
            atom.SetIsotope(int(match['isotope']))
        hydrogens = match['hcount']
        atom.SetNoImplicit(hydrogens is not None)
        if hydrogens is not None:
            atom.SetNumExplicitHs(int(hydrogens) if hydrogens else 1)
        charge = match['charge']
        if charge in ('+', '++', '-', '--'):
            charge = charge.count('+') - charge.count('-')
        atom.SetFormalCharge(int(charge) if charge else 0)
        if match['map']:
            atom.SetAtomMapNum(int(match['map']))
    except (RuntimeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid atom type string: {atom_type!r}') from exc
    return atom


def smiles_to_mol(smiles, kekulize=True):
    """Parse SMILES or raise ValueError, consistently in both aromatic modes."""
    if not isinstance(smiles, str):
        raise ValueError('SMILES must be a string')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    if kekulize:
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f'Cannot kekulize SMILES: {smiles!r}') from exc
    return mol


def mol_to_smiles(mol, canonical=True):
    if mol is None:
        raise ValueError('Expected an RDKit molecule, got None')
    return Chem.MolToSmiles(mol, canonical=canonical)


_TAG_SIGNS = {Chem.ChiralType.CHI_TETRAHEDRAL_CCW: '@', Chem.ChiralType.CHI_TETRAHEDRAL_CW: '@@'}
_BOND_DESCRIPTORS = {Chem.StereoDescriptor.Bond_Cis: 'c', Chem.StereoDescriptor.Bond_Trans: 't'}
_CHIRAL_TAGS = {'@': Chem.ChiralType.CHI_TETRAHEDRAL_CCW, '@@': Chem.ChiralType.CHI_TETRAHEDRAL_CW}
_BOND_STEREO = {'c': Chem.BondStereo.STEREOCIS, 't': Chem.BondStereo.STEREOTRANS}


@contextmanager
def _stereo_perception():
    """Pin RDKit's modern stereo perception, restoring the caller's setting."""
    legacy = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(False)
    try:
        yield
    finally:
        Chem.SetUseLegacyStereoPerception(legacy)


def _specified_stereo(mol):
    """({atom: (ligands, sign)}, {bond: (ref_begin, ref_end, mark)}) for specified stereo."""
    # Judge on the aromatic form so the verdict does not depend on the Kekule structure.
    perceived = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(perceived)
    except (RuntimeError, ValueError):
        perceived = Chem.Mol(mol)
    with _stereo_perception():
        infos = Chem.FindPotentialStereo(perceived, cleanIt=True)
    num_atoms = mol.GetNumAtoms()
    atoms, bonds = {}, {}
    for info in infos:
        if info.specified != Chem.StereoSpecified.Specified:
            continue
        # Missing ligands come back as an out-of-range index.
        controlling = [a if a < num_atoms else None for a in info.controllingAtoms]
        if info.type == Chem.StereoType.Atom_Tetrahedral:
            # The FindPotentialStereo descriptor can disagree with the tag on ring atoms.
            atom = mol.GetAtomWithIdx(info.centeredOn)
            sign = _TAG_SIGNS.get(atom.GetChiralTag())
            ligands = [bond.GetOtherAtomIdx(atom.GetIdx()) for bond in atom.GetBonds()]
            if sign is not None and len(ligands) in (3, 4):
                atoms[atom.GetIdx()] = (tuple(ligands + [None] * (4 - len(ligands))), sign)
        elif info.type == Chem.StereoType.Bond_Double and info.descriptor in _BOND_DESCRIPTORS:
            bond = mol.GetBondWithIdx(info.centeredOn)
            # Two slots per end; the descriptor compares slot 0 with slot 2.
            ref_a, ref_b = controlling[0], controlling[2]
            if mol.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), ref_a) is None:
                ref_a, ref_b = ref_b, ref_a
            bonds[info.centeredOn] = (ref_a, ref_b, _BOND_DESCRIPTORS[info.descriptor])
    return atoms, bonds


def _default_hydrogens(node_types, bonds):
    """Implicit H counts SNACK decoding infers for H-free versions of atoms."""
    mol = Chem.RWMol()
    for token in node_types:
        atom = string_to_atom(token)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        mol.AddAtom(atom)
    for u, v, order in bonds:
        mol.AddBond(u, v, order=order)
    mol.UpdatePropertyCache(strict=False)
    return [atom.GetNumImplicitHs() for atom in mol.GetAtoms()]


def mol_to_graph(mol, use_hydrogens=False, use_charge=True, *, stereo=False):
    """Convert to an undirected typed graph. Directional bond types are unsupported.

    use_hydrogens='auto' writes H counts only where they differ from the
    inferred count (e.g. radicals). stereo=True keeps 'tetra'/'cis_trans'.
    """
    if mol is None:
        raise ValueError('Expected an RDKit molecule, got None')
    graph = nx.Graph()
    bonds = []
    for bond in mol.GetBonds():
        try:
            BONDTYPE_TO_SYMBOL[bond.GetBondType()]
        except KeyError as exc:
            raise ValueError(f'Unsupported bond type: {bond.GetBondType()}') from exc
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()))
    if use_hydrogens == 'auto':
        bare = [atom_to_string(atom, False, use_charge) for atom in mol.GetAtoms()]
        defaults = _default_hydrogens(bare, bonds)
        write_h = [atom.GetTotalNumHs() != default for atom, default in zip(mol.GetAtoms(), defaults)]
    else:
        write_h = [bool(use_hydrogens)] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        token = atom_to_string(atom, write_h[atom.GetIdx()], use_charge)
        graph.add_node(atom.GetIdx(), node_type=token)
    for u, v, bond_type in bonds:
        graph.add_edge(u, v, edge_type=BONDTYPE_TO_SYMBOL[bond_type])
    if stereo:
        atoms, double_bonds = _specified_stereo(mol)
        for atom, value in atoms.items():
            graph.nodes[atom]['tetra'] = value
        for index, (ref_a, ref_b, mark) in double_bonds.items():
            bond = mol.GetBondWithIdx(index)
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            graph.edges[u, v]['cis_trans'] = ((u, ref_a), (v, ref_b), mark)
    return graph


def graph_to_mol(graph, *, stereo='clean'):
    """Build and sanitize a molecule, raising ValueError for invalid chemistry.

    stereo='clean' drops annotations that are not stereo elements; 'strict' raises.
    """
    if stereo not in ('clean', 'strict'):
        raise ValueError(f"stereo must be 'clean' or 'strict', got {stereo!r}")
    _validate_graph(graph)
    mol = Chem.RWMol()
    nodes = list(graph)
    rank = {node: i for i, node in enumerate(nodes)}
    try:
        for node in nodes:
            if 'node_type' not in graph.nodes[node]:
                raise ValueError('Molecular graphs require node_type on every node')
            mol.AddAtom(string_to_atom(graph.nodes[node]['node_type']))
        for u, v, data in graph.edges(data=True):
            if 'weight' in data:
                raise ValueError('Numeric edge weights cannot be converted into molecular bonds')
            edge_type = data.get('edge_type')
            if not isinstance(edge_type, str):
                raise ValueError('Molecular graphs require edge_type on every edge')
            if edge_type not in SYMBOL_TO_BONDTYPE:
                raise ValueError(f'Unsupported molecular edge type: {edge_type!r}')
            mol.AddBond(rank[u], rank[v], order=SYMBOL_TO_BONDTYPE[edge_type])
        Chem.SanitizeMol(mol)
        has_cis_trans = _apply_stereo(graph, mol, nodes, rank, stereo)
    except (RuntimeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid molecular graph: {exc}') from exc
    return _normalize_bond_stereo(mol.GetMol()) if has_cis_trans else mol.GetMol()


def _normalize_bond_stereo(mol):
    """Reparse through SMILES, keeping atom order, so cis/trans gives canonical SMILES."""
    smiles = Chem.MolToSmiles(mol)
    order = [int(i) for i in mol.GetProp('_smilesAtomOutputOrder').strip('[]').split(',') if i]
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None or parsed.GetNumAtoms() != mol.GetNumAtoms() or len(order) != len(set(order)):
        return mol
    # order[i] is the original atom written i-th; invert it to restore the original order.
    inverse = [0] * len(order)
    for position, atom in enumerate(order):
        inverse[atom] = position
    return Chem.RenumberAtoms(parsed, inverse)


def _apply_stereo(graph, mol, nodes, rank, mode):
    tetra = {node: _check_tetra(graph, node) for node, data in graph.nodes(data=True) if 'tetra' in data}
    cis_trans = {(u, v): _check_cis_trans(graph, u, v)
                 for u, v, data in graph.edges(data=True) if 'cis_trans' in data}
    if not tetra and not cis_trans:
        return False
    for node, (ligands, sign) in tetra.items():
        atom = mol.GetAtomWithIdx(rank[node])
        # RDKit tags refer to bond order, with an implicit ligand last.
        order = [nodes[bond.GetOtherAtomIdx(rank[node])] for bond in atom.GetBonds()]
        order += [None] * (4 - len(order))
        atom.SetChiralTag(_CHIRAL_TAGS[sign if perm_parity(ligands, order) == 1 else _FLIP[sign]])
    requested_bonds = set()
    for (u, v), (refs, mark) in cis_trans.items():
        bond = mol.GetBondBetweenAtoms(rank[u], rank[v])
        begin, end = nodes[bond.GetBeginAtomIdx()], nodes[bond.GetEndAtomIdx()]
        bond.SetStereoAtoms(rank[refs[begin]], rank[refs[end]])
        bond.SetStereo(_BOND_STEREO[mark])
        requested_bonds.add(bond.GetIdx())
    atoms, bonds = _specified_stereo(mol)
    if mode == 'strict':
        lost = sorted(nodes[i] for i in {rank[n] for n in tetra} - set(atoms))
        lost_bonds = sorted(requested_bonds - set(bonds))
        if lost or lost_bonds:
            raise ValueError(f'Stereo annotations on non-stereogenic nodes {lost} or '
                             f'bonds {lost_bonds}')
    # cleanIt removes tags on atoms and bonds that are not stereogenic ('clean' mode).
    with _stereo_perception():
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return bool(cis_trans)


def _redundant_hydrogens(graph):
    """Nodes whose explicit H count equals the count decoding would infer."""
    nodes = list(graph)
    rank = {node: i for i, node in enumerate(nodes)}
    tokens = [graph.nodes[node]['node_type'] for node in nodes]
    explicit = [RE_ATOM_TYPE.fullmatch(token[1:-1]) for token in tokens]
    if not any(match is not None and match['hcount'] is not None for match in explicit):
        return []
    bonds = [(rank[u], rank[v], SYMBOL_TO_BONDTYPE[data['edge_type']])
             for u, v, data in graph.edges(data=True)]
    defaults = _default_hydrogens(tokens, bonds)
    return [node for node, match, default in zip(nodes, explicit, defaults)
            if match is not None and match['hcount'] is not None
            and int(match['hcount'] or 1) == default]


def is_canonical_snack(snack):
    """Whether every stereo annotation and explicit H count is necessary."""
    try:
        graph = snack_to_graph(snack)
        graph_to_mol(graph, stereo='strict')
    except ValueError:
        return False
    return not _redundant_hydrogens(graph)


def smiles_to_graph(smiles, kekulize=True, use_hydrogens=False, use_charge=True, *,
                    stereo=False):
    return mol_to_graph(smiles_to_mol(smiles, kekulize), use_hydrogens, use_charge, stereo=stereo)


def graph_to_smiles(graph, canonical=True, *, stereo='clean'):
    return mol_to_smiles(graph_to_mol(graph, stereo=stereo), canonical)


def smiles_to_snack(smiles, ordering='bfs', kekulize=True, use_hydrogens=False,
                    use_charge=True, *, attributed=True, stereo=False):
    """Encode SMILES with BFS by default; None preserves input atom order.

    Stereochemistry is omitted unless stereo=True. Omitting hydrogen counts
    or charge can also make this lossy; use_hydrogens='auto' keeps radicals.
    BFS uses the package's randomized traversal.
    """
    graph = smiles_to_graph(smiles, kekulize, use_hydrogens, use_charge, stereo=stereo)
    return graph_to_snack(graph, ordering, attributed=attributed)


def snack_to_smiles(snack, canonical=True, *, stereo='clean'):
    return graph_to_smiles(snack_to_graph(snack, strict=stereo == 'strict'), canonical, stereo=stereo)


def mol_to_snack(mol, ordering=None, *, use_hydrogens=False, use_charge=True,
                 attributed=True, stereo=False):
    """Encode a molecule; use_hydrogens=False discards all H-count constraints."""
    graph = mol_to_graph(mol, use_hydrogens, use_charge, stereo=stereo)
    return graph_to_snack(graph, ordering, attributed=attributed)


def snack_to_mol(snack, *, stereo='clean'):
    """Decode; stereo='clean' drops meaningless annotations, 'strict' raises."""
    return graph_to_mol(snack_to_graph(snack, strict=stereo == 'strict'), stereo=stereo)


def randomize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_indices = list(range(mol.GetNumAtoms()))
    random.shuffle(atom_indices)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol, atom_indices), canonical=False)


def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)
