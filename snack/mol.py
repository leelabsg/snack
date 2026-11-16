import re
import random
from functools import lru_cache
from rdkit import Chem
import networkx as nx
from snack.graph import (graph_to_snack, snack_to_graph, _validate_graph, perm_parity,
                         _check_chirality, _check_cis_trans, _FLIP, _selected, EDGE_TYPES)


BONDTYPE_TO_SYMBOL = {
    Chem.BondType.SINGLE: '-',
    Chem.BondType.DOUBLE: '=',
    Chem.BondType.TRIPLE: '#',
    Chem.BondType.AROMATIC: 'a',
}
SYMBOL_TO_BONDTYPE = {symbol: bond for bond, symbol in BONDTYPE_TO_SYMBOL.items()}
assert tuple(SYMBOL_TO_BONDTYPE) == EDGE_TYPES

RE_ATOM_TYPE = re.compile(r'^(?P<element>[A-Z][a-z]?|\*)$')
_PERIODIC_TABLE = Chem.GetPeriodicTable()


def bond_symbol(data):
    """An edge's bond symbol, reading a missing edge_type as a single bond.

    A plain edge is written -, and decoding leaves edge_type unset so that a
    graph without edge data round-trips unchanged, so the molecular layer
    supplies the default the way an omitted SMILES bond does.
    """
    symbol = data.get('edge_type')
    return '-' if symbol is None else symbol


def atom_to_string(atom) -> str:
    """Encode an atom's element as a SNACK node type, e.g. 'Cl'."""
    return atom.GetSymbol()


@lru_cache(maxsize=None)
def _element(atom_type):
    if not isinstance(atom_type, str):
        raise ValueError(f'Invalid atom type string: {atom_type!r}')
    match = RE_ATOM_TYPE.fullmatch(atom_type)
    if match is None:
        raise ValueError(f'Invalid atom type string: {atom_type!r}')
    try:
        Chem.Atom(match['element'])
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f'Invalid atom type string: {atom_type!r}') from exc
    return match['element']


def string_to_atom(atom_type: str) -> Chem.Atom:
    """Decode a SNACK node type such as 'C' or 'Cl' into a bare RDKit atom."""
    return Chem.Atom(_element(atom_type))


def smiles_to_mol(smiles, *, kekulize=True):
    """Parse SMILES into a molecule or raise ValueError.

    Kekulized by default, so an aromatic ring is written with alternating -
    and = bonds. ``kekulize=False`` keeps RDKit's aromatic bonds, which are
    written as a: benzene is C1aC1aC1aC1aC5a1aC instead of C1=C1-C1=C1-C5-1=C.
    """
    if not isinstance(smiles, str):
        raise ValueError('SMILES must be a string')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    if not kekulize:
        return mol
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
_CHIRAL_TAGS = {'@': Chem.ChiralType.CHI_TETRAHEDRAL_CCW, '@@': Chem.ChiralType.CHI_TETRAHEDRAL_CW}
_BOND_STEREO = {'c': Chem.BondStereo.STEREOCIS, 't': Chem.BondStereo.STEREOTRANS}


_BOND_MARKS = {Chem.BondStereo.STEREOCIS: 'c', Chem.BondStereo.STEREOZ: 'c',
               Chem.BondStereo.STEREOTRANS: 't', Chem.BondStereo.STEREOE: 't'}


def _specified_stereo(mol):
    """({atom: (ligands, sign)}, {bond: (ref_begin, ref_end, mark)}) for specified stereo.

    A stereo element is a mark that RDKit's SMILES writer keeps: a chiral tag
    or double-bond stereo that survives AssignStereochemistry(cleanIt=True)
    under the active perception. Judged on the aromatic form so the verdict
    does not depend on the Kekule structure.
    """
    perceived = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(perceived)
    except (RuntimeError, ValueError):
        perceived = Chem.Mol(mol)
    Chem.AssignStereochemistry(perceived, cleanIt=True, force=True)
    atoms, bonds = {}, {}
    for atom in perceived.GetAtoms():
        sign = _TAG_SIGNS.get(atom.GetChiralTag())
        if sign is None:
            continue
        index = atom.GetIdx()
        ligands = [bond.GetOtherAtomIdx(index) for bond in mol.GetAtomWithIdx(index).GetBonds()]
        if len(ligands) in (3, 4):
            atoms[index] = (tuple(ligands + [None] * (4 - len(ligands))), sign)
    for bond in perceived.GetBonds():
        mark = _BOND_MARKS.get(bond.GetStereo())
        refs = list(bond.GetStereoAtoms())
        if mark is None or len(refs) != 2 or bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        # Stereo atoms are one neighbor of each end; E/Z and trans/cis both refer to them.
        ref_a, ref_b = refs
        if mol.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), ref_a) is None:
            ref_a, ref_b = ref_b, ref_a
        bonds[bond.GetIdx()] = (ref_a, ref_b, mark)
    return atoms, bonds


def _bare_atom(node_type, data):
    """An atom as decoding builds it before bonds: element, isotope, charge, H count.

    Radicals are never written; RDKit derives them from a given H count when
    it sanitizes the molecule, exactly as it does for a SMILES bracket atom.
    An isotope is the mass number and leaves the valence alone.
    """
    atom = string_to_atom(node_type)
    charge, hydrogens = data.get('charge', 0), data.get('hydrogens')
    isotope = data.get('isotope')
    for key, value in (('charge', charge), ('hydrogens', hydrogens), ('isotope', isotope)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f'{key} must be an integer, got {value!r}')
    if hydrogens is not None and hydrogens < 0:
        raise ValueError('hydrogens must be non-negative')
    if isotope is not None:
        if isotope < 1:
            raise ValueError('isotope must be a mass number >= 1')
        atom.SetIsotope(isotope)
    atom.SetFormalCharge(charge)
    atom.SetNoImplicit(hydrogens is not None)
    atom.SetNumExplicitHs(hydrogens or 0)
    return atom


def _implicit_hydrogens(mol):
    """Per-atom H counts RDKit infers, or None where it rejects the valence instead.

    Explicit hydrogen counts preserve radicals and unusual hydrogen states;
    RDKit still checks valence when an explicit H count is present.
    """
    from rdkit import rdBase
    mol.UpdatePropertyCache(strict=False)
    with rdBase.BlockLogs():
        failed = {problem.GetAtomIdx() for problem in Chem.DetectChemistryProblems(mol)
                  if problem.GetType() == 'AtomValenceException'}
    return [None if atom.GetIdx() in failed else atom.GetNumImplicitHs() for atom in mol.GetAtoms()]


def _default_hydrogens(node_types, data, bonds):
    """Implicit H counts decoding infers for atoms written without a hydrogens attribute."""
    mol = Chem.RWMol()
    for token, attributes in zip(node_types, data):
        mol.AddAtom(_bare_atom(token, {**attributes, 'hydrogens': None}))
    for u, v, order in bonds:
        mol.AddBond(u, v, order=order)
    return _implicit_hydrogens(mol)


def _has_stereo(mol):
    return (any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms())
            or any(bond.GetStereo() != Chem.BondStereo.STEREONONE
                   or bond.GetBondDir() in (Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT)
                   for bond in mol.GetBonds()))


def _inferred_hydrogens(mol, keep_charge):
    """H counts decoding infers when no atom carries a hydrogens attribute.

    Radicals are not encoded, so an atom with unpaired electrons (fewer
    hydrogens than its valence allows) needs a hydrogens attribute.
    """
    bare = Chem.RWMol(mol)
    for atom in bare.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)
        if not keep_charge:
            atom.SetFormalCharge(0)
    return _implicit_hydrogens(bare)


def mol_to_graph(mol, attributes=None):
    """Convert to an undirected typed graph with node types such as 'C'.

    ``attributes`` selects which of isotope, charge, hydrogens, chirality and
    cis_trans to keep (default: all). Charges and isotopes are stored when set;
    a hydrogens count is stored only where decoding would infer a different
    count, which covers radicals ([CH2]C is C(H2)1-C). Atom maps and bonds
    other than single, double, triple and aromatic are not supported and raise
    ValueError.
    """
    if mol is None:
        raise ValueError('Expected an RDKit molecule, got None')
    selected = _selected(attributes)
    graph = nx.Graph()
    bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() not in BONDTYPE_TO_SYMBOL:
            raise ValueError(f'Unsupported bond type: {bond.GetBondType()}; kekulize the molecule '
                             'or use single, double, triple or aromatic bonds')
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()))
    node_types, data = [], []
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            raise ValueError('Atom map numbers are not supported')
        node_types.append(atom_to_string(atom))
        attributes = {}
        if 'isotope' in selected and atom.GetIsotope():
            attributes['isotope'] = atom.GetIsotope()
        if 'charge' in selected and atom.GetFormalCharge():
            attributes['charge'] = atom.GetFormalCharge()
        data.append(attributes)
    if 'hydrogens' in selected:
        defaults = _inferred_hydrogens(mol, 'charge' in selected)
        for atom, attributes, default in zip(mol.GetAtoms(), data, defaults):
            if atom.GetTotalNumHs() != default:
                attributes['hydrogens'] = atom.GetTotalNumHs()
    for index, (token, attributes) in enumerate(zip(node_types, data)):
        graph.add_node(index, node_type=token, **attributes)
    for u, v, bond_type in bonds:
        graph.add_edge(u, v, edge_type=BONDTYPE_TO_SYMBOL[bond_type])
    if ('chirality' in selected or 'cis_trans' in selected) and _has_stereo(mol):
        atoms, double_bonds = _specified_stereo(mol)
        if 'chirality' in selected:
            for atom, value in atoms.items():
                graph.nodes[atom]['chirality'] = value
        if 'cis_trans' in selected:
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
    if graph is None:
        raise ValueError('Expected a networkx graph, got None')
    _validate_graph(graph)
    mol = Chem.RWMol()
    nodes = list(graph)
    rank = {node: i for i, node in enumerate(nodes)}
    try:
        for node in nodes:
            data = graph.nodes[node]
            if 'node_type' not in data:
                raise ValueError('Molecular graphs require node_type on every node')
            mol.AddAtom(_bare_atom(data['node_type'], data))
        for u, v, data in graph.edges(data=True):
            edge_type = bond_symbol(data)
            if edge_type not in SYMBOL_TO_BONDTYPE:
                raise ValueError(f'Molecular graphs require edge_type -, =, # or a on every '
                                 f'edge, got {edge_type!r}')
            mol.AddBond(rank[u], rank[v], order=SYMBOL_TO_BONDTYPE[edge_type])
        Chem.SanitizeMol(mol)
        has_stereo = _apply_stereo(graph, mol, nodes, rank, stereo)
    except (RuntimeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid molecular graph: {exc}') from exc
    return _normalize_stereo(mol.GetMol()) if has_stereo else mol.GetMol()


def _normalize_stereo(mol):
    """Reparse through SMILES, keeping atom order, so MolToSmiles matches a parsed molecule.

    Stereo flags set by AssignStereochemistry differ from those of a freshly
    parsed molecule and can change RDKit's canonical SMILES for the same
    stereoisomer (e.g. ring pseudo-asymmetric centers).
    """
    try:
        smiles = Chem.MolToSmiles(mol)
    except RuntimeError:
        return mol
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
    chirality = {node: _check_chirality(graph, node)
                 for node, data in graph.nodes(data=True) if 'chirality' in data}
    cis_trans = {(u, v): _check_cis_trans(graph, u, v)
                 for u, v, data in graph.edges(data=True) if 'cis_trans' in data}
    if not chirality and not cis_trans:
        return False
    for node, (ligands, sign) in chirality.items():
        atom = mol.GetAtomWithIdx(rank[node])
        # RDKit tags refer to bond order, with an implicit ligand last.
        order = [nodes[bond.GetOtherAtomIdx(rank[node])] for bond in atom.GetBonds()]
        order += [None] * (4 - len(order))
        atom.SetChiralTag(_CHIRAL_TAGS[sign if perm_parity(ligands, order) == 1 else _FLIP[sign]])
    requested_bonds = set()
    for (u, v), (refs, mark) in cis_trans.items():
        # Only a double bond whose substituents hang off single or aromatic
        # bonds has cis/trans stereo; cumulated double bonds have axial stereo
        # instead, which RDKit cannot write with / and \\.
        if bond_symbol(graph.edges[u, v]) != '=' or any(
                bond_symbol(graph.edges[end, other]) not in ('-', 'a')
                for end, mate in ((u, v), (v, u)) for other in graph[end] if other != mate):
            if mode == 'strict':
                raise ValueError(f'Cis/trans annotation on edge {(u, v)!r} requires single bonds '
                                 'to the substituents')
            continue
        bond = mol.GetBondBetweenAtoms(rank[u], rank[v])
        begin, end = nodes[bond.GetBeginAtomIdx()], nodes[bond.GetEndAtomIdx()]
        bond.SetStereoAtoms(rank[refs[begin]], rank[refs[end]])
        bond.SetStereo(_BOND_STEREO[mark])
        requested_bonds.add(bond.GetIdx())
    if requested_bonds:
        # Stereo perception recomputes double bonds from / and \\ directions, not flags.
        Chem.SetDoubleBondNeighborDirections(mol)
    atoms, bonds = _specified_stereo(mol)
    if mode == 'strict':
        lost = sorted(nodes[i] for i in {rank[n] for n in chirality} - set(atoms))
        lost_bonds = sorted(requested_bonds - set(bonds))
        if lost or lost_bonds:
            raise ValueError(f'Stereo annotations on non-stereogenic nodes {lost} or '
                             f'bonds {lost_bonds}')
    # cleanIt removes tags on atoms and bonds that are not stereogenic ('clean' mode).
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return True


def _redundant_hydrogens(graph):
    """Nodes whose hydrogens attribute equals the count decoding would infer."""
    nodes = [node for node, data in graph.nodes(data=True) if data.get('hydrogens') is not None]
    if not nodes:
        return []
    order = list(graph)
    rank = {node: i for i, node in enumerate(order)}
    tokens = [graph.nodes[node]['node_type'] for node in order]
    data = [graph.nodes[node] for node in order]
    bonds = [(rank[u], rank[v], SYMBOL_TO_BONDTYPE[bond_symbol(attributes)])
             for u, v, attributes in graph.edges(data=True)]
    defaults = _default_hydrogens(tokens, data, bonds)
    return [node for node in nodes if graph.nodes[node]['hydrogens'] == defaults[rank[node]]]


def is_canonical_snack(snack):
    """Whether every stereo annotation and hydrogens attribute is necessary."""
    try:
        graph = snack_to_graph(snack)
        graph_to_mol(graph, stereo='strict')
    except ValueError:
        return False
    return not _redundant_hydrogens(graph)


def smiles_to_graph(smiles, attributes=None, *, kekulize=True):
    return mol_to_graph(smiles_to_mol(smiles, kekulize=kekulize), attributes)


def graph_to_smiles(graph, canonical=True, *, stereo='clean'):
    return mol_to_smiles(graph_to_mol(graph, stereo=stereo), canonical)


def smiles_to_snack(smiles, ordering='bfs', *, attributes=None, attributed=True, kekulize=True):
    """Encode SMILES with randomized BFS by default; None preserves input atom order.

    ``attributes`` selects which of isotope, charge, hydrogens, chirality and
    cis_trans to write (default: all). Omitting charge or hydrogens can change
    the molecule's identity or prevent decoding. ``kekulize=False`` keeps
    aromatic bonds and writes them as a.
    """
    graph = smiles_to_graph(smiles, attributes, kekulize=kekulize)
    return graph_to_snack(graph, ordering, attributed=attributed)


def snack_to_smiles(snack, canonical=True, *, stereo='clean'):
    return graph_to_smiles(snack_to_graph(snack, strict=stereo == 'strict'), canonical, stereo=stereo)


def mol_to_snack(mol, ordering=None, *, attributes=None, attributed=True):
    """Encode a molecule, preserving its atom order unless ordering is given."""
    graph = mol_to_graph(mol, attributes)
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
