import re
import random
from rdkit import Chem
import networkx as nx
from snack.graph import graph_to_snack, snack_to_graph, _validate_graph


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


def mol_to_graph(mol, use_hydrogens=False, use_charge=True):
    """Convert to an undirected typed graph, omitting all stereochemistry.

    Directional bond types, including dative bonds, are unsupported.
    """
    if mol is None:
        raise ValueError('Expected an RDKit molecule, got None')
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        token = atom_to_string(atom, use_hydrogens, use_charge)
        graph.add_node(atom.GetIdx(), node_type=token)
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            edge_type = BONDTYPE_TO_SYMBOL[bond.GetBondType()]
        except KeyError as exc:
            raise ValueError(f'Unsupported bond type: {bond.GetBondType()}') from exc
        graph.add_edge(u, v, edge_type=edge_type)
    return graph


def graph_to_mol(graph):
    """Build and sanitize a molecule, raising ValueError for invalid chemistry."""
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
    except (RuntimeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid molecular graph: {exc}') from exc
    return mol.GetMol()


def smiles_to_graph(smiles, kekulize=True, use_hydrogens=False, use_charge=True):
    return mol_to_graph(smiles_to_mol(smiles, kekulize), use_hydrogens, use_charge)


def graph_to_smiles(graph, canonical=True):
    return mol_to_smiles(graph_to_mol(graph), canonical)


def smiles_to_snack(smiles, ordering='bfs', kekulize=True, use_hydrogens=False,
                    use_charge=True, *, attributed=True):
    """Encode SMILES with BFS by default; None preserves input atom order.

    All stereochemistry is omitted. Omitting hydrogen counts or charge can
    also make this lossy. BFS uses the package's randomized traversal.
    """
    graph = smiles_to_graph(smiles, kekulize, use_hydrogens, use_charge)
    return graph_to_snack(graph, ordering, attributed=attributed)


def snack_to_smiles(snack, canonical=True):
    return graph_to_smiles(snack_to_graph(snack), canonical)


def mol_to_snack(mol, ordering=None, *, use_hydrogens=False, use_charge=True,
                 attributed=True):
    """Encode a molecule; use_hydrogens=False discards all H-count constraints."""
    graph = mol_to_graph(mol, use_hydrogens, use_charge)
    return graph_to_snack(graph, ordering, attributed=attributed)


def snack_to_mol(snack):
    return graph_to_mol(snack_to_graph(snack))


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
