"""SNACK tokenizer and next-token masks; no torch/numpy/pandas imports.

Masks are vocabulary/logit masks (True = allowed), not transformer attention
matrices. The molecular vocabulary covers every element, the bonds -, = and #,
and the attribute tokens for isotope, charge, hydrogens, chirality and
cis/trans. The ranges of the numeric attributes and the default node limit
cover everything observed in PubChem and ZINC15 (see docs/dataset_scan.md),
except isotopes, whose range covers the whole periodic table.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral

from snack import graph as _graph

PRESETS = ('graph', 'mol', 'qm9', 'moses')
MAX_NODES = {'graph': 128, 'mol': 128, 'qm9': 9, 'moses': 26}
# Dataset presets, measured over the whole dataset with
# scripts/fit_tokenizer.py: QM9 is 133,885 molecules of at most 9 heavy atoms
# over C, F, N, O, net neutral but with charged atoms in 1,847 of them; MOSES
# is 1,936,962 molecules of at most 26 atoms over Br, C, Cl, F, N, O, S with
# no charges and no stereochemistry. Everything else follows 'mol'.
DATASETS = {
    'qm9': {'node_types': ('C', 'F', 'N', 'O'), 'attributes': ('charge',),
            'charges': (-1, 1)},
    'moses': {'node_types': ('Br', 'C', 'Cl', 'F', 'N', 'O', 'S'), 'attributes': ()},
}
CHARGES = tuple(range(-8, 0)) + tuple(range(1, 9))
MAX_HYDROGENS = 8
# One token per mass number, up to the heaviest known nuclide, so every isotope
# of every element in the vocabulary can be written. Isotope labels are rare
# (about one molecule in ten thousand in ZINC-22), so a vocabulary fitted to a
# corpus should narrow this or drop the kind entirely.
MAX_ISOTOPE = 294
UNLIMITED = 1 << 16
# Bond orders for the valence mask. An aromatic bond is 1.5 in truth, and
# counting it as 1 keeps the arithmetic integral and the mask permissive
# rather than wrongly restrictive; the sanitization check at EOS is what
# rejects an aromatic ring that no kekule structure supports.
_BOND_ORDERS = {'-': 1, '=': 2, '#': 3, 'a': 1}
# Molecular vocabularies list the kekule bonds. Aromaticity is a property of
# a whole ring system, so a mask that looks at one node at a time cannot
# promise a kekulizable ring: an aromatic bond that no kekule structure
# supports never sanitizes, EOS stays blocked and generation dead-ends. Pass
# edge_types=('-', '=', '#', 'a') to model the aromatic form anyway, and
# handle an empty allowed set as a dead end.
DEFAULT_EDGE_TYPES = ('-', '=', '#')


def elements():
    """Every element symbol as a node type, plus the dummy atom *."""
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    return tuple(table.GetElementSymbol(z) for z in range(1, 119)) + ('*',)


def attribute_tokens(attributes=None, *, charges=CHARGES, max_hydrogens=MAX_HYDROGENS,
                     max_isotope=MAX_ISOTOPE):
    """Attribute tokens for the selected attribute kinds, in vocabulary order.

    Kinds follow the order in which they are written after a node (isotope,
    charge, hydrogens, chirality), then the cis/trans bond tokens.
    """
    selected = _graph._selected(attributes)
    values = {
        'isotope': list(range(1, max_isotope + 1)),
        'charge': [q for q in sorted(charges) if q],
        'hydrogens': list(range(max_hydrogens + 1)),
        'chirality': ['@', '@@'],
        'cis_trans': ['c', 't'],
    }
    return [_graph.attribute_token(key, value) for key in _graph.ATTRIBUTES if key in selected
            for value in values[key]]


def basic_vocabulary(preset='mol', max_nodes=None, max_distance=None, *, attributes=None):
    """Return stable tokens, including [pad], [bos], [eos], for a preset.

    'mol' lists every element, distances, the bonds -, =, # and a and the
    attribute tokens; 'graph' lists o, the distances and the plain edge -.
    'qm9' and 'moses' are 'mol' narrowed to what those datasets contain.
    """
    if preset not in PRESETS:
        raise ValueError(f'preset must be one of {PRESETS}')
    if attributes is not None and preset == 'graph':
        raise ValueError('Attributes require the mol preset')
    dataset = DATASETS.get(preset)
    ranges = {}
    if dataset is not None:
        if attributes is None:
            attributes = dataset['attributes']
        ranges = {key: dataset[key] for key in ('charges', 'max_hydrogens', 'max_isotope')
                  if key in dataset}
    maximum = MAX_NODES[preset] if max_nodes is None else max_nodes
    if not isinstance(maximum, Integral) or isinstance(maximum, bool) or maximum < 1:
        raise ValueError('max_nodes must be a positive integer')
    distance = maximum - 1 if max_distance is None else max_distance
    if not isinstance(distance, Integral) or isinstance(distance, bool) or not 0 <= distance < maximum:
        raise ValueError('max_distance must be between 0 and max_nodes - 1')
    if preset == 'graph':
        return (['[pad]', '[bos]', '[eos]', _graph.DEFAULT_NODE_TOKEN]
                + [str(i) for i in range(1, distance + 1)] + [_graph.PLAIN_EDGE])
    node_types = list(elements()) if dataset is None else list(dataset['node_types'])
    return (['[pad]', '[bos]', '[eos]'] + node_types + [str(i) for i in range(1, distance + 1)]
            + list(DEFAULT_EDGE_TYPES) + attribute_tokens(attributes, **ranges))


@lru_cache(maxsize=None)
def capacity(node_type, charge=0, lenient=False):
    """Largest bond-order sum RDKit accepts for an element with a formal charge.

    None means the charge state is impossible; UNLIMITED means RDKit imposes
    no valence limit (metals, dummy atoms). By default the atom's hydrogens
    are inferred, as decoding does for a node without a hydrogens attribute.
    ``lenient=True`` is the limit RDKit applies to an atom whose H count is
    given (a node with a hydrogens attribute): bonds and hydrogens count
    against it, and noble gases are unlimited.
    """
    from rdkit import Chem, rdBase
    from snack.mol import string_to_atom
    best = None
    for total in range(0, 13):
        atom = string_to_atom(node_type)
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(lenient)
        mol = Chem.RWMol()
        if lenient:
            atom.SetNumExplicitHs(total)
            mol.AddAtom(atom)
        else:
            mol.AddAtom(atom)
            for i in range(total):
                mol.AddAtom(Chem.Atom(1))
                mol.AddBond(0, i + 1, Chem.BondType.SINGLE)
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError):
            continue
        best = total
    return UNLIMITED if best == 12 else best


@dataclass
class _State:
    started: bool = False
    ended: bool = False
    nodes: list = field(default_factory=list)      # node tokens
    charge: list = field(default_factory=list)     # per node, 0 unless (+q) was given
    hydrogens: list = field(default_factory=list)  # per node, None unless (Hn) was given
    used: list = field(default_factory=list)       # bond-order sum per node
    degree: list = field(default_factory=list)
    edges: list = field(default_factory=list)      # (node, target, edge)
    block: dict = field(default_factory=dict)      # targets of the current node's edges
    pending: int | None = None                     # target awaiting its bond token
    last_distance: int | None = None
    bfs_floor: int = 0
    last: tuple | None = None                      # ('node', index) | ('edge', (node, target))
    attribute_rank: int = -1                       # last node attribute written

    def copy(self):
        return _State(self.started, self.ended, self.nodes[:], self.charge[:], self.hydrogens[:],
                      self.used[:], self.degree[:], self.edges[:], dict(self.block),
                      self.pending, self.last_distance, self.bfs_floor, self.last, self.attribute_rank)


class SnackTokenizer:
    """Encode/decode SNACK and constrain autoregressive generation.

    ``ordering='bfs'`` enforces a possible BFS traversal. ``None`` imposes no
    traversal constraint. Connected generation requires a backward edge on
    every node after the first (connected prefixes). Disconnected BFS starts
    a new, contiguous component whenever a node has no backward edges.

    ``node_first`` and ``ascending`` default to current SNACK settings, captured
    at construction. ``valence`` defaults to True for the mol preset: bonds
    and hydrogens may not exceed the RDKit valence of each atom and
    charge (a node with a hydrogens attribute gets RDKit's limit for atoms
    with a given H count, which is unlimited for noble gases), and EOS
    requires RDKit sanitization to succeed. ``attributes``
    selects which attribute kinds are in the vocabulary (default: all).
    Optional node_types/edge_types replace the preset vocabulary; optional
    allowed_node_types/allowed_edge_types restrict generation only. Chirality
    and cis/trans tokens are permitted wherever the structure might support
    them; decoding drops the ones that are not stereogenic.
    """
    def __init__(self, preset='mol', *, max_nodes=None, max_distance=None,
                 ordering=None, allow_disconnected=False, node_first=None, ascending=None,
                 valence=None, node_types=None, edge_types=None, attributes=None,
                 allowed_node_types=None, allowed_edge_types=None):
        base = basic_vocabulary(preset, max_nodes, max_distance, attributes=attributes)
        self.preset = preset
        self.max_nodes = MAX_NODES[preset] if max_nodes is None else int(max_nodes)
        self.max_distance = self.max_nodes - 1 if max_distance is None else int(max_distance)
        if ordering not in (None, 'bfs'):
            raise ValueError('Only ordering=None or ordering="bfs" has a generation mask')
        self.ordering = ordering
        self.allow_disconnected = allow_disconnected
        self.node_first = _graph.NODE_FIRST if node_first is None else node_first
        self.ascending = _graph.ASCENDING if ascending is None else ascending
        self.molecular = preset != 'graph'
        self.valence = self.molecular if valence is None else valence
        if self.valence and not self.molecular:
            raise ValueError('Valence masks require the mol preset')
        self.node_types = tuple(node_types) if node_types is not None else tuple(
            token for token in base[3:] if _graph.token_type(token) == 'node_type')
        self.edge_types = tuple(edge_types) if edge_types is not None else tuple(
            token for token in base[3:] if _graph.token_type(token) == 'edge_type')
        if not self.node_types or any(_graph.token_type(token) != 'node_type' for token in self.node_types):
            raise ValueError('node_types must contain valid SNACK node tokens')
        if any(_graph.token_type(token) != 'edge_type' for token in self.edge_types):
            raise ValueError('edge_types must contain -, =, # or a')
        if not self.edge_types:
            raise ValueError('Every edge needs a type token: -, =, # or a')
        if not self.molecular and self.edge_types != (_graph.PLAIN_EDGE,):
            raise ValueError(f'Graph mode uses the plain edge {_graph.PLAIN_EDGE} alone')
        distances = [token for token in base if _graph.token_type(token) == 'distance']
        self.attribute_types = tuple(token for token in base if _graph.token_type(token) == 'attribute')
        self.tokens = (tuple(base[:3]) + self.node_types + tuple(distances) + self.edge_types
                       + self.attribute_types)
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError('Vocabulary must not contain duplicate or reserved tokens')
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.eos_id = range(3)
        self._nodes = self._restrict(self.node_types, allowed_node_types, 'node')
        self._edges = self._restrict(self.edge_types, allowed_edge_types, 'edge')
        if self.valence:
            from snack.mol import _element
            for token in self._nodes:
                _element(token)
            if any(edge not in _BOND_ORDERS for edge in self._edges):
                raise ValueError('Valence masking supports only -, = and #')
        self._node_ids = [self.token_to_id[token] for token in self._nodes]
        self._edge_ids = [self.token_to_id[token] for token in self._edges]
        self._distance_ids = [self.token_to_id[token] for token in distances]
        # Parsed attribute tokens grouped by kind: {kind: [(id, value), ...]}.
        self._attributes = {}
        for token in self.attribute_types:
            key, value = _graph.parse_attribute(token)
            self._attributes.setdefault(key, []).append((self.token_to_id[token], value))
        self._charges = (0,) + tuple(value for _, value in self._attributes.get('charge', ()))
        # Nodes may take RDKit's limit for a given H count once a hydrogens token is written.
        self._lenient = self.valence and 'hydrogens' in self._attributes
        # Capacity over every charge in the vocabulary, per allowed node type, and the
        # most permissive one, for edges written before their node.
        self._capacities = {token: self._max_capacity(token) if self.valence else UNLIMITED
                            for token in self._nodes}
        self._max_node_capacity = max(self._capacities.values(), default=0)

    @staticmethod
    def _restrict(vocabulary, allowed, kind):
        values = tuple(vocabulary) if allowed is None else tuple(allowed)
        if len(set(values)) != len(values) or not set(values).issubset(vocabulary):
            raise ValueError(f'Allowed {kind} types must be unique vocabulary members')
        if vocabulary and not values:
            raise ValueError(f'At least one {kind} type must be allowed')
        return values

    def __len__(self):
        return len(self.tokens)

    def _token(self, index):
        if not isinstance(index, Integral) or isinstance(index, bool) or not 0 <= index < len(self):
            raise ValueError(f'Invalid token ID: {index!r}')
        return self.tokens[index]

    def encode(self, snack, *, add_special_tokens=True, validate=False):
        """Map exact tokens to IDs; no reordering, normalization or UNK.

        validate=True checks the complete sequence against generation masks,
        including configured traversal, connectivity, node limit and chemistry.
        The default only requires that every token is in the vocabulary.
        """
        try:
            ids = [self.token_to_id[token] for token in _graph.split(snack)]
        except KeyError as exc:
            raise ValueError(f'Token {exc.args[0]!r} is outside this vocabulary') from exc
        complete = [self.bos_id] + ids + [self.eos_id]
        if validate:
            self._read(complete)
        return complete if add_special_tokens else ids

    def decode(self, ids):
        """Reconstruct SNACK, allowing optional BOS/EOS and trailing PAD only."""
        tokens = [self._token(index) for index in ids]
        if tokens and tokens[0] == '[bos]':
            tokens.pop(0)
        result = []
        ended = False
        for token in tokens:
            if token == '[eos]' and not ended:
                ended = True
            elif token == '[pad]':
                ended = True
            elif token in ('[bos]', '[eos]') or ended:
                raise ValueError('Misplaced special token or data after EOS/PAD')
            else:
                result.append(token)
        return ''.join(result)

    def __call__(self, snack, *, validate=False):
        """Return encoded IDs in an input_ids dictionary."""
        return {'input_ids': self.encode(snack, validate=validate)}

    def new_state(self):
        return _State()

    # ----- valence bookkeeping -------------------------------------------------
    def _max_capacity(self, token):
        """Capacity over every charge in the vocabulary and H convention (0 when none is possible)."""
        modes = (False, True) if self._lenient else (False,)
        return max((capacity(token, q, lenient) or 0 for q in self._charges for lenient in modes
                    if capacity(token, q, lenient) is not None), default=0)

    def _remaining(self, state, index, charge=None, lenient=None):
        """Bond-order capacity left on a node, for its written (or given) charge.

        ``lenient`` selects RDKit's convention for atoms with a given H count;
        by default it follows whether the node carries a hydrogens attribute.
        """
        if not self.valence:
            return UNLIMITED
        charge = state.charge[index] if charge is None else charge
        if lenient is None:
            lenient = state.hydrogens[index] is not None
        limit = capacity(state.nodes[index], charge, lenient)
        if limit is None:
            return -1
        return limit - (state.hydrogens[index] or 0) - state.used[index]

    def _hydrogens_pending(self, state, index):
        """Whether the current node can still receive a hydrogens token."""
        return (self._lenient and state.hydrogens[index] is None
                and state.attribute_rank < _graph._ATTRIBUTE_RANK['hydrogens'])

    def _room(self, state, index, charge=None):
        """Capacity the current node can still reach: as written, or after a hydrogens token."""
        best = self._remaining(state, index, charge)
        if self._hydrogens_pending(state, index):
            best = max(best, self._remaining(state, index, charge, lenient=True))
        return best

    def _node_complete(self, state):
        """Whether the most recent node is valid as written (no attribute still required)."""
        return not state.nodes or not self.valence or self._remaining(state, len(state.nodes) - 1) >= 0

    def _fixable(self, state, index):
        """Charges in the vocabulary after which node ``index`` can still become valid."""
        return [q for q in self._charges if self._room(state, index, q) >= 0]

    # ----- structural helpers ---------------------------------------------------
    def _current(self, state):
        return len(state.nodes) - int(self.node_first)

    def _block_ok(self, state):
        return self._current(state) == 0 or bool(state.block) or self.allow_disconnected

    def _finish_block(self, state):
        index = self._current(state)
        state.bfs_floor = min(state.block) if state.block else index
        state.block.clear()
        state.last_distance = None

    def _bond_options(self, state, target, room=None):
        """Bond tokens allowed between the current node and target.

        ``room`` is the current node's spare capacity, when known.
        """
        if not self.molecular:
            return (_graph.PLAIN_EDGE,)
        if not self.valence:
            return self._edges
        if room is None:
            room = self._current_room(state)
        room = min(room, self._remaining(state, target))
        return tuple(edge for edge in self._edges if _BOND_ORDERS[edge] <= room)

    def _current_room(self, state):
        if self.node_first:
            return self._remaining(state, self._current(state))
        return self._max_node_capacity - sum(_BOND_ORDERS[edge] for edge in state.block.values())

    def _distance_candidates(self, state):
        current = self._current(state)
        if current < 1 or (not self.node_first and len(state.nodes) >= self.max_nodes):
            return []
        if not self._node_complete(state):
            return []
        room = self._current_room(state) if self.valence else UNLIMITED
        floor = state.bfs_floor if self.ordering == 'bfs' else 0
        allowed = []
        for distance in range(1, min(current - floor, self.max_distance) + 1):
            target = current - distance
            if target in state.block:
                continue
            if state.last_distance is not None and (
                    distance <= state.last_distance if self.ascending else distance >= state.last_distance):
                continue
            if self._bond_options(state, target, room):
                allowed.append(self._distance_ids[distance - 1])
        return allowed

    def _chemical_end(self, state):
        if not self.valence:
            return True
        import networkx as nx
        from rdkit import rdBase
        from snack.mol import graph_to_mol
        graph = nx.Graph()
        for i, token in enumerate(state.nodes):
            data = {'node_type': token}
            if state.charge[i]:
                data['charge'] = state.charge[i]
            if state.hydrogens[i] is not None:
                data['hydrogens'] = state.hydrogens[i]
            graph.add_node(i, **data)
        graph.add_edges_from((u, v, {'edge_type': edge}) for u, v, edge in state.edges)
        with rdBase.BlockLogs():
            try:
                graph_to_mol(graph)
            except ValueError:
                return False
        return True

    def _can_end(self, state):
        if not state.nodes or state.pending is not None or not self._node_complete(state):
            return False
        if self.node_first:
            if not self._block_ok(state):
                return False
        elif state.block:
            return False
        return self._chemical_end(state)

    def _node_candidates(self, state):
        if len(state.nodes) >= self.max_nodes or state.pending is not None:
            return []
        if not self._node_complete(state):
            return []
        if self.node_first:
            if state.nodes and not self._block_ok(state):
                return []
            if not state.nodes or self.allow_disconnected:
                return list(self._node_ids)
            # Do not start a connected node if it cannot attach to any predecessor.
            floor = (min(state.block) if state.block else len(state.nodes) - 1) if self.ordering == 'bfs' else 0
            targets = range(max(floor, len(state.nodes) - self.max_distance), len(state.nodes))
            if not any(self._remaining(state, t) >= 1 for t in targets):
                return []
            return [index for index in self._node_ids if self._capacities[self.tokens[index]] >= 1]
        if not self._block_ok(state):
            return []
        if not self.valence:
            return list(self._node_ids)
        used = sum(_BOND_ORDERS[edge] for edge in state.block.values())
        final = len(state.nodes) + 1 == self.max_nodes
        allowed = []
        for index in self._node_ids:
            if self._capacities[self.tokens[index]] < used:
                continue
            if final and not self._final_node_ok(state, index):
                continue
            allowed.append(index)
        return allowed

    def _final_node_ok(self, state, index):
        """Whether the last permitted node can still finish, possibly after a charge token."""
        trial = state.copy()
        self._consume(trial, index)
        fixes = self._fixable(trial, len(trial.nodes) - 1)
        if not fixes:
            return False
        trial.charge[-1] = 0 if 0 in fixes else fixes[0]
        if self._remaining(trial, len(trial.nodes) - 1) < 0:
            trial.hydrogens[-1] = 0   # valid only under the given-H convention
        return self._can_end(trial)

    def _attribute_ids(self, state):
        """Attribute tokens that may directly follow the node or edge just completed."""
        if state.last is None:
            return []
        kind, key = state.last
        if kind == 'edge':
            node, target = key
            return ([index for index, _ in self._attributes.get('cis_trans', ())]
                    if state.block.get(target) == '=' else [])
        index = key
        allowed = []
        for rank, name in enumerate(_graph.NODE_ATTRIBUTES):
            if rank <= state.attribute_rank:
                continue
            for token_id, value in self._attributes.get(name, ()):
                if self._attribute_ok(state, index, name, value):
                    allowed.append(token_id)
        return allowed

    def _attribute_ok(self, state, index, name, value):
        if name == 'isotope':
            return True        # a mass number constrains nothing about bonding
        if not self.valence:
            return True
        # A node-first node that still needs its backward edge must keep room for it.
        floor = int(self.node_first and index > 0 and not state.block and not self.allow_disconnected)
        if name == 'charge':
            return self._room(state, index, value) >= floor
        if name == 'hydrogens':
            return self._remaining(state, index, lenient=True) - value >= floor
        if name == 'chirality':
            # The last attribute: the node must already be valid as written.
            neutral = capacity(state.nodes[index], 0)
            return (state.degree[index] <= 4 and (neutral is None or neutral >= 3)
                    and self._remaining(state, index) >= floor)
        raise ValueError(name)

    def allowed_token_ids(self, state):
        """Allowed IDs for an incremental state. An empty result is a dead end.

        At max_nodes no further node is allowed. Edge-first then permits only
        attributes of the final node and a valid EOS. Node-first may finish the
        final node's backward edges before EOS.
        """
        if not state.started:
            return [self.bos_id]
        if state.ended:
            return [self.pad_id]
        if state.pending is not None:
            return [self.token_to_id[edge] for edge in self._bond_options(state, state.pending)]
        allowed = self._node_candidates(state) + self._distance_candidates(state)
        allowed += self._attribute_ids(state)
        if self._can_end(state):
            allowed.append(self.eos_id)
        return allowed

    def _add_edge(self, state, target, edge):
        current = self._current(state)
        state.edges.append((current, target, edge))
        state.block[target] = edge
        state.degree[target] += 1
        state.used[target] += _BOND_ORDERS.get(edge, 0)
        if self.node_first:
            state.degree[current] += 1
            state.used[current] += _BOND_ORDERS.get(edge, 0)
        state.last = ('edge', (current, target))

    def _consume(self, state, index):
        token = self.tokens[index]
        state.last = None
        if index == self.bos_id:
            state.started = True
        elif index == self.eos_id:
            state.ended = True
        elif index == self.pad_id:
            pass
        elif token in self.node_types:
            if self.node_first:
                if state.nodes:
                    self._finish_block(state)
                used, degree = 0, 0
            else:
                used = sum(_BOND_ORDERS.get(e, 0) for e in state.block.values())
                degree = len(state.block)
                # Finish while the new node's index is still len(nodes).
                self._finish_block(state)
            state.nodes.append(token)
            state.charge.append(0)
            state.hydrogens.append(None)
            state.used.append(used)
            state.degree.append(degree)
            state.attribute_rank = -1
            state.last = ('node', len(state.nodes) - 1)
        elif token in self.attribute_types:
            key, value = _graph.parse_attribute(token)
            if key == 'cis_trans':
                return
            current = len(state.nodes) - 1
            if key == 'hydrogens':
                state.hydrogens[current] = value
            elif key == 'charge':
                state.charge[current] = value
            # chirality and isotope constrain no later token, so carry no state
            # beyond the attribute rank below.
            state.attribute_rank = _graph._ATTRIBUTE_RANK[key]
            state.last = ('node', current)
        elif token in self.edge_types:
            self._add_edge(state, state.pending, token)
            state.pending = None
        else:
            distance = int(token)
            state.last_distance = distance
            # Both presets write the type after the distance, so the edge is
            # only complete once that token arrives.
            state.pending = self._current(state) - distance

    def step(self, state, token_id):
        """Consume one allowed token in place; reject invalid prefixes immediately."""
        token = self._token(token_id)
        if token_id not in self.allowed_token_ids(state):
            raise ValueError(f'Token {token!r} violates the configured generation constraints')
        self._consume(state, token_id)
        return state

    def _read(self, ids):
        state = self.new_state()
        for index in ids:
            self.step(state, index)
        return state

    def allowed_next_ids(self, prefix_ids):
        """Prefix API; use new_state/step/allowed_token_ids for long generations."""
        return self.allowed_token_ids(self._read(prefix_ids))

    def next_token_mask(self, prefix_ids):
        allowed = set(self.allowed_next_ids(prefix_ids))
        return [index in allowed for index in range(len(self))]

    def get_logit_mask(self, ids):
        """Row i masks the next token AFTER consuming ids[i], for shifted targets."""
        state = self.new_state()
        result = []
        for index in ids:
            self.step(state, index)
            allowed = set(self.allowed_token_ids(state))
            result.append([i in allowed for i in range(len(self))])
        return result
