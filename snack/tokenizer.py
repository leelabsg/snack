"""Small SNACK tokenizer and next-token masks; no torch/numpy/pandas imports.

Masks are vocabulary/logit masks (True = allowed), not transformer attention
matrices. Molecular presets model simple, kekulized, non-stereochemical graphs.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral

from snack import graph as _graph

# Conservative organic chemistry domain, including the charged ZINC atom types.
# Explicit H counts consume valence. Implicit H can fill unused valence;
# with fixed H counts, RDKit infers radicals from unfilled valence.
_VALENCES = {
    ('C', 0): (4,), ('C', -1): (3,), ('N', 0): (3,), ('N', 1): (4,),
    ('N', -1): (2,), ('O', 0): (2,), ('O', 1): (3,), ('O', -1): (1,),
    ('P', 0): (3, 5), ('P', 1): (4,), ('S', 0): (2, 4, 6),
    ('S', 1): (3, 5), ('S', -1): (1, 3, 5),
    ('F', 0): (1,), ('Cl', 0): (1,), ('Br', 0): (1,), ('I', 0): (1,),
}
# GuacaMol includes hypervalent iodine/phosphorus/silicon and additional
# charged atoms. Keep these extensions separate so existing preset IDs and
# valence masks do not change. Halide anions are supported as isolated ions.
_GUACAMOL_VALENCES = {
    ('B', 0): (3,), ('B', -1): (4,), ('C', 1): (3,),
    ('F', 1): (2,), ('F', -1): (0,),
    ('Cl', 1): (2,), ('Cl', 2): (3,), ('Cl', 3): (4,), ('Cl', -1): (0,),
    ('Br', 2): (3,), ('Br', -1): (0,),
    ('I', 0): (1, 3, 5), ('I', 1): (2, 4, 6),
    ('I', 2): (3,), ('I', 3): (4,), ('I', -1): (0,),
    ('P', -1): (2, 4, 6), ('Si', 0): (4,), ('Si', -1): (3, 5),
    ('Se', 0): (2, 4, 6), ('Se', 1): (3, 5), ('Se', -1): (1, 3, 5),
}
_BASE_ATOMS = {
    'zinc250k': ('[Br]', '[C-]', '[C]', '[Cl]', '[F]', '[I]', '[N+]', '[N-]',
                '[N]', '[O+]', '[O-]', '[O]', '[P+]', '[P]', '[S+]', '[S-]', '[S]'),
    'moses': ('[Br]', '[C]', '[Cl]', '[F]', '[N]', '[O]', '[S]'),
    'guacamol': ('[B-]', '[B]', '[Br+2]', '[Br-]', '[Br]', '[C+]', '[C-]', '[C]',
                 '[Cl+2]', '[Cl+3]', '[Cl+]', '[Cl-]', '[Cl]', '[F+]', '[F-]', '[F]',
                 '[I+2]', '[I+3]', '[I+]', '[I-]', '[I]', '[N+]', '[N-]', '[N]',
                 '[O+]', '[O-]', '[O]', '[P+]', '[P-]', '[P]', '[S+]', '[S-]', '[S]',
                 '[Se+]', '[Se-]', '[Se]', '[Si-]', '[Si]'),
}
_MAX_NODES = {'graph': 128, 'zinc250k': 38, 'moses': 27, 'guacamol': 88}
_BOND_ORDERS = {'-': 1, '=': 2, '#': 3}


@lru_cache(maxsize=None)
def _atom_spec(token, preset):
    """(maximum bond-order sum, allowed completed sums) for a basic atom."""
    from snack.mol import RE_ATOM_TYPE, string_to_atom
    match = RE_ATOM_TYPE.fullmatch(token[1:-1]) if token.startswith('[') and token.endswith(']') else None
    if match is None or any(match[key] is not None for key in ('isotope', 'map')):
        raise ValueError(f'Valence masking only supports simple atom tokens, got {token!r}')
    atom = string_to_atom(token)
    key = (atom.GetSymbol(), atom.GetFormalCharge())
    values = _VALENCES.get(key)
    if preset == 'guacamol':
        values = _GUACAMOL_VALENCES.get(key, values)
    if values is None:
        raise ValueError(f'No supported valence rule for {token!r}')
    hydrogens = atom.GetNumExplicitHs()
    # A fixed H count limits capacity but does not require a closed shell.
    # For example, [CH2] with one single bond is a valid carbon radical.
    allowed = frozenset(range(max(values) - hydrogens + 1))
    if not allowed:
        raise ValueError(f'Explicit hydrogens exceed the supported valence of {token!r}')
    return max(allowed), allowed


def basic_vocabulary(preset='graph', max_nodes=None, max_distance=None):
    """Return stable tokens, including [pad], [bos], [eos], for a preset.

    Molecular presets include explicit H variants for use_hydrogens=True
    output. Complex edge labels require a custom vocabulary and valence=False;
    no chemistry guarantee applies then.
    """
    if preset not in _MAX_NODES:
        raise ValueError('preset must be graph, zinc250k, moses, or guacamol')
    maximum = _MAX_NODES[preset] if max_nodes is None else max_nodes
    if not isinstance(maximum, Integral) or isinstance(maximum, bool) or maximum < 1:
        raise ValueError('max_nodes must be a positive integer')
    distance = maximum - 1 if max_distance is None else max_distance
    if not isinstance(distance, Integral) or isinstance(distance, bool) or not 0 <= distance < maximum:
        raise ValueError('max_distance must be between 0 and max_nodes - 1')
    atoms = ['v'] if preset == 'graph' else list(_BASE_ATOMS[preset])
    if preset != 'graph':
        from snack.mol import RE_ATOM_TYPE
        for base in _BASE_ATOMS[preset]:
            match = RE_ATOM_TYPE.fullmatch(base[1:-1])
            capacity, _ = _atom_spec(base, preset)
            for h in range(capacity + 1):
                atoms.append(f'[{match["element"]}H{h if h != 1 else ""}{match["charge"] or ""}]')
    distances = [f'{i}_' if preset == 'graph' else str(i) for i in range(1, distance + 1)]
    return ['[pad]', '[bos]', '[eos]'] + atoms + distances + ([] if preset == 'graph' else list(_BOND_ORDERS))


@dataclass
class _State:
    started: bool = False
    ended: bool = False
    nodes: list = field(default_factory=list)
    used: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    block: dict = field(default_factory=dict)
    pending: int | None = None
    last_distance: int | None = None
    bfs_floor: int = 0


class SnackTokenizer:
    """Encode/decode SNACK and constrain autoregressive generation.

    ``ordering='bfs'`` enforces a possible BFS traversal. ``None`` imposes no
    traversal constraint. Connected generation requires a backward edge on
    every node after the first (connected prefixes). Disconnected BFS starts
    a new, contiguous component whenever a node has no backward edges.

    ``node_first`` and ``ascending`` default to current SNACK settings, captured
    at construction. ``valence`` defaults to True for molecular presets.
    Optional node_types/edge_types replace the preset vocabulary; optional
    allowed_node_types/allowed_edge_types restrict generation only.

    Preset valence masks default to implicit-H atom types. Explicit-H variants
    stay in the vocabulary for encoding; opt into generating them via
    allowed_node_types. Custom restrictions can lead to an empty mask:
    backtrack or restart rather than emitting an invalid EOS. Every permitted
    molecular EOS passes RDKit sanitization, which also infers radical counts.
    """
    def __init__(self, preset='graph', *, max_nodes=None, max_distance=None,
                 ordering=None, allow_disconnected=False, node_first=None, ascending=None,
                 valence=None, node_types=None, edge_types=None,
                 allowed_node_types=None, allowed_edge_types=None):
        base = basic_vocabulary(preset, max_nodes, max_distance)
        self.preset = preset
        self.max_nodes = _MAX_NODES[preset] if max_nodes is None else int(max_nodes)
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
            raise ValueError('Valence masks require a molecular preset')
        self.node_types = tuple(node_types) if node_types is not None else tuple(
            token for token in base[3:] if _graph.token_type(token) == 'node_type')
        self.edge_types = tuple(edge_types) if edge_types is not None else tuple(
            token for token in base[3:] if _graph.token_type(token) == 'edge_type')
        if not self.node_types or any(_graph.token_type(token) != 'node_type' for token in self.node_types):
            raise ValueError('node_types must contain valid SNACK node tokens')
        if any(_graph.token_type(token) != 'edge_type' for token in self.edge_types):
            raise ValueError('edge_types must contain serialized edge tokens, e.g. (contact)')
        if (not self.molecular and self.edge_types) or (self.molecular and not self.edge_types):
            raise ValueError('Graph mode uses untyped edges; molecular mode requires edge types')
        distances = [token for token in base if _graph.token_type(token) == 'distance']
        self.tokens = tuple(base[:3]) + self.node_types + tuple(distances) + self.edge_types
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError('Vocabulary must not contain duplicate or reserved tokens')
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.eos_id = range(3)
        # Default generation uses implicit-H atoms. H variants remain in the
        # vocabulary for encoding; opt into their generation via allowed_node_types.
        if allowed_node_types is None and node_types is None and self.valence:
            allowed_node_types = _BASE_ATOMS[preset]
        self._nodes = self._restrict(self.node_types, allowed_node_types, 'node')
        self._edges = self._restrict(self.edge_types, allowed_edge_types, 'edge')
        self._specs = {token: _atom_spec(token, preset) for token in self._nodes} if self.valence else {}
        if self.valence and any(edge not in _BOND_ORDERS for edge in self._edges):
            raise ValueError('Valence masking supports only -, =, #; use kekulized molecules')
        self._node_ids = [self.token_to_id[token] for token in self._nodes]
        self._edge_ids = [self.token_to_id[token] for token in self._edges]
        self._distance_ids = [self.token_to_id[token] for token in distances]

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

    def _current(self, state):
        return len(state.nodes) - int(self.node_first)

    def _block_ok(self, state):
        return self._current(state) == 0 or bool(state.block) or self.allow_disconnected

    def _finish_block(self, state):
        index = self._current(state)
        state.bfs_floor = min(state.block) if state.block else index
        state.block.clear()
        state.last_distance = None

    def _capacity(self, state, target):
        return self._specs[state.nodes[target]][0] - state.used[target]

    def _bond_options(self, state, target):
        if not self.molecular:
            return ('',)
        if not self.valence:
            return self._edges
        current = self._current(state)
        used = sum(_BOND_ORDERS[edge] for edge in state.block.values())
        capacity = (self._capacity(state, current) if self.node_first else
                    max(self._specs[node][0] for node in self._nodes) - used)
        return tuple(edge for edge in self._edges
                     if _BOND_ORDERS[edge] <= min(capacity, self._capacity(state, target)))

    def _distance_ok(self, state, distance):
        current = self._current(state)
        if current < 1 or (not self.node_first and len(state.nodes) >= self.max_nodes):
            return False
        target = current - distance
        if target < 0 or target in state.block:
            return False
        if state.last_distance is not None:
            if (distance <= state.last_distance if self.ascending else distance >= state.last_distance):
                return False
        if self.ordering == 'bfs' and target < state.bfs_floor:
            return False
        return bool(self._bond_options(state, target))

    def _chemical_end(self, state):
        if not self.valence:
            return True
        if any(used not in self._specs[node][1] for node, used in zip(state.nodes, state.used)):
            return False
        import networkx as nx
        from rdkit import rdBase
        from snack.mol import graph_to_mol
        graph = nx.Graph()
        graph.add_nodes_from((i, {'node_type': token}) for i, token in enumerate(state.nodes))
        graph.add_edges_from((u, v, {'edge_type': edge}) for u, v, edge in state.edges)
        with rdBase.BlockLogs():
            try:
                graph_to_mol(graph)
            except ValueError:
                return False
        return True

    def _can_end(self, state):
        if not state.nodes or state.pending is not None:
            return False
        if self.node_first:
            if not self._block_ok(state):
                return False
        elif state.block:
            return False
        return self._chemical_end(state)

    def _node_ok(self, state, token):
        if len(state.nodes) >= self.max_nodes or state.pending is not None:
            return False
        if self.node_first:
            if state.nodes and not self._block_ok(state):
                return False
            # Do not start a connected node if it cannot attach to any predecessor.
            if state.nodes and not self.allow_disconnected:
                floor = (min(state.block) if state.block else len(state.nodes) - 1) if self.ordering == 'bfs' else 0
                targets = range(max(floor, len(state.nodes) - self.max_distance), len(state.nodes))
                if not any(not self.valence or any(_BOND_ORDERS[e] <= min(
                    self._specs[token][0], self._capacity(state, t)) for e in self._edges) for t in targets):
                    return False
            return True
        if not self._block_ok(state):
            return False
        if self.valence:
            used = sum(_BOND_ORDERS[edge] for edge in state.block.values())
            if used > self._specs[token][0]:
                return False
            if len(state.nodes) + 1 == self.max_nodes:
                trial = self._copy(state)
                self._consume(trial, self.token_to_id[token])
                return self._can_end(trial)
        return True

    @staticmethod
    def _copy(state):
        return _State(state.started, state.ended, state.nodes[:], state.used[:],
                      state.edges[:], dict(state.block), state.pending,
                      state.last_distance, state.bfs_floor)

    def allowed_token_ids(self, state):
        """Allowed IDs for an incremental state. An empty result is a dead end.

        At max_nodes no further node is allowed. Edge-first then permits only
        valid EOS. Node-first may finish the final node's backward edges before
        EOS; stopping immediately on that node token would omit its bonds.
        """
        if not state.started:
            return [self.bos_id]
        if state.ended:
            return [self.pad_id]
        if state.pending is not None:
            return [self.token_to_id[edge] for edge in self._bond_options(state, state.pending)]
        allowed = [index for index in self._node_ids if self._node_ok(state, self.tokens[index])]
        allowed += [index for index in self._distance_ids
                    if self._distance_ok(state, int(self.tokens[index].rstrip('_')))]
        if self._can_end(state):
            allowed.append(self.eos_id)
        return allowed

    def _add_edge(self, state, target, edge):
        current = self._current(state)
        state.edges.append((current, target, edge))
        state.block[target] = edge
        if self.valence:
            order = _BOND_ORDERS[edge]
            state.used[target] += order
            if self.node_first:
                state.used[current] += order

    def _consume(self, state, index):
        token = self.tokens[index]
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
                state.nodes.append(token)
                state.used.append(0)
            else:
                used = sum(_BOND_ORDERS[e] for e in state.block.values()) if self.valence else 0
                # Finish while the new node's index is still len(nodes).
                self._finish_block(state)
                state.nodes.append(token)
                state.used.append(used)
        elif token in self.edge_types:
            self._add_edge(state, state.pending, token)
            state.pending = None
        else:
            distance = int(token.rstrip('_'))
            state.last_distance = distance
            target = self._current(state) - distance
            if self.molecular:
                state.pending = target
            else:
                self._add_edge(state, target, '')

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
