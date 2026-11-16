# ordering functions are adapted from:
# https://github.com/Genentech/bandwidth-graph-generation/blob/main/graph_gen/data/bandwidth.py
import math
import re
import random
from numbers import Integral, Real
from operator import itemgetter
from collections import deque, defaultdict, Counter
import networkx as nx
import pynauty

NODE_FIRST = False
ASCENDING = False

# Bare o is an untyped node; molecular nodes are element symbols such as C or Cl, or *.
DEFAULT_NODE_TOKEN = 'o'

# Every edge is a distance followed by a type: - = # for single, double and
# triple bonds, and - for a plain edge that carries no type of its own. So
# o1-o1-o is a three-node path and C1-C1=C is propene. A plain edge and a
# single bond are written alike, which is why decoding leaves edge_type unset
# for - and graph_to_mol reads a missing type as a single bond, as SMILES does.
NODE_TYPE_STR = r'o|\*|[A-Z][a-z]?'
DISTANCE_STR = r'[1-9][0-9]*'
EDGE_TYPE_STR = r'[-=#a]'
ATTRIBUTE_STR = r'\([^()\r\n]+\)'
# a is an aromatic bond, for callers who would rather not kekulize. A type
# always follows a distance, so the a in Na1-C is sodium and never a bond.
EDGE_TYPES = ('-', '=', '#', 'a')
PLAIN_EDGE = '-'
AROMATIC_EDGE = 'a'

RE_NODE_TYPE = re.compile(NODE_TYPE_STR)
RE_DISTANCE = re.compile(DISTANCE_STR)
RE_EDGE_TYPE = re.compile(EDGE_TYPE_STR)
RE_ATTRIBUTE = re.compile(ATTRIBUTE_STR)
RE_SPLIT_TOKEN = re.compile(f'{NODE_TYPE_STR}|{DISTANCE_STR}|{EDGE_TYPE_STR}|{ATTRIBUTE_STR}')

# Attributes follow the node or edge they describe, one per parenthesized token,
# in this fixed order. Graph attribute keys share these names:
#   node 'chirality' = (four ligand IDs, None for an implicit H or lone pair, '@' | '@@')
#   node 'hydrogens' = int >= 0, 'charge' = nonzero int, 'isotope' = int >= 1
#   edge 'cis_trans' = ((u, ref_u), (v, ref_v), 'c' | 't')
# Chirality and cis/trans tokens are relative to the node order; graphs store
# them order-independently. '@'/'@@' have the SMILES meaning for the ligand
# list; 'c' means the two reference substituents are on the same side.
# An isotope is the mass number, written bare: C(13) is carbon-13, H(2) is
# deuterium. It comes first, as the mass number does in a SMILES bracket atom.
NODE_ATTRIBUTES = ('isotope', 'charge', 'hydrogens', 'chirality')
EDGE_ATTRIBUTES = ('cis_trans',)
ATTRIBUTES = NODE_ATTRIBUTES + EDGE_ATTRIBUTES
_ATTRIBUTE_RANK = {key: i for i, key in enumerate(NODE_ATTRIBUTES)}
_FLIP = {'@': '@@', '@@': '@', 'c': 't', 't': 'c'}
_RE_ATTRIBUTE_BODIES = {
    'isotope': re.compile(r'[1-9][0-9]*'),
    'chirality': re.compile(r'@@?'),
    'hydrogens': re.compile(r'H(0|[2-9][0-9]*)?'),
    'charge': re.compile(r'[+-]([2-9][0-9]*)?'),
    'cis_trans': re.compile(r'[ct]'),
}
_ATTRIBUTE_KEY_BY_PREFIX = {'@': 'chirality', 'H': 'hydrogens', '+': 'charge', '-': 'charge',
                            'c': 'cis_trans', 't': 'cis_trans',
                            **{str(digit): 'isotope' for digit in range(1, 10)}}

# Any other node or edge data is written as one (key=value) token per entry,
# so o(label=hub)1-(weight=3)o keeps both. The kinds above have compact forms
# and may not be written this way; node_type and edge_type use it only when
# their value has no bare token, as in o(node_type=hub) or o1-(edge_type=knows)o.
NAMED_KEY_STR = r'[A-Za-z_][A-Za-z0-9_]*'
_RE_NAMED = re.compile(rf'({NAMED_KEY_STR})=(.*)', re.S)
_RE_INTEGER = re.compile(r'-?(?:0|[1-9][0-9]*)')
_RE_REAL = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
# Only what would break the grammar is escaped, so values stay readable.
_ESCAPED = {'%': '%25', '(': '%28', ')': '%29', '\r': '%0D', '\n': '%0A'}
_RE_ESCAPE = re.compile(r'%([0-9A-Fa-f]{2})')


def _escape(text):
    return ''.join(_ESCAPED.get(character, character) for character in text)


def _unescape(text):
    if re.search(r'%(?![0-9A-Fa-f]{2})', text):
        raise ValueError(f'Invalid escape in attribute value: {text!r}')
    return _RE_ESCAPE.sub(lambda match: chr(int(match[1], 16)), text)


def _value_text(value):
    """Serialize an attribute value so that parsing returns it unchanged."""
    if isinstance(value, bool):
        raise ValueError('Attribute values must be an integer, a real number or a string, '
                         f'got {value!r}')
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError(f'Attribute values must be finite, got {value!r}')
        return repr(float(value))
    if not isinstance(value, str):
        raise ValueError('Attribute values must be an integer, a real number or a string, '
                         f'got {value!r}')
    text = _escape(value)
    # A string that reads as a number would come back as one, so hide its
    # first character: (size=%33) is the string '3', (size=3) the integer 3.
    if _RE_REAL.fullmatch(text):
        return f'%{ord(text[0]):02X}{text[1:]}'
    return text


def _parse_value(text):
    if _RE_INTEGER.fullmatch(text):
        return int(text)
    if _RE_REAL.fullmatch(text):
        return float(text)
    return _unescape(text)


def attribute_token(key, value):
    """Serialize one attribute value, e.g. ('charge', -2) -> '(-2)'.

    Keys outside the fixed kinds get the named form, ('weight', 3) -> '(weight=3)'.
    """
    if key not in ATTRIBUTES:
        if not isinstance(key, str) or not re.fullmatch(NAMED_KEY_STR, key):
            raise ValueError(f'Invalid attribute name: {key!r}; '
                             'names start with a letter or underscore')
        return f'({key}={_value_text(value)})'
    if key == 'chirality':
        if value not in ('@', '@@'):
            raise ValueError(f'chirality must be @ or @@, got {value!r}')
        return f'({value})'
    if key == 'cis_trans':
        if value not in ('c', 't'):
            raise ValueError(f'cis_trans must be c or t, got {value!r}')
        return f'({value})'
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{key} must be an integer, got {value!r}')
    value = int(value)
    if key == 'isotope':
        if value < 1:
            raise ValueError(f'isotope must be a mass number >= 1, got {value}')
        return f'({value})'
    if key == 'hydrogens':
        if value < 0:
            raise ValueError(f'hydrogens must be >= 0, got {value}')
        return '(H)' if value == 1 else f'(H{value})'
    if key == 'charge':
        if value == 0:
            raise ValueError('charge attribute must be nonzero')
        sign = '+' if value > 0 else '-'
        return f'({sign})' if abs(value) == 1 else f'({sign}{abs(value)})'
    raise ValueError(f'Unknown attribute: {key!r}')


def parse_attribute(token):
    """Return (key, value) for an attribute token such as '(+2)', '(@@)' or '(13)'.

    Counts of one are written without a digit: (H), (+) and (-). An isotope is
    a bare mass number, always written in full: (2) is deuterium.
    """
    body = token[1:-1] if RE_ATTRIBUTE.fullmatch(token) else ''
    named = _RE_NAMED.fullmatch(body)
    if named is not None:
        if named[1] in ATTRIBUTES:
            raise ValueError(f'{named[1]} has a compact form; write it that way, '
                             f'not as {token!r}')
        return named[1], _parse_value(named[2])
    key = _ATTRIBUTE_KEY_BY_PREFIX.get(body[:1])
    match = _RE_ATTRIBUTE_BODIES[key].fullmatch(body) if key else None
    if match is None:
        raise ValueError(f'Invalid attribute token: {token!r}')
    if key in ('chirality', 'cis_trans'):
        return key, body
    if key == 'isotope':
        return key, int(body)
    count = int(match[1]) if match[1] is not None else 1
    return key, -count if body[0] == '-' else count


def random_neighbor_order(G: nx.Graph) -> list[int]:
    start = random.choice(list(G))
    visited = {start}
    frontier = [start]
    order = []
    while frontier:
        i = random.choice(range(len(frontier)))
        parent = frontier.pop(i)
        order.append(parent)
        children = list(set(G[parent]) - visited)
        visited.update(children)
        frontier.extend(children)
    return order


def uniform_random_order(G: nx.Graph) -> list[int]:
    order = list(G.nodes())
    random.shuffle(order)
    return order

def pseudo_peripheral_node(G: nx.Graph) -> int:
    """adapted from NX source"""
    # helper for cuthill-mckee to find a node in a "pseudo peripheral pair"
    # to use as good starting node
    u = random.choice(list(G))
    lp = 0
    v = u
    while True:
        spl = dict(nx.shortest_path_length(G, v))
        l = max(spl.values())
        if l <= lp:
            break
        lp = l
        farthest = (n for n, dist in spl.items() if dist == l)
        v, deg = min(G.degree(farthest), key=itemgetter(1))
    return v


def random_cuthill_mckee_order(G: nx.Graph) -> list[int]:
    """
    adapted from NX source.
    :return: node order
    """
    # the cuthill mckee algorithm for connected graphs
    start = pseudo_peripheral_node(G)
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        parent = queue.popleft()
        order.append(parent)
        frontier = set(G[parent]) - visited
        # low degree first
        nd = sorted(G.degree(frontier), key=lambda x: (x[1], random.random()))
        children = [n for n, d in nd]
        visited.update(children)
        queue.extend(children)
    return order


def random_bfs_order(G: nx.Graph) -> list[int]:
    start = random.choice(list(G))
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        parent = queue.popleft()
        order.append(parent)
        children = sorted(set(G[parent]) - visited, key=lambda x: random.random())
        visited.update(children)
        queue.extend(children)
    return order


ORDER_FUNCS = {
    "cm": random_cuthill_mckee_order,
    "bfs": random_bfs_order,
    "uniform": uniform_random_order,
    "neighbor": random_neighbor_order,
}

def _validate_graph(graph):
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError('SNACK supports simple undirected graphs without self-loops')


def perm_parity(reference, target) -> int:
    """Return +1 if target is an even permutation of reference, else -1."""
    reference, target = list(reference), list(target)
    if (len(reference) != len(target) or len(set(reference)) != len(reference)
            or set(reference) != set(target)):
        raise ValueError('perm_parity requires two orderings of the same distinct items')
    position = {item: i for i, item in enumerate(reference)}
    permutation = [position[item] for item in target]
    sign = 1
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            if permutation[i] > permutation[j]:
                sign = -sign
    return sign


def _ligand_listing(graph, node, rank):
    """Neighbors in serialized order; an implicit ligand takes the node's slot.

    E.g. node 2 with neighbors 0, 3, 5 gives [0, None, 3, 5], like X[C@H](Y)Z in SMILES.
    """
    neighbors = sorted(graph[node], key=rank.__getitem__)
    if len(neighbors) == 3:
        neighbors.insert(sum(rank[n] < rank[node] for n in neighbors), None)
    return neighbors


def _check_chirality(graph, node):
    value = graph.nodes[node]['chirality']
    try:
        ligands, sign = value
        ligands = tuple(ligands)
        neighbors = set(graph[node])
        valid = (sign in ('@', '@@') and len(neighbors) in (3, 4) and len(ligands) == 4
                 and set(ligands) == neighbors | ({None} if len(neighbors) == 3 else set()))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'Invalid chirality on node {node!r}: {value!r}')
    return ligands, sign


def _check_cis_trans(graph, u, v):
    value = graph.edges[u, v]['cis_trans']
    try:
        (end_a, ref_a), (end_b, ref_b), mark = value
        refs = {end_a: ref_a, end_b: ref_b}
        valid = mark in ('c', 't') and set(refs) == {u, v} and all(
            1 <= len(set(graph[end]) - {other}) <= 2 and refs[end] in set(graph[end]) - {other}
            for end, other in ((u, v), (v, u)))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'Invalid cis_trans on edge {(u, v)!r}: {value!r}')
    return refs, mark


def _cis_trans_relative(graph, u, v, new_refs):
    """Cis/trans of the edge relative to other reference substituents.

    Changing the reference at one end flips c <-> t; changing both keeps it.
    """
    refs, mark = _check_cis_trans(graph, u, v)
    return mark if sum(refs[end] != new_refs[end] for end in (u, v)) % 2 == 0 else _FLIP[mark]


def _first_substituent(graph, end, other, rank):
    return min(set(graph[end]) - {other}, key=rank.__getitem__)


def _node_chirality(graph, node, rank):
    # sign holds for the stored ligand order; an odd reordering flips it.
    ligands, sign = _check_chirality(graph, node)
    return sign if perm_parity(ligands, _ligand_listing(graph, node, rank)) == 1 else _FLIP[sign]


def _edge_cis_trans(graph, u, v, rank):
    _check_cis_trans(graph, u, v)
    refs = {u: _first_substituent(graph, u, v, rank), v: _first_substituent(graph, v, u, rank)}
    return _cis_trans_relative(graph, u, v, refs)


def _resolve_stereo(graph, node_marks, edge_marks, strict=True):
    """Store parsed marks as chirality/cis_trans; invalid ones raise if strict, else are dropped."""
    rank = {node: node for node in graph}  # parsed nodes are 0..n-1 in sequence order
    for node, sign in node_marks.items():
        if graph.degree(node) not in (3, 4):
            if strict:
                raise ValueError(f'Chirality on node {node} requires 3 or 4 neighbors')
            continue
        graph.nodes[node]['chirality'] = (tuple(_ligand_listing(graph, node, rank)), sign)
    for (u, v), mark in edge_marks.items():
        if graph.edges[u, v].get('edge_type') != '=':
            if strict:
                raise ValueError(f'Cis/trans stereo on edge {(u, v)} requires a double bond')
            continue
        substituents = [set(graph[u]) - {v}, set(graph[v]) - {u}]
        if not all(1 <= len(group) <= 2 for group in substituents):
            if strict:
                raise ValueError(f'Cis/trans stereo on edge {(u, v)} requires one or two '
                                 'substituents on each endpoint')
            continue
        graph.edges[u, v]['cis_trans'] = ((u, min(substituents[0])), (v, min(substituents[1])), mark)
    return graph


def graph_order(G: nx.Graph, ordering='bfs') -> list:
    """Order all nodes, including disconnected components; ordering may be a node list."""
    _validate_graph(G)
    if isinstance(ordering, (list, tuple)):
        order = list(ordering)
        if len(order) != len(G) or set(order) != set(G):
            raise ValueError('An explicit ordering must contain every graph node exactly once')
        return order
    if not isinstance(ordering, str) or ordering.lower() not in ORDER_FUNCS:
        raise ValueError(f'Invalid ordering: {ordering!r}')
    ordering = ordering.lower()
    if ordering == 'uniform':
        return uniform_random_order(G)
    return [node for component in nx.connected_components(G)
            for node in ORDER_FUNCS[ordering](G.subgraph(component))]


def _selected(attributes):
    if attributes is None:
        return frozenset(ATTRIBUTES)
    if isinstance(attributes, str):
        attributes = (attributes,)
    selected = frozenset(attributes)
    if not selected.issubset(ATTRIBUTES):
        raise ValueError(f'Unknown attributes: {sorted(selected - set(ATTRIBUTES))}; '
                         f'choose from {list(ATTRIBUTES)}')
    return selected


def _named_tokens(data, reserved, extra=None):
    """(key=value) tokens for the entries without a compact form.

    Sorted by name, and after the compact tokens, so one graph has one
    spelling. ``extra`` adds a type whose value has no bare token.
    """
    named = {key: value for key, value in data.items() if key not in reserved}
    named.update(extra or {})
    return [attribute_token(key, named[key]) for key in sorted(named)]


def _node_tokens(graph, node, rank, selected):
    """[node_type, attribute tokens...] for one node."""
    data = graph.nodes[node]
    node_type = data.get('node_type')
    bare = node_type is None or (isinstance(node_type, str) and RE_NODE_TYPE.fullmatch(node_type))
    if not bare and not (isinstance(node_type, str) and node_type):
        raise ValueError(f'Invalid node_type: {node_type!r}; use an element symbol, * or a '
                         'nonempty label')
    tokens = [node_type if bare and node_type is not None else DEFAULT_NODE_TOKEN]
    for key in NODE_ATTRIBUTES:
        if key not in selected or key not in data:
            continue
        if key == 'chirality':
            tokens.append(attribute_token(key, _node_chirality(graph, node, rank)))
        elif key == 'hydrogens' or data[key]:
            tokens.append(attribute_token(key, data[key]))
    # A node type with no bare token keeps its value in the named form.
    extra = None if bare else {'node_type': node_type}
    return tokens + _named_tokens(data, set(NODE_ATTRIBUTES) | {'node_type'}, extra)


def _edge_tokens(graph, node, neighbor, distance, rank, selected):
    """[distance, edge_type, attribute tokens...] for one edge."""
    data = graph.edges[node, neighbor]
    edge_type = data.get('edge_type')
    bare = edge_type is None or edge_type in EDGE_TYPES
    if not bare and not (isinstance(edge_type, str) and edge_type):
        raise ValueError(f'Invalid edge_type: {edge_type!r}; use -, =, # or a nonempty label')
    tokens = [str(distance), edge_type if bare and edge_type is not None else PLAIN_EDGE]
    if 'cis_trans' in selected and 'cis_trans' in data:
        tokens.append(attribute_token('cis_trans', _edge_cis_trans(graph, node, neighbor, rank)))
    # An edge type outside - = # keeps its value in the named form.
    extra = None if bare else {'edge_type': edge_type}
    return tokens + _named_tokens(data, set(EDGE_ATTRIBUTES) | {'edge_type'}, extra)


def graph_to_tokens(graph: nx.Graph, ordering=None, *, attributed: bool = True,
                    attributes=None) -> list[str]:
    """Encode a simple graph, using its iteration order when ordering is None.

    Set attributed=False to discard node types, edge types and attributes
    while preserving connectivity. Edge types are -, = and #, and a plain edge
    is written -, so o1-o1-o is a path. Node attributes isotope, charge,
    hydrogens and chirality and the edge attribute cis_trans follow their node
    or edge as (13), (+), (H2), (@@) and (t); any other node or edge data
    follows as (key=value), sorted by name. ``attributes`` selects a subset of
    the compact kinds to write; None writes every attribute present.
    """
    _validate_graph(graph)
    selected = _selected(attributes) if attributed else frozenset()
    nodelist = list(graph) if ordering is None else graph_order(graph, ordering)
    rank = {node: i for i, node in enumerate(nodelist)}
    tokens = []
    for i, node in enumerate(nodelist):
        previous = sorted((i - rank[neighbor], neighbor) for neighbor in graph[node]
                          if rank[neighbor] < i)
        if not ASCENDING:
            previous.reverse()
        edge_tokens = []
        for distance, neighbor in previous:
            if attributed:
                edge_tokens.extend(_edge_tokens(graph, node, neighbor, distance, rank, selected))
            else:
                edge_tokens.extend([str(distance), PLAIN_EDGE])
        node_tokens = _node_tokens(graph, node, rank, selected) if attributed else [DEFAULT_NODE_TOKEN]
        tokens.extend(node_tokens + edge_tokens if NODE_FIRST else edge_tokens + node_tokens)
    # An element symbol is read greedily, so a bare o straight after a
    # one-letter type would be swallowed: C then o reads as cobalt. That only
    # happens in a graph mixing element types with untyped nodes.
    for before, after in zip(tokens, tokens[1:]):
        if after == DEFAULT_NODE_TOKEN and len(before) == 1 and before.isupper():
            raise ValueError(f'Node type {before!r} followed by an untyped node would read as '
                             f'{before + DEFAULT_NODE_TOKEN!r}; give every node a type or none')
    return tokens


_TOKEN_KINDS = {**{c: ('node_type', RE_NODE_TYPE) for c in 'o*ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
                '(': ('attribute', RE_ATTRIBUTE),
                **{c: ('edge_type', RE_EDGE_TYPE) for c in EDGE_TYPES},
                **{c: ('distance', RE_DISTANCE) for c in '123456789'}}


def token_type(token: str) -> str:
    """Classify a complete serialized token."""
    kind = _TOKEN_KINDS.get(token[:1]) if isinstance(token, str) else None
    if kind is not None and kind[1].fullmatch(token):
        return kind[0]
    return 'unknown'


def _units(tokens):
    """Yield (kind, token, attributes) for nodes and complete edges.

    kind is 'node' or 'edge'. Every edge is a distance plus a type token, so
    edge attributes always map edge_type; the plain type - is dropped again so
    that a graph without edge data round-trips unchanged. Attributes with a
    compact form come first, in the order of NODE_ATTRIBUTES or
    EDGE_ATTRIBUTES, then named ones sorted by key.
    """
    i = 0
    while i < len(tokens):
        token = tokens[i]
        kind = token_type(token)
        attributes = {}
        if kind == 'distance':
            kind = 'edge'
            i += 1
            if i >= len(tokens) or token_type(tokens[i]) != 'edge_type':
                raise ValueError('A distance must be followed by an edge type: - = or #')
            if tokens[i] != PLAIN_EDGE:
                attributes['edge_type'] = tokens[i]
        elif kind == 'attribute':
            raise ValueError(f'Attribute {token!r} must directly follow a node or an edge type')
        elif kind != 'node_type':
            raise ValueError(f'Unexpected token: {token!r}')
        else:
            kind = 'node'
        compact = NODE_ATTRIBUTES if kind == 'node' else EDGE_ATTRIBUTES
        last, last_named = -1, None
        while i + 1 < len(tokens) and token_type(tokens[i + 1]) == 'attribute':
            i += 1
            key, value = parse_attribute(tokens[i])
            if key in compact:
                if last_named is not None:
                    raise ValueError(f'Attribute {tokens[i]!r} must come before the named '
                                     'attributes')
                if compact.index(key) <= last:
                    raise ValueError(f'Attribute {tokens[i]!r} is duplicated or out of order; '
                                     f'the order is {list(compact)}')
                last = compact.index(key)
            elif key in ATTRIBUTES:
                raise ValueError(f'Attribute {tokens[i]!r} cannot follow {token!r}')
            else:
                if last_named is not None and key <= last_named:
                    raise ValueError(f'Named attribute {tokens[i]!r} is duplicated or out of '
                                     'order; they are sorted by name')
                last_named = key
            if key in attributes:
                raise ValueError(f'Duplicate attribute {tokens[i]!r}')
            attributes[key] = value
        yield kind, token, attributes
        i += 1


def tokens_to_graph(tokens: list[str], *, strict: bool = True) -> nx.Graph:
    """Parse complete tokens with strict distance, edge and block validation.

    strict=False drops stereo annotations on structures that cannot carry them.
    """
    graph = nx.Graph()
    pending = []
    node_marks, edge_marks = {}, {}

    def add_edges(index, edges):
        for distance, attributes in edges:
            previous = index - distance
            if previous < 0:
                raise ValueError(f'Distance {distance} refers to a non-existent previous node')
            if graph.has_edge(index, previous):
                raise ValueError(f'Duplicate edge from node {index} at distance {distance}')
            mark = attributes.pop('cis_trans', None)
            graph.add_edge(index, previous, **attributes)
            if mark is not None:
                edge_marks[index, previous] = mark

    for kind, token, attributes in _units(list(tokens)):
        if kind == 'node':
            index = len(graph)
            mark = attributes.pop('chirality', None)
            if token != DEFAULT_NODE_TOKEN:
                if 'node_type' in attributes:
                    raise ValueError(f'Node {index} gives a node type twice: {token!r} and '
                                     f'{attributes["node_type"]!r}')
                attributes['node_type'] = token
            graph.add_node(index, **attributes)
            if mark is not None:
                node_marks[index] = mark
            if not NODE_FIRST:
                add_edges(index, pending)
                pending.clear()
        else:
            distance = int(token)
            if NODE_FIRST:
                add_edges(len(graph) - 1, [(distance, attributes)])
            else:
                pending.append((distance, attributes))
    if pending:
        raise ValueError('Edge-first SNACK must end with a node token')
    return _resolve_stereo(graph, node_marks, edge_marks, strict)


def graph_to_snack(graph: nx.Graph, ordering=None, *, attributed: bool = True,
                   attributes=None) -> str:
    """Serialize a graph; attributed=False emits only bare nodes and distances."""
    return ''.join(graph_to_tokens(graph, ordering, attributed=attributed, attributes=attributes))


def snack_to_graph(snack: str, *, strict: bool = True) -> nx.Graph:
    return tokens_to_graph(split(snack, pattern='token'), strict=strict)


def split(snack: str, pattern='token') -> list[str]:
    """Split into tokens, distance/type pairs (graph), or per-node blocks.

    All characters must belong to a token; invalid input is never skipped.
    Token mode performs lexical validation; snack_to_graph also checks grammar.
    In graph mode an edge is its distance, type and attributes, and a node is
    its type and attributes. A block is one node with its backward edges.
    """
    if pattern not in ('token', 'graph', 'block'):
        raise ValueError(f'Invalid pattern: {pattern}')
    if not isinstance(snack, str):
        raise TypeError('SNACK must be a string')
    tokens = []
    position = 0
    while position < len(snack):
        match = RE_SPLIT_TOKEN.match(snack, position)
        if match is None:
            raise ValueError(f'Invalid SNACK token at position {position}: {snack[position:position+20]!r}')
        tokens.append(match[0])
        position = match.end()
    if pattern == 'token':
        return tokens
    # Validate token pairing before grouping the original spelling.
    list(_units(tokens))
    if pattern == 'graph':
        parts = []
        for token in tokens:
            if token_type(token) in ('edge_type', 'attribute'):
                parts[-1] += token
            else:
                parts.append(token)
        return parts
    blocks = []
    block = ''
    for token in tokens:
        if token_type(token) == 'attribute':
            # A node's attribute stays in the block the node closed.
            if block:
                block += token
            else:
                blocks[-1] += token
            continue
        if NODE_FIRST and token_type(token) == 'node_type' and block:
            blocks.append(block)
            block = ''
        if NODE_FIRST and not block and token_type(token) != 'node_type':
            raise ValueError('Node-first blocks must start with a node')
        block += token
        if not NODE_FIRST and token_type(token) == 'node_type':
            blocks.append(block)
            block = ''
    if block:
        if not NODE_FIRST:
            raise ValueError('Edge-first blocks must end with a node')
        blocks.append(block)
    return blocks


def filter(snack, key='node_type'):
    """Extract complete tokens of one kind: node, distance, edge or attribute."""
    for prefix, kind in [('node', 'node_type'), ('dist', 'distance'), ('edge', 'edge_type'),
                         ('attr', 'attribute')]:
        if key.startswith(prefix):
            return [token for token in split(snack) if token_type(token) == kind]
    raise ValueError(f'Invalid key: {key}')


def relabel_nodes(graph: nx.Graph, order: list) -> nx.Graph:
    """Relabel nodes to positions in order, updating stereo references."""
    if len(order) != len(graph) or set(order) != set(graph):
        raise ValueError('order must contain every graph node exactly once')
    mapping = {node: i for i, node in enumerate(order)}
    result = nx.relabel_nodes(graph, mapping)
    for node, data in result.nodes(data=True):
        if 'chirality' in data:
            ligands, sign = data['chirality']
            data['chirality'] = (tuple(None if n is None else mapping[n] for n in ligands), sign)
    for u, v, data in result.edges(data=True):
        if 'cis_trans' in data:
            (end_a, ref_a), (end_b, ref_b), mark = data['cis_trans']
            data['cis_trans'] = ((mapping[end_a], mapping[ref_a]), (mapping[end_b], mapping[ref_b]), mark)
    return result


def _node_color(data, node_key):
    return (data.get(node_key), data.get('isotope', 0),
            data.get('hydrogens'), data.get('charge', 0) or 0)


def _edge_color(data, edge_key):
    return (data.get(edge_key), data.get('weight'))


def count_automorphisms(graph: nx.Graph, node_key='node_type', edge_key='edge_type', *,
                        stereo: bool = False, max_automorphisms: int = 10**6) -> int:
    """Return the exact group order, respecting node and edge types and attributes.

    Node colors combine node_key with isotope, hydrogens and charge. Edge
    colors combine edge_key with weight. Each edge becomes a separately
    colored incidence vertex. Orbit-stabilizer products
    avoid pynauty's floating-point group-size limit.

    stereo=True also preserves 'chirality'/'cis_trans' (meso 2,3-butanediol: 1,
    not 2), by enumeration up to max_automorphisms.
    """
    _validate_graph(graph)
    if stereo and (any('chirality' in data for _, data in graph.nodes(data=True))
                   or any('cis_trans' in data for _, _, data in graph.edges(data=True))):
        return _count_stereo_automorphisms(graph, node_key, edge_key, max_automorphisms)
    rank = {node: i for i, node in enumerate(graph)}
    adjacency = {i: set() for i in range(len(graph))}
    colors = defaultdict(set)
    for node, attributes in graph.nodes(data=True):
        colors[('node', _node_color(attributes, node_key))].add(rank[node])
    for index, (u, v, attributes) in enumerate(graph.edges(data=True), len(graph)):
        colors[('edge', _edge_color(attributes, edge_key))].add(index)
        adjacency[index] = {rank[u], rank[v]}
        adjacency[rank[u]].add(index)
        adjacency[rank[v]].add(index)
    coloring = list(colors.values())
    result = 1
    while True:
        nauty_graph = pynauty.Graph(number_of_vertices=len(adjacency), directed=False,
                                   adjacency_dict=adjacency, vertex_coloring=coloring)
        orbits = pynauty.autgrp(nauty_graph)[3]
        sizes = Counter(orbits)
        orbit = next((orbit for orbit, size in sizes.items() if size > 1), None)
        if orbit is None:
            return result
        result *= sizes[orbit]
        vertex = orbits.index(orbit)
        group = next(group for group in coloring if vertex in group)
        group.remove(vertex)
        coloring.append({vertex})


def _count_stereo_automorphisms(graph, node_key, edge_key, limit):
    from networkx.algorithms.isomorphism import GraphMatcher
    chirality = {node: _check_chirality(graph, node)
                 for node, data in graph.nodes(data=True) if 'chirality' in data}
    cis_trans = {(u, v): _check_cis_trans(graph, u, v)
                 for u, v, data in graph.edges(data=True) if 'cis_trans' in data}

    def node_match(a, b):
        return _node_color(a, node_key) == _node_color(b, node_key) and ('chirality' in a) == ('chirality' in b)

    def edge_match(a, b):
        return _edge_color(a, edge_key) == _edge_color(b, edge_key) and ('cis_trans' in a) == ('cis_trans' in b)

    def preserves(mapping):
        for node, (ligands, sign) in chirality.items():
            image_ligands, image_sign = chirality[mapping[node]]
            mapped = [None if n is None else mapping[n] for n in ligands]
            # The mapped center must have the same handedness as its image.
            if perm_parity(image_ligands, mapped) * (1 if sign == image_sign else -1) != 1:
                return False
        for (u, v), (refs, mark) in cis_trans.items():
            new_refs = {mapping[u]: mapping[refs[u]], mapping[v]: mapping[refs[v]]}
            if _cis_trans_relative(graph, mapping[u], mapping[v], new_refs) != mark:
                return False
        return True

    count = 0
    for enumerated, mapping in enumerate(GraphMatcher(graph, graph, node_match, edge_match)
                                         .isomorphisms_iter(), 1):
        if enumerated > limit:
            raise ValueError(f'More than {limit} graph automorphisms; raise max_automorphisms')
        count += preserves(mapping)
    return count
