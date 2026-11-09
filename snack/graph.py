# ordering functions are adapted from:
# https://github.com/Genentech/bandwidth-graph-generation/blob/main/graph_gen/data/bandwidth.py
import re
import random
import math
from numbers import Integral, Real
from operator import itemgetter
from collections import deque, defaultdict, Counter
from urllib.parse import quote, unquote
import networkx as nx
import pynauty

NODE_FIRST = False
ASCENDING = False

# Bare v represents a node without attributes; typed nodes use brackets.
DEFAULT_NODE_TOKEN = 'v'


# The separator is used to distinguish between distance tokens.
# E.g. 1_2_ indicates two distance tokens.
# When edge_type is available, edge_type is instead of the separator.
# E.g. 1-2= indicates two edges with edge types '-' and '=' respectively.
SEPARATOR = '_'


NODE_TYPE_STR = r'v|\[[^\[\]\r\n]+\]'
DISTANCE_STR = r'[1-9][0-9]*_?'
EDGE_TYPE_STR = r'[-=#au]|\([^()\r\n]+\)'
STEREO_STR = r'\{[^{}\r\n]+\}'
SIMPLE_EDGE_TYPES = frozenset({'-', '=', '#', 'a', 'u'})

RE_NODE_TYPE = re.compile(NODE_TYPE_STR)
RE_DISTANCE = re.compile(DISTANCE_STR)
RE_EDGE_TYPE = re.compile(EDGE_TYPE_STR)
RE_STEREO = re.compile(STEREO_STR)
RE_SPLIT_TOKEN = re.compile(f'{NODE_TYPE_STR}|{DISTANCE_STR}|{EDGE_TYPE_STR}|{STEREO_STR}')

# Stereo tokens are relative to node order; graphs store them order-independently:
#   node 'tetra' = (four ligand IDs, None for an implicit H or lone pair, '@' | '@@')
#   edge 'cis_trans' = ((u, ref_u), (v, ref_v), 'c' | 't')
# '@'/'@@' have the SMILES meaning for the ligand list; 'c' means same side.
NODE_STEREO_TOKENS = {'{@}': '@', '{@@}': '@@'}
EDGE_STEREO_TOKENS = {'{c}': 'c', '{t}': 't'}
STEREO_TOKENS = tuple(NODE_STEREO_TOKENS) + tuple(EDGE_STEREO_TOKENS)
_STEREO_TOKEN = {value: token for token, value in
                 (*NODE_STEREO_TOKENS.items(), *EDGE_STEREO_TOKENS.items())}
_FLIP = {'@': '@@', '@@': '@', 'c': 't', 't': 'c'}


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


def _check_tetra(graph, node):
    value = graph.nodes[node]['tetra']
    try:
        ligands, sign = value
        ligands = tuple(ligands)
        neighbors = set(graph[node])
        valid = (sign in ('@', '@@') and len(neighbors) in (3, 4) and len(ligands) == 4
                 and set(ligands) == neighbors | ({None} if len(neighbors) == 3 else set()))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'Invalid tetra on node {node!r}: {value!r}')
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


def _node_stereo(graph, node, rank):
    # sign holds for the stored ligand order; an odd reordering flips it.
    ligands, sign = _check_tetra(graph, node)
    return sign if perm_parity(ligands, _ligand_listing(graph, node, rank)) == 1 else _FLIP[sign]


def _edge_stereo(graph, u, v, rank):
    _check_cis_trans(graph, u, v)
    refs = {u: _first_substituent(graph, u, v, rank), v: _first_substituent(graph, v, u, rank)}
    return _cis_trans_relative(graph, u, v, refs)


def _resolve_stereo(graph, node_marks, edge_marks, strict=True):
    """Store parsed marks as tetra/cis_trans; invalid ones raise if strict, else are dropped."""
    rank = {node: node for node in graph}  # parsed nodes are 0..n-1 in sequence order
    for node, sign in node_marks.items():
        if graph.degree(node) not in (3, 4):
            if strict:
                raise ValueError(f'Tetrahedral stereo on node {node} requires 3 or 4 neighbors')
            continue
        graph.nodes[node]['tetra'] = (tuple(_ligand_listing(graph, node, rank)), sign)
    for (u, v), mark in edge_marks.items():
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


def _weight_text(weight):
    if isinstance(weight, bool) or not isinstance(weight, Real):
        raise ValueError('Edge weights must be finite real numbers')
    if isinstance(weight, Integral):
        return str(int(weight))
    if not math.isfinite(weight):
        raise ValueError('Edge weights must be finite real numbers')
    return repr(float(weight))


def _encode_edge(attributes):
    edge_type = attributes.get('edge_type')
    if edge_type is not None and (not isinstance(edge_type, str) or not edge_type):
        raise ValueError('edge_type must be a nonempty string')
    if 'weight' in attributes:
        weight = _weight_text(attributes['weight'])
        if edge_type is None:
            return f'(weight={weight})'
        return f'(type={quote(edge_type, safe="")};weight={weight})'
    if edge_type is None:
        return None
    if edge_type in SIMPLE_EDGE_TYPES:
        return edge_type
    return f'({quote(edge_type, safe="=")})' if not edge_type.startswith(
        ('weight=', 'type=')) else f'({quote(edge_type, safe="")})'


def _decode_edge(token):
    if token in SIMPLE_EDGE_TYPES:
        return {'edge_type': token}
    body = token[1:-1]
    attributes = {}
    if body.startswith('type='):
        match = re.fullmatch(r'type=(.+);weight=(.+)', body)
        if match is None:
            raise ValueError(f'Invalid typed weight token: {token}')
        attributes['edge_type'] = _unquote_type(match[1])
        body = 'weight=' + match[2]
    if body.startswith('weight='):
        raw = body[7:]
        if not re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?', raw):
            raise ValueError(f'Invalid weight token: {token}')
        weight = float(raw) if any(c in raw for c in '.eE') else int(raw)
        _weight_text(weight)
        attributes['weight'] = weight
    else:
        attributes['edge_type'] = _unquote_type(body)
    return attributes


def _unquote_type(text):
    if re.search(r'%(?![0-9A-Fa-f]{2})', text):
        raise ValueError(f'Invalid escaped edge type: {text}')
    try:
        return unquote(text, errors='strict')
    except UnicodeError as exc:
        raise ValueError(f'Invalid escaped edge type: {text}') from exc


def graph_to_tokens(graph: nx.Graph, ordering=None, *,
                    attributed: bool = True) -> list[str]:
    """Encode a simple graph, using its iteration order when ordering is None.

    Set attributed=False to discard node types, edge types, weights and stereo
    while preserving connectivity. Otherwise weights use (weight=3), complex
    types use (contact), and edges with both use (type=contact;weight=3). The
    simple types -, =, #, a and u remain bare. Node 'tetra' and edge 'cis_trans'
    become {@}/{@@} and {c}/{t}. Other attributes are not serialized.
    """
    _validate_graph(graph)
    nodelist = list(graph) if ordering is None else graph_order(graph, ordering)
    rank = {node: i for i, node in enumerate(nodelist)}
    tokens = []
    for i, node in enumerate(nodelist):
        edge_tokens = []
        previous = sorted((i - rank[neighbor], neighbor) for neighbor in graph[node]
                          if rank[neighbor] < i)
        if not ASCENDING:
            previous.reverse()
        for distance, neighbor in previous:
            attributes = dict(graph.edges[node, neighbor]) if attributed else {}
            edge = _encode_edge(attributes)
            edge_tokens.extend([f'{distance}_'] if edge is None else [str(distance), edge])
            if 'cis_trans' in attributes:
                edge_tokens.append(_STEREO_TOKEN[_edge_stereo(graph, node, neighbor, rank)])
        node_type = graph.nodes[node].get('node_type') if attributed else None
        node_type = DEFAULT_NODE_TOKEN if node_type is None else node_type
        if not isinstance(node_type, str) or not RE_NODE_TYPE.fullmatch(node_type):
            raise ValueError(f'Invalid node_type: {node_type!r}; use a bracketed token')
        node_tokens = [node_type]
        if attributed and 'tetra' in graph.nodes[node]:
            node_tokens.append(_STEREO_TOKEN[_node_stereo(graph, node, rank)])
        tokens.extend(node_tokens + edge_tokens if NODE_FIRST else edge_tokens + node_tokens)
    return tokens


def token_type(token: str) -> str:
    """Classify a complete serialized token; complex edge tokens include ()."""
    if not isinstance(token, str):
        return 'unknown'
    for name, pattern in [('node_type', RE_NODE_TYPE), ('distance', RE_DISTANCE),
                          ('edge_type', RE_EDGE_TYPE), ('stereo', RE_STEREO)]:
        if pattern.fullmatch(token):
            return name
    return 'unknown'


def _units(tokens):
    """Yield (kind, token, attributes, stereo) for nodes and complete edges."""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        kind = token_type(token)
        attributes = None
        if kind == 'distance':
            kind = 'edge'
            if token.endswith('_'):
                attributes = {}
            else:
                i += 1
                if i >= len(tokens) or token_type(tokens[i]) != 'edge_type':
                    raise ValueError('A distance must end in _ or be followed by an edge type')
                attributes = _decode_edge(tokens[i])
        elif kind == 'stereo':
            raise ValueError(f'Stereo token {token!r} must directly follow a node or an edge')
        elif kind != 'node_type':
            raise ValueError(f'Unexpected token: {token!r}')
        stereo = None
        if i + 1 < len(tokens) and token_type(tokens[i + 1]) == 'stereo':
            i += 1
            values = NODE_STEREO_TOKENS if kind == 'node_type' else EDGE_STEREO_TOKENS
            if tokens[i] not in values:
                target = 'a node' if kind == 'node_type' else 'an edge'
                raise ValueError(f'Invalid stereo token {tokens[i]!r} after {target}; '
                                 f'expected one of {sorted(values)}')
            stereo = values[tokens[i]]
        yield kind, token, attributes, stereo
        i += 1


def tokens_to_graph(tokens: list[str], *, strict: bool = True) -> nx.Graph:
    """Parse complete tokens with strict distance, edge and block validation.

    strict=False drops stereo annotations on structures that cannot carry them.
    """
    graph = nx.Graph()
    pending = []
    node_marks, edge_marks = {}, {}

    def add_edges(index, edges):
        for distance, attributes, stereo in edges:
            previous = index - distance
            if previous < 0:
                raise ValueError(f'Distance {distance} refers to a non-existent previous node')
            if graph.has_edge(index, previous):
                raise ValueError(f'Duplicate edge from node {index} at distance {distance}')
            graph.add_edge(index, previous, **attributes)
            if stereo is not None:
                edge_marks[index, previous] = stereo

    for kind, token, attributes, stereo in _units(list(tokens)):
        if kind == 'node_type':
            index = len(graph)
            graph.add_node(index, **({} if token == DEFAULT_NODE_TOKEN else {'node_type': token}))
            if stereo is not None:
                node_marks[index] = stereo
            if not NODE_FIRST:
                add_edges(index, pending)
                pending.clear()
        else:
            distance = int(token.removesuffix(SEPARATOR))
            if NODE_FIRST:
                add_edges(len(graph) - 1, [(distance, attributes, stereo)])
            else:
                pending.append((distance, attributes, stereo))
    if pending:
        raise ValueError('Edge-first SNACK must end with a node token')
    return _resolve_stereo(graph, node_marks, edge_marks, strict)


def graph_to_snack(graph: nx.Graph, ordering=None, *,
                   attributed: bool = True) -> str:
    """Serialize a graph; attributed=False emits only bare nodes and distances."""
    return ''.join(graph_to_tokens(graph, ordering, attributed=attributed))


def snack_to_graph(snack: str, *, strict: bool = True) -> nx.Graph:
    return tokens_to_graph(split(snack, pattern='token'), strict=strict)


def split(snack: str, pattern='token') -> list[str]:
    """Split into tokens, distance/type pairs (graph), or per-node blocks.

    All characters must belong to a token; invalid input is never skipped.
    Token mode performs lexical validation; snack_to_graph also checks grammar.
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
            if token_type(token) in ('edge_type', 'stereo'):
                parts[-1] += token
            else:
                parts.append(token)
        return parts
    blocks = []
    block = ''
    for token in tokens:
        if token_type(token) == 'stereo':
            # A node's annotation stays in the block the node closed.
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
    """Extract complete tokens, excluding digits inside node/edge attributes."""
    for prefix, kind in [('node', 'node_type'), ('dist', 'distance'), ('edge', 'edge_type'),
                         ('stereo', 'stereo')]:
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
        if 'tetra' in data:
            ligands, sign = data['tetra']
            data['tetra'] = (tuple(None if n is None else mapping[n] for n in ligands), sign)
    for u, v, data in result.edges(data=True):
        if 'cis_trans' in data:
            (end_a, ref_a), (end_b, ref_b), mark = data['cis_trans']
            data['cis_trans'] = ((mapping[end_a], mapping[ref_a]), (mapping[end_b], mapping[ref_b]), mark)
    return result


def count_automorphisms(graph: nx.Graph, node_key='node_type', edge_key='edge_type', *,
                        stereo: bool = False, max_automorphisms: int = 10**6) -> int:
    """Return the exact group order, respecting node/edge types and weights.

    Each edge becomes a separately colored incidence vertex. Original vertices
    and edge vertices have disjoint color classes, including missing attributes.
    Orbit-stabilizer products avoid pynauty's floating-point group-size limit.

    stereo=True also preserves 'tetra'/'cis_trans' (meso 2,3-butanediol: 1, not 2),
    by enumeration up to max_automorphisms.
    """
    _validate_graph(graph)
    if stereo and (any('tetra' in data for _, data in graph.nodes(data=True))
                   or any('cis_trans' in data for _, _, data in graph.edges(data=True))):
        return _count_stereo_automorphisms(graph, node_key, edge_key, max_automorphisms)
    missing = object()
    rank = {node: i for i, node in enumerate(graph)}
    adjacency = {i: set() for i in range(len(graph))}
    colors = defaultdict(set)
    for node, attributes in graph.nodes(data=True):
        colors[('node', attributes.get(node_key, missing))].add(rank[node])
    for index, (u, v, attributes) in enumerate(graph.edges(data=True), len(graph)):
        colors[('edge', attributes.get(edge_key, missing), attributes.get('weight', missing))].add(index)
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
    missing = object()
    tetra = {node: _check_tetra(graph, node) for node, data in graph.nodes(data=True) if 'tetra' in data}
    cis_trans = {(u, v): _check_cis_trans(graph, u, v)
                 for u, v, data in graph.edges(data=True) if 'cis_trans' in data}

    def node_match(a, b):
        return a.get(node_key, missing) == b.get(node_key, missing) and ('tetra' in a) == ('tetra' in b)

    def edge_match(a, b):
        return (a.get(edge_key, missing) == b.get(edge_key, missing)
                and a.get('weight', missing) == b.get('weight', missing)
                and ('cis_trans' in a) == ('cis_trans' in b))

    def preserves(mapping):
        for node, (ligands, sign) in tetra.items():
            image_ligands, image_sign = tetra[mapping[node]]
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
