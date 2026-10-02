# ordering functions are adapted from:
# https://github.com/Genentech/bandwidth-graph-generation/blob/main/graph_gen/data/bandwidth.py
import re
import random
import math
from numbers import Integral, Real
from typing import Optional
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
SIMPLE_EDGE_TYPES = frozenset({'-', '=', '#', 'a', 'u'})

RE_NODE_TYPE = re.compile(NODE_TYPE_STR)
RE_DISTANCE = re.compile(DISTANCE_STR)
RE_EDGE_TYPE = re.compile(EDGE_TYPE_STR)
RE_SPLIT_TOKEN = re.compile(f'{NODE_TYPE_STR}|{DISTANCE_STR}|{EDGE_TYPE_STR}')


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


def graph_order(G: nx.Graph, ordering: str = 'bfs') -> list:
    """Order all nodes, including disconnected components and isolated nodes."""
    _validate_graph(G)
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


def graph_to_tokens(graph: nx.Graph, ordering: Optional[str] = None, *,
                    attributed: bool = True) -> list[str]:
    """Encode a simple graph, using its iteration order when ordering is None.

    Set attributed=False to discard node types, edge types and weights while
    preserving connectivity. Otherwise weights use (weight=3), complex types
    use (contact), and edges with both use (type=contact;weight=3). The simple
    types -, =, #, a and u remain bare. Other attributes are not serialized.
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
        node_type = graph.nodes[node].get('node_type') if attributed else None
        node_type = DEFAULT_NODE_TOKEN if node_type is None else node_type
        if not isinstance(node_type, str) or not RE_NODE_TYPE.fullmatch(node_type):
            raise ValueError(f'Invalid node_type: {node_type!r}; use a bracketed token')
        tokens.extend([node_type] + edge_tokens if NODE_FIRST else edge_tokens + [node_type])
    return tokens


def token_type(token: str) -> str:
    """Classify a complete serialized token; complex edge tokens include ()."""
    if not isinstance(token, str):
        return 'unknown'
    for name, pattern in [('node_type', RE_NODE_TYPE), ('distance', RE_DISTANCE),
                          ('edge_type', RE_EDGE_TYPE)]:
        if pattern.fullmatch(token):
            return name
    return 'unknown'


def _units(tokens):
    """Yield nodes or complete edges, rejecting incomplete/unknown tokens."""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        kind = token_type(token)
        if kind == 'node_type':
            yield kind, token, None
        elif kind == 'distance':
            if token.endswith('_'):
                attributes = {}
            else:
                i += 1
                if i >= len(tokens) or token_type(tokens[i]) != 'edge_type':
                    raise ValueError('A distance must end in _ or be followed by an edge type')
                attributes = _decode_edge(tokens[i])
            yield 'edge', token, attributes
        else:
            raise ValueError(f'Unexpected token: {token!r}')
        i += 1


def tokens_to_graph(tokens: list[str]) -> nx.Graph:
    """Parse complete tokens with strict distance, edge and block validation."""
    graph = nx.Graph()
    pending = []

    def add_edges(index, edges):
        for distance, attributes in edges:
            previous = index - distance
            if previous < 0:
                raise ValueError(f'Distance {distance} refers to a non-existent previous node')
            if graph.has_edge(index, previous):
                raise ValueError(f'Duplicate edge from node {index} at distance {distance}')
            graph.add_edge(index, previous, **attributes)

    for kind, token, attributes in _units(list(tokens)):
        if kind == 'node_type':
            index = len(graph)
            graph.add_node(index, **({} if token == DEFAULT_NODE_TOKEN else {'node_type': token}))
            if not NODE_FIRST:
                add_edges(index, pending)
                pending.clear()
        else:
            distance = int(token.removesuffix(SEPARATOR))
            if NODE_FIRST:
                add_edges(len(graph) - 1, [(distance, attributes)])
            else:
                pending.append((distance, attributes))
    if pending:
        raise ValueError('Edge-first SNACK must end with a node token')
    return graph


def graph_to_snack(graph: nx.Graph, ordering: Optional[str] = None, *,
                   attributed: bool = True) -> str:
    """Serialize a graph; attributed=False emits only bare nodes and distances."""
    return ''.join(graph_to_tokens(graph, ordering, attributed=attributed))


def snack_to_graph(snack: str) -> nx.Graph:
    return tokens_to_graph(split(snack, pattern='token'))


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
            if token_type(token) == 'edge_type':
                parts[-1] += token
            else:
                parts.append(token)
        return parts
    blocks = []
    block = ''
    for token in tokens:
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
    for prefix, kind in [('node', 'node_type'), ('dist', 'distance'), ('edge', 'edge_type')]:
        if key.startswith(prefix):
            return [token for token in split(snack) if token_type(token) == kind]
    raise ValueError(f'Invalid key: {key}')


def relabel_nodes(graph: nx.Graph, order: list) -> nx.Graph:
    if len(order) != len(graph) or set(order) != set(graph):
        raise ValueError('order must contain every graph node exactly once')
    return nx.relabel_nodes(graph, {node: i for i, node in enumerate(order)})


def count_automorphisms(graph: nx.Graph, node_key='node_type', edge_key='edge_type') -> int:
    """Return the exact group order, respecting node/edge types and weights.

    Each edge becomes a separately colored incidence vertex. Original vertices
    and edge vertices have disjoint color classes, including missing attributes.
    Orbit-stabilizer products avoid pynauty's floating-point group-size limit.
    """
    _validate_graph(graph)
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
