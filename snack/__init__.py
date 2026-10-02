# Sequential Notation of Adjacent Connectivity FrameworK (SNACK)
__version__ = "0.1.0"

from snack.dataset import load_dataset
from snack.mol import (
    graph_to_mol,
    mol_to_graph,
    smiles_to_mol,
    mol_to_smiles,
    graph_to_smiles,
    smiles_to_graph,
    smiles_to_snack,
    snack_to_smiles,
    snack_to_mol,
    mol_to_snack,
    canonicalize_smiles,
    randomize_smiles,
)
from snack.graph import (
    graph_to_tokens,
    tokens_to_graph,
    graph_to_snack,
    snack_to_graph,
    split,
    filter,
    token_type,
    count_automorphisms,
    relabel_nodes,
)
from snack import graph as _graph
from snack.tokenizer import SnackTokenizer, basic_vocabulary


class config:
    NODE_FIRST = _graph.NODE_FIRST
    ASCENDING = _graph.ASCENDING

    @classmethod
    def set(cls, node_first: bool = False, ascending: bool = False):
        cls.NODE_FIRST = _graph.NODE_FIRST = node_first
        cls.ASCENDING = _graph.ASCENDING = ascending
