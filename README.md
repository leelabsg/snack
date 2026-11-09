# SNACK

> **SNACK: A Sequential Notation Framework for Probabilistic Graph Generation**

Hohyun Kim\* &nbsp;|&nbsp; Hyesung Kim\* &nbsp;|&nbsp; Min-hwan Oh &nbsp;|&nbsp; Seunggeun Lee

Neural Information Processing Systems (NeurIPS), 2026

[![ArXiv](https://img.shields.io/badge/arXiv-2610.19781-b31b1b.svg)](https://arxiv.org/abs/2610.19781)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/pdf/2610.19781)

---

<p align="center">
   <img src="image/snack_fig.png" alt="SNACK encoding process" width="650px">
</p>

`snack` is a compact string representation for discrete graphs, with helper
utilities to convert between:

- **NetworkX graphs** ↔ **SNACK strings**
- **SMILES / RDKit molecules** ↔ **SNACK strings** (molecular graphs)

Dependencies: `networkx`, `pynauty`, and `rdkit`. The tokenizer and dataset
loader do not require PyTorch or pandas.

## Installation

```bash
pip install snack-graph
```

## Usage

| Function | Purpose |
| --- | --- |
| `graph_to_snack()` / `snack_to_graph()` | Convert between NetworkX graphs and SNACK strings. |
| `smiles_to_snack()` / `snack_to_smiles()` | Convert between SMILES and SNACK strings. |
| `mol_to_snack()` / `snack_to_mol()` | Convert between RDKit molecules and SNACK strings. |
| `smiles_to_graph()` / `graph_to_smiles()` | Convert between SMILES and NetworkX graphs. |
| `mol_to_graph()` / `graph_to_mol()` | Convert between RDKit molecules and NetworkX graphs. |
| `graph_to_tokens()` / `tokens_to_graph()` | Convert between NetworkX graphs and lists of SNACK tokens. |
| `split()` | Split a SNACK string into tokens, distance–edge pairs, or node blocks. |
| `filter()` | Extract node types, edge types, or distances from a SNACK string. |
| `SnackTokenizer()` | Create a tokenizer using the `graph`, `zinc250k`, `moses`, or `guacamol` preset. |
| `tokenizer.encode()` / `tokenizer.decode()` | Convert between SNACK strings and integer token IDs. |
| `tokenizer.get_logit_mask()` | Return an allowed-next-token mask after each token in a sequence. |
| `load_dataset()` | Load a bundled dataset as a dictionary of `train`, `val`, and `test` SNACK-string lists. |
| `count_automorphisms()` | Count graph symmetries while respecting node types, edge types, and weights. |
| `canonicalize_smiles()` / `randomize_smiles()` | Produce canonical SMILES or an alternative SMILES for the same molecule. |



## Examples

### 1) NetworkX graph → SNACK → graph

SNACK specifies node types in brackets (e.g., [node_type]).

```python
import random
import networkx as nx
import snack as snk

random.seed(0)
G = nx.Graph()
G.add_node(0, node_type="[A]")
G.add_node(1, node_type="[B]")
G.add_node(2, node_type="[C]")
G.add_edge(0, 1, edge_type="-")
G.add_edge(1, 2, edge_type="=")

sn = snk.graph_to_snack(G, ordering="uniform")
G2 = snk.snack_to_graph(sn)

print(sn)
print(G2.nodes(data=True))
print(G2.edges(data=True))
```

```text
[A][C]2-1=[B]
[(0, {'node_type': '[A]'}), (1, {'node_type': '[C]'}), (2, {'node_type': '[B]'})]
[(0, 2, {'edge_type': '-'}), (1, 2, {'edge_type': '='})]
```

### 2) Non-attributed graphs

`v` denotes a node. `_` denotes an untyped edge (used to separate consecutive distances).

```python
import networkx as nx
import snack as snk

G = nx.Graph()
G.add_edge(0, 1)
G.add_edge(0, 2)
G.add_edge(2, 3)
G.add_edge(2, 4)

sn = snk.graph_to_snack(G)
tokens = snk.split(sn, pattern="token")

print(sn)
print(tokens)
```

```text
v1_v2_v1_v2_v
['v', '1_', 'v', '2_', 'v', '1_', 'v', '2_', 'v']
```


### 3) Weighted graphs and complex edge types

Complex edge types use parentheses. Use `attributed=False` to retain
only connectivity.

```python
import networkx as nx
import snack as snk

G = nx.Graph()
G.add_edge(0, 1, weight=3)
sn = snk.graph_to_snack(G)
print("weighted:", sn)
print("decoded edge:", snk.snack_to_graph(sn).edges[0, 1])

G.edges[0, 1]["edge_type"] = "contact"
print("type and weight:", snk.graph_to_snack(G))
print("connectivity only:", snk.graph_to_snack(G, attributed=False))
```

```text
weighted: v1(weight=3)v
decoded edge: {'weight': 3}
type and weight: v1(type=contact;weight=3)v
connectivity only: v1_v
```

### 4) SMILES → SNACK → SMILES (round-trip)

`smiles_to_snack()` uses randomized BFS by default. Set a random seed for
reproducible sequences, or use `ordering=None` to preserve input atom order.

```python
import random
import snack as snk

random.seed(0)
smiles = "CCO"  # ethanol
sn = snk.smiles_to_snack(smiles, ordering="bfs")
smiles_rt = snk.snack_to_smiles(sn, canonical=True)

print("snack:", sn)
print("canonical smiles:", smiles_rt)
```

```text
snack: [C]1-[O]2-[C]
canonical smiles: CCO
```

### 5) RDKit molecule ↔ NetworkX graph ↔ SNACK

```python
from rdkit import Chem
import snack as snk

mol = Chem.MolFromSmiles("CCO")
G = snk.mol_to_graph(mol)
sn = snk.mol_to_snack(mol)

print("snack:", sn)
print("from graph:", snk.graph_to_smiles(G))
print("from snack:", Chem.MolToSmiles(snk.snack_to_mol(sn)))
```

```text
snack: [C]1-[C]1-[O]
from graph: CCO
from snack: CCO
```

`smiles_to_graph()` accepts SMILES directly; `graph_to_mol()` returns an RDKit
molecule. Molecular conversion preserves isotopes and atom maps. It omits
stereochemistry unless `stereo=True` is passed (see example 18). Dative bonds,
invalid valence, and unsupported molecular tokens
raise `ValueError`. Connectivity-only graphs cannot be decoded as molecules.

### 6) Hydrogen counts and molecular conversion options

With `use_hydrogens=True`, every atom includes its hydrogen count, including
`H0` for zero hydrogens. These counts can distinguish a radical from a molecule
whose remaining valence is filled by implicit hydrogens:

```python
import snack as snk

with_h = snk.smiles_to_snack("[CH3]", use_hydrogens=True)
without_h = snk.smiles_to_snack("[CH3]", use_hydrogens=False)

print("with H counts:", with_h, "->", snk.snack_to_smiles(with_h))
print("without H counts:", without_h, "->", snk.snack_to_smiles(without_h))
```

```text
with H counts: [CH3] -> [CH3]
without H counts: [C] -> C
```

Hydrogen counts are omitted by default; separate hydrogen nodes remain in the
graph. `use_hydrogens="auto"` writes a count only where it differs from the
inferred one (radicals such as `[CH3]`), so ordinary molecules keep their
default spelling. `use_charge=True` preserves formal charges, and `kekulize=True` expresses
aromatic bonds using single and double bonds. Omitting hydrogen counts or
charges can change molecular identity or prevent decoding.

### 7) Splitting SNACK: `split`

```python
import random
import snack as snk

random.seed(0)
sn = snk.smiles_to_snack("c1ccccc1", ordering="cm", kekulize=False)  # benzene

tokens = snk.split(sn, pattern="token")
elements = snk.split(sn, pattern="graph")
blocks = snk.split(sn, pattern="block")

print("tokens:", tokens)
print("graph elements:", elements)
print("blocks:", blocks)
```

```text
tokens: ['[C]', '1', 'a', '[C]', '2', 'a', '[C]', '2', 'a', '[C]', '2', 'a', '[C]', '2', 'a', '1', 'a', '[C]']
graph elements: ['[C]', '1a', '[C]', '2a', '[C]', '2a', '[C]', '2a', '[C]', '2a', '1a', '[C]']
blocks: ['[C]', '1a[C]', '2a[C]', '2a[C]', '2a[C]', '2a1a[C]']
```

### 8) Token-level operations: `filter` and `token_type`

```python
import random
import snack as snk

random.seed(0)
sn = snk.smiles_to_snack("c1ccccc1", ordering="bfs", kekulize=False)  # benzene

tokens = snk.split(sn, pattern="token")
node_tokens = snk.filter(sn, key="node_type")
edge_tokens = snk.filter(sn, key="edge_type")
dist_tokens = snk.filter(sn, key="distance")

print("num tokens:", len(tokens))
print("first 10:", tokens[:10])
print("node_type tokens:", node_tokens[:5])
print("edge_type tokens:", edge_tokens[:5])
print("distance tokens:", dist_tokens)
print("types:", [snk.token_type(t) for t in tokens[:10]])
```

```text
num tokens: 18
first 10: ['[C]', '1', 'a', '[C]', '2', 'a', '[C]', '2', 'a', '[C]']
node_type tokens: ['[C]', '[C]', '[C]', '[C]', '[C]']
edge_type tokens: ['a', 'a', 'a', 'a', 'a']
distance tokens: ['1', '2', '2', '2', '2', '1']
types: ['node_type', 'distance', 'edge_type', 'node_type', 'distance', 'edge_type', 'node_type', 'distance', 'edge_type', 'node_type']
```

### 9) Global formatting: node-first vs edge-first, ascending vs descending

Encoding and decoding must use the same `node_first` setting; it is not stored
in the string. The tokenizer captures the settings when it is constructed.

Supported orderings are `None` (input order), `"bfs"`, `"cm"` (Cuthill–McKee),
`"uniform"`, and `"neighbor"`. All named orderings use Python's `random` module.
Graph and RDKit-molecule conversion preserve input order by default.

```python
import random
import snack as snk

random.seed(0)
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine

snk.config.set(node_first=False, ascending=False)
s1 = snk.smiles_to_snack(smiles, ordering="neighbor")

random.seed(0)
snk.config.set(node_first=True, ascending=True)
s2 = snk.smiles_to_snack(smiles, ordering="neighbor")

print("edge-first / descending:", s1)
print("node-first / ascending :", s2)
snk.config.set()  # Restore defaults for subsequent examples.
```

```text
edge-first / descending: [C]1-[N]1-[C]1=[O]2-[N]1-[C]2-[C]1-[N]2=[C]1-[N]9-2-[C]1=[O]5=3-[C]4-[C]
node-first / ascending : [C][N]1-[C]1-[O]1=[N]2-[C]1-[C]2-[N]1-[C]2=[N]1-[C]2-9-[O]1=[C]3-5=[C]4-
```

### 10) Automorphism counting on a graph

The result is an exact integer. Node types, edge types, and weights are treated
as graph labels when determining symmetry.

```python
import networkx as nx
import snack as snk

# A 4-cycle has 8 automorphisms (dihedral group D4)
G = nx.cycle_graph(4)
aut = snk.count_automorphisms(G, node_key="node_type", edge_key="edge_type")
print("automorphisms:", aut)
```

```text
automorphisms: 8
```

### 11) Canonical and randomized SMILES

Both helpers return `None` for invalid SMILES.

```python
import random
import snack as snk

random.seed(0)
smi1 = snk.randomize_smiles('c1ccccc1CCC')
smi2 = snk.randomize_smiles(smi1)
canon = snk.canonicalize_smiles(smi1)

print(smi1)
print(smi2)
print(canon)
```

```text
C(C)Cc1ccccc1
C(CC)c1ccccc1
CCCc1ccccc1
```

### 12) SNACK → token IDs → SNACK

```python
import snack as snk

tokenizer = snk.SnackTokenizer("graph", max_nodes=2)
ids = tokenizer.encode("v1_v")

print(ids)
print(tokenizer.decode(ids))
```

```text
[1, 3, 4, 3, 2]
v1_v
```

`encode()` adds `[bos]` (beginning) and `[eos]` (end). Presets provide different
vocabularies:

| Preset | Vocabulary | Default `max_nodes` |
| --- | --- | --- |
| `"graph"` | Non-attributed graphs | 128 |
| `"zinc250k"` | ZINC250k atom types | 38 |
| `"moses"` | MOSES atom types | 27 |
| `"guacamol"` | GuacaMol atom types | 88 |

### 13) Can a node or EOS come next?

`v1_` starts an edge but still needs its next node. A mask entry is `True` when
that token is allowed. `[:-1]` removes EOS so we can continue the sequence.

```python
import snack as snk

tokenizer = snk.SnackTokenizer("graph", ordering="bfs", max_nodes=2)
ids = tokenizer.encode("v1_")[:-1]
mask = tokenizer.next_token_mask(ids)

print("node allowed:", mask[tokenizer.token_to_id["v"]])
print("EOS allowed:", mask[tokenizer.eos_id])
```

```text
node allowed: True
EOS allowed: False
```

### 14) Masks for a whole sequence

`get_logit_mask()` returns the same check at every position. Columns follow
`tokenizer.tokens`; each row describes what can come next after its prefix.

```python
import snack as snk

tokenizer = snk.SnackTokenizer("graph", ordering="bfs", max_nodes=2)
ids = tokenizer.encode("v1_v")
masks = tokenizer.get_logit_mask(ids[:-1])

print("vocabulary:", tokenizer.tokens)
print("after BOS:", masks[0])
print("after v:", masks[1])
print("after v1_:", masks[2])
print("after v1_v:", masks[3])
```

```text
vocabulary: ('[pad]', '[bos]', '[eos]', 'v', '1_')
after BOS: [False, False, False, True, False]
after v: [False, False, True, False, True]
after v1_: [False, False, False, True, False]
after v1_v: [False, False, True, False, False]
```

For training, row `i` predicts `ids[i + 1]`. Before sampling, set model scores
(logits) to `-inf` where the mask is `False`.

### 15) Stop at the node limit

With `max_nodes=2`, completing `v1_v` leaves only EOS allowed.

```python
import snack as snk

tokenizer = snk.SnackTokenizer("graph", ordering="bfs", max_nodes=2)
ids = tokenizer.encode("v1_v")[:-1]
mask = tokenizer.next_token_mask(ids)

print("EOS allowed:", mask[tokenizer.eos_id])
print("another edge allowed:", mask[tokenizer.token_to_id["1_"]])
```

```text
EOS allowed: True
another edge allowed: False
```

Use `allow_disconnected=True` for multiple components. `ordering="bfs"`
restricts generation to BFS order; `ordering=None` disables that constraint.

### 16) Molecular valence masks

An oxygen with no existing bonds can accept a single or double bond, but the
preset blocks a triple bond:

```python
import snack as snk

tokenizer = snk.SnackTokenizer("moses", ordering="bfs")
ids = tokenizer.encode("[O]1")[:-1]
mask = tokenizer.next_token_mask(ids)

print("single bond:", mask[tokenizer.token_to_id["-"]])
print("double bond:", mask[tokenizer.token_to_id["="]])
print("triple bond:", mask[tokenizer.token_to_id["#"]])
```

```text
single bond: True
double bond: True
triple bond: False
```

Molecular presets use `-`, `=`, and `#` bonds and check RDKit validity before EOS.
`valence=False` disables chemical checks. If every mask entry is `False`,
backtrack or restart generation.

### 17) Loading bundled datasets

```python
import snack as snk

data = snk.load_dataset("planar")
train, val, test = data["train"], data["val"], data["test"]
print("split sizes:", len(train), len(val), len(test))
```

```text
split sizes: 128 32 40
```

Each split is a list of SNACK strings. Supported names are `community_small`,
`ego`, `ego_small`, `enzymes`, `grid`, `grid_small`, `lobster`, `planar`, and `sbm`. 
See [dataset details](snack/data/README.md) for details. 

### 18) Stereochemistry and radicals

`stereo=True` keeps tetrahedral centers (`{@}`, `{@@}` after a node) and
double-bond cis/trans (`{c}`, `{t}` after `=`). A missing annotation means
"unspecified". `use_hydrogens="auto"` writes an H count only where it differs
from the inferred one, so radicals survive.

```python
import snack as snk

for smiles in ["C[C@H](N)C(=O)O", "F/C=C/F", "[CH2]C"]:
    sn = snk.smiles_to_snack(smiles, ordering=None, stereo=True, use_hydrogens="auto")
    print(f"{smiles:16} {sn:32} {snk.snack_to_smiles(sn)}")
```

```text
C[C@H](N)C(=O)O  [C]1-[C]{@}1-[N]2-[C]1=[O]2-[O]  C[C@H](N)C(=O)O
F/C=C/F          [F]1-[C]1={t}[C]1-[F]            F/C=C/F
[CH2]C           [CH2]1-[C]                       [CH2]C
```

Annotations are relative to the node order and recomputed for every order:

- `{@}`/`{@@}` have the SMILES meaning, with neighbors listed by increasing
  node index. An implicit H or lone pair takes the center's own position.
- `{c}`/`{t}` tell whether the smallest-index neighbors at the two ends of the
  double bond are on the same or opposite side (local cis/trans, not CIP E/Z).

Graphs store order-independent attributes, node `tetra` and edge
`cis_trans`, so any `ordering` stays correct. RDKit decides which atoms and
bonds are stereogenic. Decoding drops annotations that are not
(`stereo="strict"` raises instead), and `is_canonical_snack()` checks that
every annotation and H count is necessary. `count_automorphisms(G,
stereo=True)` respects stereo, e.g. 1 instead of 2 for meso 2,3-butanediol.

`SnackTokenizer(preset, stereo=True)` appends `{@} {@@} {c} {t}` to the
vocabulary without changing existing token IDs. Generated outputs always
decode but may carry unnecessary annotations; filter them with
`is_canonical_snack()`. `hydrogens="auto"` allows generating H-count atoms.

## Not supported

- Axial chirality (allenes, atropisomers), square-planar/octahedral metal
  centers, enhanced stereo groups (and/or), and unknown ("wavy") stereo.
- 3D coordinates, conformers, and tautomers.
