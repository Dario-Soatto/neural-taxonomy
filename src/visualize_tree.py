import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import re
import textwrap
from pathlib import Path

BASE_DIR = Path("experiments/wiki_biographies/hierarchy_results__standard")
LABELED_TREE = BASE_DIR / "labeled_hierarchical_tree.gml"
RAW_TREE = BASE_DIR / "hierarchical_tree.gml"
FIXED_TREE = BASE_DIR / "hierarchical_tree_fixed.gml"
OUTPUT_PATH = BASE_DIR / "tree_viz_labeled.png"


def _fix_gml_numbers(input_path: Path, output_path: Path) -> Path:
    content = input_path.read_text()
    content = re.sub(r'NP\.FLOAT64\(([\d.e+-]+)\)', r'\1', content)
    output_path.write_text(content)
    return output_path


def _get_tree_path() -> Path:
    if LABELED_TREE.exists():
        return LABELED_TREE
    return RAW_TREE


def _short_label(text: str, max_len: int = 40) -> str:
    if not text:
        return ""
    match = re.search(r'"([^"]+)"', str(text))
    if match:
        text = match.group(1)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return textwrap.fill(text, width=18)


def _hierarchy_pos(G: nx.DiGraph):
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
        raise ValueError("No root nodes found in graph")
    if len(roots) > 1:
        root = "__root__"
        G = G.copy()
        G.add_node(root)
        for r in roots:
            G.add_edge(root, r)
    else:
        root = roots[0]

    # Order leaves left-to-right and set internal nodes to mean of children.
    leaf_x = {}
    x_counter = [0]

    def set_positions(node):
        children = list(G.successors(node))
        if not children:
            leaf_x[node] = x_counter[0]
            x_counter[0] += 1
            return leaf_x[node]
        child_x = [set_positions(c) for c in children]
        leaf_x[node] = sum(child_x) / len(child_x)
        return leaf_x[node]

    set_positions(root)

    depths = {root: 0}
    for parent, child in nx.bfs_edges(G, root):
        depths[child] = depths[parent] + 1

    max_depth = max(depths.values()) if depths else 1
    pos = {n: (leaf_x[n], -depths[n]) for n in G.nodes()}

    # Normalize x to [0, 1]
    xs = [p[0] for p in pos.values()]
    if xs:
        min_x, max_x = min(xs), max(xs)
        if max_x > min_x:
            pos = {n: ((x - min_x) / (max_x - min_x), y) for n, (x, y) in pos.items()}
    return pos


tree_path = _get_tree_path()
fixed_path = _fix_gml_numbers(tree_path, FIXED_TREE)
# Use "id" for node keys so a "label" node attribute is preserved.
G = nx.read_gml(fixed_path, label="id")

# Load cluster (leaf) labels if available
label_map = {}
labels_path = Path("experiments/wiki_biographies/models/cluster_labels.csv")
if labels_path.exists():
    labels_df = pd.read_csv(labels_path)
    label_map = dict(zip(labels_df["node_id"].astype(str), labels_df["label"]))

# Load inner node labels if available
inner_label_map = {}
inner_labels_path = BASE_DIR / "inner_node_labels.csv"
if inner_labels_path.exists():
    inner_df = pd.read_csv(inner_labels_path)
    inner_label_map = dict(zip(inner_df["node_id"].astype(str), inner_df["label"]))

display_labels = {}
for node in G.nodes():
    node_str = str(node)
    node_label = inner_label_map.get(node_str) or G.nodes[node].get("label") or label_map.get(node_str) or f"Node {node}"
    display_labels[node] = _short_label(str(node_label))

pos = _hierarchy_pos(G)
plt.figure(figsize=(20, 10))
nx.draw(
    G,
    pos,
    labels=display_labels,
    with_labels=True,
    node_size=600,
    font_size=7,
    arrows=True,
    node_color="lightblue",
    edge_color="gray",
)
plt.title("Wiki Biographies Taxonomy")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.show()
