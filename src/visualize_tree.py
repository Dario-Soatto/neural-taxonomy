import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import re

# Read and fix the GML file
with open('experiments/wiki_biographies/hierarchy_results__standard/hierarchical_tree.gml', 'r') as f:
    content = f.read()

content = re.sub(r'NP\.FLOAT64\(([\d.e+-]+)\)', r'\1', content)

with open('experiments/wiki_biographies/hierarchy_results__standard/hierarchical_tree_fixed.gml', 'w') as f:
    f.write(content)

G = nx.read_gml('experiments/wiki_biographies/hierarchy_results__standard/hierarchical_tree_fixed.gml')

# Load cluster labels
labels_df = pd.read_csv('experiments/wiki_biographies/models/cluster_labels.csv')
label_map = dict(zip(labels_df['node_id'].astype(str), labels_df['label']))

# Extract short labels (just the quoted part)
def get_short_label(full_label):
    match = re.search(r'"([^"]+)"', str(full_label))
    return match.group(1) if match else str(full_label)[:20]

# Create display labels
display_labels = {}
for node in G.nodes():
    node_str = str(node)
    if node_str in label_map:
        display_labels[node] = get_short_label(label_map[node_str])
    else:
        display_labels[node] = f"Cluster {node}"

# Visualize
plt.figure(figsize=(24, 18))
pos = nx.spring_layout(G, k=3, iterations=100)
nx.draw(G, pos, labels=display_labels, with_labels=True, 
        node_size=500, font_size=7, arrows=True,
        node_color='lightblue', edge_color='gray')
plt.title("Wiki Biographies Taxonomy")
plt.tight_layout()
plt.savefig('experiments/wiki_biographies/hierarchy_results__standard/tree_viz_labeled.png', dpi=200)
plt.show()