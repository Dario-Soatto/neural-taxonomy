import pandas as pd 
import numpy as np 
import hdbscan
import os 
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from adjustText import adjust_text
from numpy.linalg import norm
from sklearn.preprocessing import Normalizer
from tqdm.auto import tqdm
import math
from collections import defaultdict
from utils_summarization import single_pass_summarize_labels_with_examples, discretize_embedding_summaries
from typing import Dict, Any, Optional, List, Tuple, Union
from utils_openai_client import prompt_openai_model, load_model, generate_responses
from prompts import TreeNodeLabelResponse
from tooldantic import ToolBaseModel as BaseModel


def get_root(G):
    """
    Returns the root of the tree (node with no incoming edges).
    Assumes there is exactly one such node.
    """
    for node in G.nodes():
        if G.in_degree(node) == 0:
            return node
    raise ValueError("No root found in the graph.")


def compute_subtree_sizes_and_membership(
        G, 
        node, 
        leaf_node_counts=None, 
        count_attr='subtree_size', 
        leaf_node_ids_attr='leaf_node_ids', 
        recount=False, 
        examples_df=None, 
        cluster_col='cluster',
        max_depth=13
):
    """
    Recursively computes the total number of datapoints in the subtree
    rooted at 'node' and stores it as an attribute 'subtree_size'.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The tree graph
    node : int
        The current node being processed
    leaf_node_counts : dict, optional
        Dictionary mapping leaf node IDs to their respective counts. If not provided,
        it will be computed from examples_df if available, otherwise defaults to 1 per leaf.
    count_attr : str, default='subtree_size'
        Name of the attribute to store the subtree size
    leaf_node_ids_attr : str, default='leaf_node_ids'
        Name of the attribute to store the list of original leaf node IDs
    recount : bool, default=False
        Whether to recompute sizes even if they already exist
    examples_df : pandas.DataFrame, optional
        DataFrame containing original datapoints with cluster assignments
    cluster_col : str, default='cluster'
        Column name in examples_df that contains cluster IDs
    max_depth : int, default=13
        Maximum depth of the tree to be considered during pruning
    Returns:
    --------
    tuple
        (subtree_size, list_of_leaf_node_ids)
    """
    # If this is the first call (at root), prepare the cluster mappings if examples_df is provided
    if examples_df is not None and leaf_node_counts is None:
        # Create the cluster_sizes and cluster_to_datapoints mappings
        leaf_node_counts = examples_df[cluster_col].value_counts().to_dict()
        # We don't need to reset_index if this is the first call
        if 'datapoint_indices' not in G.nodes[node]:
            # Create a mapping from clusters to datapoint indices
            cluster_to_datapoints = examples_df.reset_index().groupby(cluster_col)['index'].apply(list).to_dict()
            
            # Store this mapping in leaf nodes
            for n in G.nodes():
                if G.out_degree(n) == 0:  # Leaf node
                    orig_leaf_id = G.nodes[n].get('orig_leaf_node_id')
                    if orig_leaf_id is not None and orig_leaf_id in cluster_to_datapoints:
                        G.nodes[n]['datapoint_indices'] = cluster_to_datapoints[orig_leaf_id]
    
    # Set default leaf node counts if not provided
    if leaf_node_counts is None:
        leaf_node_counts = {k: 1 for k in range(len([n for n in G.nodes() if G.out_degree(n) == 0]))}
    
    if (recount) or (count_attr not in G.nodes[node]):
        # leaf node
        if G.out_degree(node) == 0:
            orig_leaf_node_id = G.nodes[node].get('orig_leaf_node_id')
            if orig_leaf_node_id is not None:
                # If we have leaf_node_counts, use it for the size
                size = leaf_node_counts.get(int(orig_leaf_node_id), 1)
            else:
                size = 1  # Default size if no mapping is available
                
            G.nodes[node][count_attr] = size
            return size, [orig_leaf_node_id]
    
        # For inner nodes, sum the sizes of all children
        size = 0
        leaf_node_ids = []
        datapoint_indices = []
        
        for child in G.successors(node):
            child_size, child_leaf_node_ids = compute_subtree_sizes_and_membership(
                G, child, leaf_node_counts, count_attr, leaf_node_ids_attr, recount, examples_df, cluster_col
            )
            size += child_size
            leaf_node_ids.extend(child_leaf_node_ids)
            
            # Collect datapoint indices from children if available
            if 'datapoint_indices' in G.nodes[child]:
                datapoint_indices.extend(G.nodes[child]['datapoint_indices'])
        
        G.nodes[node][count_attr] = size
        G.nodes[node][leaf_node_ids_attr] = list(set(leaf_node_ids))
        
        # Store unique datapoint indices if any were collected
        if datapoint_indices:
            G.nodes[node]['datapoint_indices'] = list(set(datapoint_indices))
            
        # Add size attribute for sampling during labeling
        if 'size' not in G.nodes[node]:
            G.nodes[node]['size'] = len(G.nodes[node][leaf_node_ids_attr])
            
    return G.nodes[node][count_attr], G.nodes[node][leaf_node_ids_attr]


# Prune the tree to include nodes with more than 5 descendants in their entire subtree
def prune_tree_by_subtree_size(G, min_descendants=5, max_depth=8):
    """
    Prune a tree graph (NetworkX DiGraph) so that only nodes whose entire subtree
    (all descendants) has more than `min_descendants` nodes are kept.

    Parameters:
    -----------
    G : networkx.DiGraph
        The input tree graph.
    min_descendants : int
        Minimum number of descendants required for a node to be kept.
    max_depth : int
        Maximum depth of the tree to be considered during pruning.

    Returns:
    --------
    pruned_G : networkx.DiGraph
        A new graph that contains only nodes with more than `min_descendants` descendants.
    """    
    # Identify nodes where the count of descendants is greater than the threshold
    root_node = [n for n,d in G.in_degree() if d==0][0]
    keep_nodes = [node for node in G.nodes() 
                  if len(nx.descendants(G, node)) >= min_descendants
                  and (nx.shortest_path_length(G, root_node, node) <= max_depth)
                  # and len(list(G.successors(node))) > 1
                 ]
    
    # Create a subgraph with only the nodes that meet the criteria
    pruned_G = G.subgraph(keep_nodes).copy()
    return pruned_G


def save_hierarchical_tree(G, output_path, format='gml'):
    """
    Save the hierarchical tree to a file.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The hierarchical tree
    output_path : str
        Path to save the tree
    format : str, default='gml'
        Format to save the tree (gml, graphml, etc.)
    """
    if format == 'gml':
        nx.write_gml(G, output_path)
    elif format == 'graphml':
        nx.write_graphml(G, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
      

def load_hierarchical_tree(input_path):
    """
    Load a hierarchical tree from a file.
    """
    if input_path.endswith('.gml'):
        return nx.read_gml(input_path)
    elif input_path.endswith('.graphml'):
        return nx.read_graphml(input_path)
    else:
        raise ValueError(f"Unsupported format: {input_path}")


def label_hierarchical_tree(
        G, 
        leaf_node_counts=None,
        num_samples_to_label=10,
        examples_dataframe=None,
        examples_metadata=None,
        use_discretization=False,
        text_embeddings=None,
        combined_cluster_texts=None,
        discretization_goal="determine the common theme or purpose",
        use_labels_and_descriptions=False,
        use_descriptions_and_examples=True,
        use_child_nodes=True,
        output_labels_and_descriptions=False,
        num_iterations_to_run_labeling=2,
        labeling_model='gpt-4o-mini',
        checking_model='gpt-4o-mini',
        # discretization parameters
        batch_size=30,
        num_summary_candidates_to_generate=10,
        num_datapoints_to_use_for_scoring=300,
        num_leaf_samples_to_use_for_generation=30
):
    """
    Labels a hierarchical tree based on specified criteria.

    Parameters:
    -----------
    G : networkx.DiGraph
        The hierarchical tree graph.
    leaf_node_counts : dict, optional
        Dictionary mapping node IDs to their respective counts. Defaults to None, in which case
        each node is assumed to have a count of 1.
    min_descendants : int, optional
        Minimum number of descendants required for a node to be retained in the pruned tree. Defaults to 1.
    max_depth : int, optional
        Maximum depth of the tree to be considered during pruning. Defaults to 13.
    num_samples_to_label : int, optional
        Number of samples to use for labeling each inner node. Defaults to 10.
    examples_dataframe : pd.DataFrame, optional
        Full dataframe containing examples for each cluster. Should have 'cluster' column.
    examples_metadata : dictionary, optional
        Dictionary containing metadata for the examples dataframe. Should have 'num_examples_per_node', 'cluster_col', 'sentence_col' keys.
    use_discretization : bool, optional
        Whether to use embedding discretization for labeling. Defaults to False.
    text_embeddings : numpy.ndarray, optional
        Embeddings for the texts in combined_texts. Required if use_discretization is True.
    combined_cluster_texts : list, optional
        List of texts (typically label+description pairs) for the cluster centers.
    discretization_goal : str, optional
        The goal or purpose for the discretization. Defaults to "determine the common theme or purpose".
    use_labels_and_descriptions : bool, optional
        Whether to use labels and descriptions for summary generation. Defaults to False.
    use_descriptions_and_examples : bool, optional
        Whether to use descriptions and examples for summary generation. Defaults to True.
    use_child_nodes : bool, optional
        Whether to use child nodes for summary generation. Defaults to True.
    output_labels_and_descriptions : bool, optional
        Whether to output labels and descriptions for each node. Defaults to False.
    num_iterations_to_run_labeling : int, optional
        Number of iterations to run labeling for discretization. The `use_child_nodes` variant of discretization uses the rest of the tree to generate summaries, we might see improvements as the tree improves. Defaults to 2.
    labeling_model : str, optional
        The model to use for labeling. Defaults to 'gpt-4o-mini'.
    checking_model : str, optional
        The model to use for checking the labeling (only relevant if use_discretization is True). Defaults to 'gpt-4o-mini'.

    Returns:    
    --------
    labeled_G : networkx.DiGraph
        The tree graph with labels on all nodes.
    pruned_G : networkx.DiGraph
        The pruned tree graph containing only nodes that meet the descendant criteria.
    inner_node_label_dict : dict
        Dictionary mapping inner node IDs to their labels.
    """
    # Validate parameters for discretization
    if use_discretization:
        if text_embeddings is None or examples_dataframe is None:
            raise ValueError("text_embeddings and examples_dataframe must be provided when use_discretization is True")
            
    nodes = list(reversed(list(nx.topological_sort(G))))
    node_label_dict = {}
    
    # Prepare examples map if not provided
    if examples_dataframe is not None and examples_metadata is not None and examples_metadata.get('cluster_col') in examples_dataframe.columns:
        examples_dataframe = examples_dataframe.set_index(examples_metadata['cluster_col'])
    else:
        # If examples_dataframe is None, or cluster_col is missing, we can't use examples for labeling.
        # Set examples_dataframe to None to ensure downstream code handles this.
        # We might also want to disable features that require examples here if they are critical.
        examples_dataframe = None # Ensure it's None if conditions aren't met
        print("Warning: Examples dataframe is not valid or cluster column is missing. Examples will not be used for labeling.")

    num_iterations_to_run_labeling = 1 if (not use_discretization and not use_child_nodes) else num_iterations_to_run_labeling
    for i in range(num_iterations_to_run_labeling):
        for p in tqdm(nodes, desc=f"Labeling nodes, iteration {i + 1} of {num_iterations_to_run_labeling}..."):
            # Skip leaves or already-labeled nodes to avoid sampling from empty descendants
            if G.nodes[p].get('level') == 'leaf' or 'label' in G.nodes[p]:
                continue
            if use_discretization and 'datapoint_indices' in G.nodes[p]:
                datapoint_indices = G.nodes[p]['datapoint_indices']
                if datapoint_indices and len(datapoint_indices) > 0:
                    print(f"Node {p} has {len(datapoint_indices)} direct datapoint indices")
                    
                    node_embeddings = text_embeddings[datapoint_indices]
                    phi_tilde = node_embeddings.mean(axis=0)
                    
                    # Use this embedding for discretization
                    top_candidates = discretize_embedding_summaries(
                        tree=G,
                        node=p,
                        phi_tilde=phi_tilde,
                        examples_df=examples_dataframe,
                        description_embeddings=text_embeddings,
                        M=1,  # Just get the top one
                        num_summary_candidates_to_generate=num_summary_candidates_to_generate,
                        num_leaf_samples_to_use_for_generation=num_leaf_samples_to_use_for_generation,
                        num_datapoints_to_use_for_scoring=num_datapoints_to_use_for_scoring,
                        # experimental variations
                        use_labels_and_descriptions=use_labels_and_descriptions,
                        use_descriptions_and_examples=use_descriptions_and_examples,
                        use_child_nodes=use_child_nodes,
                        output_labels_and_descriptions=output_labels_and_descriptions,
                        goal=discretization_goal,
                        verbose=True,
                        checking_model=checking_model,
                        labeling_model=labeling_model
                    )
                    
                    if top_candidates and len(top_candidates) > 0:
                        # Parse the top candidate
                        top_candidate = top_candidates[0]
                        if isinstance(top_candidate, dict) and 'label' in top_candidate and 'description' in top_candidate:
                            definition_response = TreeNodeLabelResponse(
                                label=top_candidate['label'],
                                description=top_candidate['description']
                            )
                        elif isinstance(top_candidate, str) and ':' in top_candidate:
                            # If it's a string in format "label: description"
                            parts = top_candidate.split(':', 1)
                            definition_response = TreeNodeLabelResponse(
                                label=parts[0].strip('"\'').strip(),
                                description=parts[1].strip()
                            )
                        else:
                            # Just use the whole thing as the label
                            definition_response = TreeNodeLabelResponse(
                                label=str(top_candidate),
                                description=None
                            )
                    else:
                        print(f"Warning: No top candidates found for node {p}")
                        definition_response = TreeNodeLabelResponse(label=f"Node {p} - No Candidates", description="Label generated without candidates.")

            else:
                # Check if examples_dataframe is available before calling a function that requires it
                if examples_dataframe is not None and examples_metadata is not None:
                    definition_response = single_pass_summarize_labels_with_examples(
                        node=p,
                        example_df=examples_dataframe,
                        tree=G,
                        num_samples_to_label=num_samples_to_label,
                        num_examples_per_node=examples_metadata['num_examples_per_node'],
                        sentence_col=examples_metadata['sentence_col']
                    )
                else:
                    print(f"Warning: Examples not available for node {p}. Using basic label summarization.")
                    definition_response = TreeNodeLabelResponse(label=f"Node {p} - No Examples", description="Label generated without examples.")

            # Set attributes on the nodes - store both label and description
            nx.set_node_attributes(G, {p: definition_response.label}, 'label')
            if definition_response.description is not None:
                nx.set_node_attributes(G, {p: definition_response.description}, 'description')
            
            # Store in the dictionary for return
            node_label_dict[p] = {
                'label': definition_response.label,
                'description': definition_response.description
            }
    
    return G, node_label_dict


def plot_graph(
        G, 
        with_labels=True, 
        with_node_id=False,
        with_sizes=False, 
        label_name='label',
        size_name='subtree_size',
        node_color='lightblue', 
        node_max_size=50,
        node_min_size=5,
        node_size_log_base=2,
        font_size=12,
        adjust_text_on_labels=True,
        adjust_text_force_static=(0.1, 0.5)
):
    """
    Visualize a graph with custom formatting options.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The graph to visualize
    with_labels : bool, default=True
        Whether to display labels on nodes
    with_node_id : bool, default=False
        Whether to include node IDs in labels
    with_sizes : bool, default=False
        Whether to display and scale nodes by size
    label_name : str, default='label'
        The node attribute to use for labels
    size_name : str, default='subtree_size'
        The node attribute to use for size calculation
    node_color : str, default='lightblue'
        Color for nodes
    node_max_size : int, default=50
        Maximum node size
    node_min_size : int, default=5
        Minimum node size
    node_size_log_base : int, default=2
        Base for logarithmic scaling of node sizes
    font_size : int, default=12
        Font size for labels
    """
    labels = nx.get_node_attributes(G, label_name)
    if with_node_id:
        labels = {k: f"[{k}] {labels[k]}" for k in labels}

    # Visualize the pruned tree using a Graphviz layout and custom labels
    pos = graphviz_layout(G, prog="dot")
    
    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    if with_sizes:
        sizes = nx.get_node_attributes(G, size_name)
        log_sizes = {k: np.log(sizes[k]) / np.log(node_size_log_base) for k in sizes}
        normed_sizes = {k: (log_sizes[k] - np.log(node_min_size)) / (np.log(node_max_size) - np.log(node_min_size)) * node_max_size for k in sizes}
        node_sizes = [normed_sizes[node] for node in G.nodes()]
        labels = {k: f"{labels[k]} ({int(sizes[k])})" for k in labels}
    else:
        node_sizes = node_max_size

    nx.draw(
        G, 
        pos, 
        labels=labels,
        with_labels=with_labels,
        node_size=node_sizes,
        node_color=node_color,
        arrows=True, 
        ax=ax,
        font_size=font_size
    )
    texts = list(ax.texts)
    if adjust_text_on_labels:
        adjust_text(texts, force_static=adjust_text_force_static)
    plt.show()


def get_leaf_to_datapoints(G, initial_kmeans_df):
    """
    Get a dict mapping leaf nodes to their corresponding datapoint IDs. These ids are the row values of the dataset and the leaf nodes are 
    the original kmeans 1024 clusters.
    """
    # get a mapping from the original node ids to the tree node ids
    tree_node_to_orig_leaf_node_id = nx.get_node_attributes(G, 'orig_leaf_node_id')
    orig_leaf_node_id_to_tree_node = {v:k for k,v in tree_node_to_orig_leaf_node_id.items()}

    # get a mapping from the leaf nodes to the datapoint ids
    leaf_to_datapoints = initial_kmeans_df['cluster'].map(orig_leaf_node_id_to_tree_node).reset_index().groupby('cluster')['index'].aggregate(list)
    leaf_to_datapoints = leaf_to_datapoints.to_dict()
    return leaf_to_datapoints, orig_leaf_node_id_to_tree_node, tree_node_to_orig_leaf_node_id


def assign_leaves_to_cut(G, cut, leaf_to_datapoints=None, initial_kmeans_df=None):
    """
    Given leaf → [datapoint_ids], assign each datapoint to the unique
    ancestor in `cut`. Returns a dict cluster_node → list of datapoints.
    """
    if leaf_to_datapoints is None:
        leaf_to_datapoints, _, _ = get_leaf_to_datapoints(G, initial_kmeans_df)

    assignment = defaultdict(list)
    for leaf, pts in leaf_to_datapoints.items():
        v = leaf
        # walk up until we hit a cut node
        while v not in cut:
            v = list(G.predecessors(v))[0]
        assignment[v].extend(pts)
    return dict(assignment)


def assign_by_threshold(G, root=None, leaf_to_datapoints=None, initial_kmeans_df=None, leaf_counts=None, K=10):
    """
    For each datapoint at a leaf, ascend the tree until its node's subtree_size 
    is >= total_count / K, then assign it there.
    After initial assignment, merge smallest clusters up to their parent until exactly K clusters remain.
    
    Returns: dict mapping cluster node -> list of datapoint IDs.
    """
    # 1) Determine root if not provided
    if root is None:
        root = get_root(G)
    
    # 2) Build leaf_to_datapoints if not provided
    if leaf_to_datapoints is None:
        leaf_to_datapoints, orig_leaf_node_id_to_tree_node, tree_node_to_orig_leaf_node_id = get_leaf_to_datapoints(G, initial_kmeans_df)
    
    # 3) Compute subtree sizes if missing
    if 'subtree_size' not in G.nodes[root]:
        if leaf_counts is None:
            leaf_counts = {leaf: len(pts) for leaf, pts in leaf_to_datapoints.items()}
        compute_subtree_sizes_and_membership(G, root, leaf_counts)
    
    # 4) Determine threshold
    total = G.nodes[root]['subtree_size']
    threshold = total / K
    
    # 5) Initial assignment based on threshold
    assignment = defaultdict(list)
    
    for leaf, pts in leaf_to_datapoints.items():
        node = leaf
        while node != root and G.nodes[node]['subtree_size'] < threshold:
            node = list(G.predecessors(node))[0]
        assignment[node].extend(pts)

    # 6) Post-merge to enforce <= K clusters
    clusters = dict(assignment)  # copy to mutable dict
    while len(clusters) > K:
        # find the cluster with smallest size
        smallest = min(clusters, key=lambda n: len(clusters[n]))
        pts = clusters.pop(smallest)
        # find its parent (or default to root)
        parent = list(G.predecessors(smallest))[0]
        clusters.setdefault(parent, []).extend(pts)
    
    return clusters
