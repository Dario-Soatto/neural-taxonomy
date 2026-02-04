#!/usr/bin/env python3
# agglomerative_clustering.py - Focused on hierarchical clustering only

import sys
import pandas as pd
import os
import warnings
import numpy as np
from tqdm.auto import tqdm
import argparse
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from pathlib import Path
from utils_trees import (
    save_hierarchical_tree, 
    plot_graph, 
    label_hierarchical_tree, 
    prune_tree_by_subtree_size, 
    get_root,
    compute_subtree_sizes_and_membership
)
from itertools import product

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.simplefilter(action='ignore')


def get_linkage_matrix_from_model(model, n_samples):
    """
    Convert an AgglomerativeClustering model to a linkage matrix compatible with scipy functions.
    
    Parameters:
    -----------
    model : AgglomerativeClustering
        Fitted AgglomerativeClustering model
    n_samples : int
        Number of original samples
        
    Returns:
    --------
    numpy.ndarray
        Linkage matrix in scipy format
    """
    # Number of original samples (i.e., leaves in the tree)
    counts = np.zeros(model.children_.shape[0])
    
    # For each merge, record the size of the new cluster
    for i, merge in enumerate(model.children_):
        child_cluster_sizes = np.zeros(2)
        
        for j, child in enumerate(merge):
            # If the child is a leaf node
            if child < n_samples:
                child_cluster_sizes[j] = 1
            else:
                # If the child is a cluster, get its size from the counts array
                child_cluster_sizes[j] = counts[child - n_samples]
                
        # Record the size of the merged cluster
        counts[i] = child_cluster_sizes.sum()
    
    # Create linkage matrix in the scipy format
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    
    return linkage_matrix


def optimal_hierarchical_cuts_dp(embeddings, min_clusters=2, max_clusters=None, 
                               min_cluster_size=2, method='ward', metric='euclidean',
                               n_samples=None, verbose=True):
    """
    Optimized version using dynamic programming approach.
    """
    if n_samples is None:
        n_samples = embeddings.shape[0]
    
    if max_clusters is None:
        max_clusters = n_samples // 2
    
    # Step 1: Compute linkage matrix using AgglomerativeClustering
    model = AgglomerativeClustering(
        n_clusters=None,  # Don't predefine number of clusters
        distance_threshold=0,  # Cut tree at zero distance -> n_clusters=n_samples
        linkage=method,
        metric=metric,
        compute_distances=True  # Ensure distances are computed
    )
    model.fit(embeddings)
    
    # Convert to scipy linkage matrix
    linkage_matrix = get_linkage_matrix_from_model(model, n_samples)
    
    # The rest of the function remains the same
    # Step 2: Precompute all possible clusterings - O(n²)
    # Map: number of clusters -> (threshold, labels)
    cluster_map = {}
    
    # Sort distances from smallest to largest for this preprocessing step
    distances = sorted(linkage_matrix[:, 2])
    
    if verbose:
        print("Precomputing clusterings...")
        
    prev_n_clusters = -1
    for dist in distances:
        labels = fcluster(linkage_matrix, dist, criterion='distance')
        n_clusters = len(np.unique(labels))
        
        # Only store if this creates a new number of clusters and is within our range
        if (n_clusters != prev_n_clusters and 
            min_clusters <= n_clusters <= max_clusters):
            
            # Check minimum cluster size
            valid = True
            for label in np.unique(labels):
                if np.sum(labels == label) < min_cluster_size:
                    valid = False
                    break
                    
            if valid:
                cluster_map[n_clusters] = (dist, labels)
                prev_n_clusters = n_clusters
                
                if verbose and n_clusters % 5 == 0:
                    print(f"Found clustering with {n_clusters} clusters at distance {dist:.4f}")
    
    # Step 3: Compute silhouette scores only for the unique clusterings - O(n² * k)
    # where k is the number of unique clusterings (much less than n)
    scores = {}
    for n_clusters, (dist, labels) in cluster_map.items():
        scores[n_clusters] = silhouette_score(embeddings, labels)
        
        if verbose:
            print(f"Clusters: {n_clusters}, Distance: {dist:.4f}, Silhouette: {scores[n_clusters]:.4f}")
        
    # Initialize DP table
    # dp[i] = (max silhouette score achievable with i clusters, prev best cluster count)
    dp = {i: (-1, None) for i in range(min_clusters, max_clusters + 1)}
    
    # Base case: fill in scores we have
    for n_clusters in scores:
        dp[n_clusters] = (scores[n_clusters], None)
    
    # Fill the DP table
    cluster_counts = sorted(scores.keys())
    
    # Reconstruct optimal sequence
    optimal_cluster_counts = []
    current = max(cluster_counts, key=lambda x: scores[x])
    
    while current is not None:
        optimal_cluster_counts.append(current)
        # Find next larger cluster count with best score
        next_clusters = [c for c in cluster_counts if c > current]
        if not next_clusters:
            break
        current = max(next_clusters, key=lambda x: scores[x])
    
    # Reverse to get from coarse to fine
    optimal_cluster_counts.reverse()
    
    # Get corresponding thresholds
    optimal_thresholds = [cluster_map[c][0] for c in optimal_cluster_counts]
    optimal_scores = [scores[c] for c in optimal_cluster_counts]
        
    return {
        'optimal_thresholds': optimal_thresholds,
        'silhouette_scores': optimal_scores,
        'n_clusters': optimal_cluster_counts,
        'linkage_matrix': linkage_matrix,
        'model': model
    }


def optimal_hierarchical_cuts(
    embeddings, 
    min_clusters=2,
    max_clusters=None, 
    min_cluster_size=2, 
    method='ward', 
    metric='euclidean',
    n_samples=None, 
    verbose=True,
    min_hierarchical_levels=1  # New parameter for minimum depth
):
    """
    Find optimal hierarchical clustering cut distances using silhouette analysis.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        The data matrix, shape (n_samples, n_features)
    min_clusters : int, default=2
        Minimum number of clusters to consider
    max_clusters : int or None, default=None
        Maximum number of clusters to consider. If None, uses n_samples/2
    min_cluster_size : int, default=2
        Minimum size a cluster must have to be considered valid
    method : str, default='ward'
        Linkage method for hierarchical clustering
    metric : str, default='euclidean'
        Distance metric for clustering
    n_samples : int or None, default=None
        Number of samples in the data. If None, uses embeddings.shape[0]
    verbose : bool, default=True
        Whether to print progress and plot results
    min_hierarchical_levels : int, default=1
        Minimum number of hierarchical levels to attempt to find.
        
    Returns:
    --------
    dict
        Contains:
        - 'optimal_thresholds': list of optimal distance thresholds
        - 'silhouette_scores': silhouette scores at each level
        - 'n_clusters': number of clusters at each level
        - 'linkage_matrix': the hierarchical clustering linkage matrix
        - 'model': the fitted AgglomerativeClustering model
    """
    if n_samples is None:
        n_samples = embeddings.shape[0]
    
    if max_clusters is None:
        max_clusters = n_samples // 2
    
    model = AgglomerativeClustering(
        n_clusters=None,       # Don't predefine number of clusters
        distance_threshold=0,  # Cut tree at zero distance -> n_clusters=n_samples
        linkage=method, 
        metric=metric,
        compute_distances=True  # Ensure distances are computed
    )
    model.fit(embeddings)
    
    # Convert to scipy linkage matrix format
    linkage_matrix = get_linkage_matrix_from_model(model, n_samples)
    
    # Extract all possible threshold distances from the linkage matrix
    distances = sorted(linkage_matrix[:, 2], reverse=True)
    
    # Initialize variables to track results
    all_scores = []
    all_distances = []
    all_n_clusters = []
    
    if verbose:
        print("Evaluating silhouette scores across distances...")
    
    # Evaluate silhouette score for each possible threshold
    # Store unique (distance, n_clusters) to avoid issues if multiple distances yield same n_clusters
    evaluated_configs = {} 

    for i, dist in enumerate(distances):
        labels = fcluster(linkage_matrix, dist, criterion='distance')
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Skip if we have too few or too many clusters
        if n_clusters < min_clusters or n_clusters > max_clusters:
            continue
        
        # Check if all clusters meet minimum size requirement
        valid_clustering = True
        for label in unique_labels:
            if np.sum(labels == label) < min_cluster_size:
                valid_clustering = False
                break
                
        if not valid_clustering:
            continue
            
        # Calculate silhouette score
        try:
            # Use (n_clusters, dist) as key to prioritize by n_clusters then by dist if scores are same
            # This is to ensure we only store one entry per n_cluster if multiple distances give it
            # We prefer the largest distance (coarsest cut) that gives this n_cluster.
            # Since 'distances' is sorted reverse (large to small), the first one we see is good.
            config_key = n_clusters 
            
            if config_key not in evaluated_configs:
                score = silhouette_score(embeddings, labels)
                evaluated_configs[config_key] = {'dist': dist, 'score': score, 'n_clusters': n_clusters}
                
                if verbose and i % 10 == 0: # Keep some progress print
                    print(f"Distance {dist:.4f}, Clusters: {n_clusters}, Silhouette: {score:.4f} (Candidate)")
        except:
            # This can happen if we have only one cluster or each point is its own cluster
            continue

    if not evaluated_configs:
        raise ValueError("No valid clusterings found. Try adjusting parameters.")

    # Convert evaluated_configs dict to lists for processing
    # Sort by n_clusters to make it easier to find next levels consistently
    sorted_configs = sorted(evaluated_configs.values(), key=lambda x: x['n_clusters'])

    all_scores = np.array([cfg['score'] for cfg in sorted_configs])
    all_distances = np.array([cfg['dist'] for cfg in sorted_configs])
    all_n_clusters = np.array([cfg['n_clusters'] for cfg in sorted_configs])

    # Data structure to manage all available cuts and their usage
    all_cuts_info = []
    for i in range(len(all_scores)):
        all_cuts_info.append({
            'id': i, # Index in all_scores/all_distances/all_n_clusters
            'dist': all_distances[i],
            'n_clusters': all_n_clusters[i],
            'score': all_scores[i],
            'used': False
        })
        
    optimal_thresholds_selected_distances = []
    current_n_clusters_for_selection = -1 # Ensures the first selection considers all valid n_clusters

    # Primary selection loop: Find a hierarchically optimal sequence
    if verbose:
        print("Selecting primary optimal thresholds...")
    while True:
        potential_next_cuts = [
            cut for cut in all_cuts_info
            if not cut['used'] and cut['n_clusters'] > current_n_clusters_for_selection
        ]

        if not potential_next_cuts:
            break 

        best_next_cut = max(potential_next_cuts, key=lambda cut: cut['score'])
        
        optimal_thresholds_selected_distances.append(best_next_cut['dist'])
        current_n_clusters_for_selection = best_next_cut['n_clusters']
        best_next_cut['used'] = True
        if verbose:
            print(f"  Selected primary level: Dist={best_next_cut['dist']:.4f}, Clusters={best_next_cut['n_clusters']}, Score={best_next_cut['score']:.4f}")

    # Extension loop: If primary sequence is shorter than min_hierarchical_levels
    num_primary_levels = len(optimal_thresholds_selected_distances)
    if min_hierarchical_levels is not None and num_primary_levels < min_hierarchical_levels:
        if verbose:
            print(f"\nPrimary selection found {num_primary_levels} levels. Trying to extend to {min_hierarchical_levels} levels...")
        
        num_to_add = min_hierarchical_levels - num_primary_levels
        
        # Get current n_clusters for all selected levels to help find gaps
        current_selected_n_clusters = []
        for dist in optimal_thresholds_selected_distances:
            for cut in all_cuts_info:
                if cut['dist'] == dist:
                    current_selected_n_clusters.append(cut['n_clusters'])
                    break
        current_selected_n_clusters.sort()  # Sort to identify gaps
        
        for i in range(num_to_add):
            # Find all unused cuts
            potential_extension_cuts = [cut for cut in all_cuts_info if not cut['used']]
            
            if not potential_extension_cuts:
                if verbose:
                    print(f"  No more unused cuts available. Stopped at {len(optimal_thresholds_selected_distances)} levels.")
                break
            
            # Strategy: Find the best cut that fills a gap in the hierarchy
            # or extends it at either end
            best_extension_cut = None
            best_extension_score = -1
            best_gap_size = 0
            
            for cut in potential_extension_cuts:
                # Calculate gap score - how well this cut fills a gap
                gap_score = 0
                n_clusters = cut['n_clusters']
                
                # Find where this would fit in the current hierarchy
                insert_position = None
                for j, existing_n in enumerate(current_selected_n_clusters):
                    if n_clusters < existing_n:
                        insert_position = j
                        break
                
                if insert_position is None:
                    # Would go at the end (finest level)
                    insert_position = len(current_selected_n_clusters)
                
                # Calculate gap size
                if insert_position == 0:
                    # Coarsest position
                    if len(current_selected_n_clusters) > 0:
                        gap_size = current_selected_n_clusters[0] - n_clusters
                    else:
                        gap_size = float('inf')  # First cut, infinite gap
                elif insert_position == len(current_selected_n_clusters):
                    # Finest position
                    if len(current_selected_n_clusters) > 0:
                        gap_size = n_clusters - current_selected_n_clusters[-1]
                    else:
                        gap_size = float('inf')  # First cut, infinite gap
                else:
                    # Middle position - between two existing levels
                    gap_size = current_selected_n_clusters[insert_position] - current_selected_n_clusters[insert_position - 1]
                
                # Prefer cuts that fill larger gaps and have good silhouette scores
                # Normalize gap_size to be comparable with silhouette score (0-1 range)
                if gap_size == float('inf'):
                    normalized_gap = 1.0
                else:
                    # Use a sigmoid-like function to normalize gap sizes
                    normalized_gap = min(1.0, gap_size / 10.0)  # Assume gaps > 10 are very large
                
                # Combined score: weighted average of silhouette and gap filling
                combined_score = 0.7 * cut['score'] + 0.3 * normalized_gap
                
                if combined_score > best_extension_score:
                    best_extension_score = combined_score
                    best_extension_cut = cut
                    best_gap_size = gap_size
            
            if best_extension_cut:
                optimal_thresholds_selected_distances.append(best_extension_cut['dist'])
                best_extension_cut['used'] = True
                
                # Update current_selected_n_clusters to maintain sorted order
                current_selected_n_clusters.append(best_extension_cut['n_clusters'])
                current_selected_n_clusters.sort()
                
                if verbose:
                    print(f"  Extended level {i+1}: Dist={best_extension_cut['dist']:.4f}, "
                          f"Clusters={best_extension_cut['n_clusters']}, Score={best_extension_cut['score']:.4f}, "
                          f"Gap filled: {best_gap_size}")

    if not optimal_thresholds_selected_distances:
         raise ValueError("No optimal thresholds could be selected. Check input data and parameters.")

    # Sort the final list of selected thresholds (distances) from largest to smallest (coarse to fine)
    optimal_thresholds = sorted(list(set(optimal_thresholds_selected_distances)), reverse=True)
    
    optimal_scores = []
    optimal_n_clusters = []
    for threshold in optimal_thresholds:
        labels = fcluster(linkage_matrix, threshold, criterion='distance')
        score = silhouette_score(embeddings, labels)
        optimal_scores.append(score)
        optimal_n_clusters.append(len(np.unique(labels)))
    
    if verbose:
        print("\nOptimal distance thresholds:")
        for i, (dist, score, n_clusters) in enumerate(zip(optimal_thresholds, optimal_scores, optimal_n_clusters)):
            print(f"Level {i+1}: Distance={dist:.4f}, Clusters={n_clusters}, Silhouette={score:.4f}")
    
    return {
        'optimal_thresholds': optimal_thresholds,
        'silhouette_scores': optimal_scores,
        'n_clusters': optimal_n_clusters,
        'linkage_matrix': linkage_matrix,
        'model': model  # Include the model for reference
    }


def create_hierarchical_tree(embeddings, linkage_matrix=None, n_samples=None, distance_thresholds=None, model=None, examples_df=None):
    """
    Create a hierarchical tree structure with multiple levels based on optimal distance thresholds.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        The data matrix, shape (n_samples, n_features)
    linkage_matrix : numpy.ndarray, optional
        The linkage matrix from scipy's hierarchical clustering
    n_samples : int, optional
        The number of original data points
    distance_thresholds : list, optional
        A sorted list of distance thresholds (from largest to smallest)
    model : AgglomerativeClustering, optional
        Fitted AgglomerativeClustering model to use for more accurate tree construction
    examples_df : pandas.DataFrame, optional
        DataFrame containing original datapoints with cluster assignments
        
    Returns:
    --------
    networkx.DiGraph
        A directed graph representing the sparse hierarchical structure
    """
    if n_samples is None:
        n_samples = embeddings.shape[0]
    if distance_thresholds is None:
        distance_thresholds = sorted(linkage_matrix[:, 2], reverse=True)[:3]  # Default to top 3 levels
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add original data points as leaf nodes
    for i in range(n_samples):
        G.add_node(i, type='sample', level='leaf', orig_leaf_node_id=i)
    
    # Generate clusters at each threshold
    all_clusters = {}
    for threshold in distance_thresholds:
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
        
        # Group samples by cluster
        clusters = {}
        for i in range(n_samples):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(i)
        
        all_clusters[threshold] = clusters
    
    # Create a unique ID for each cluster
    cluster_id_counter = n_samples
    cluster_node_ids = {}  # Maps (threshold, cluster_id) to node_id
    
    # Build the tree from bottom up (smallest threshold first)
    for threshold in reversed(distance_thresholds):
        clusters = all_clusters[threshold]
        
        for cluster_id, samples in clusters.items():
            # Create a unique node ID for this cluster
            node_id = cluster_id_counter
            cluster_id_counter += 1
            
            # Store the mapping
            cluster_node_ids[(threshold, cluster_id)] = node_id
            
            # Add the cluster node
            G.add_node(node_id, 
                      type='cluster',
                      distance_threshold=threshold,
                      size=len(samples),
                      samples=samples)
            
            # If this is the smallest threshold, connect to samples directly
            if threshold == distance_thresholds[-1]:
                for sample in samples:
                    G.add_edge(node_id, sample)
    
    # Now connect clusters across thresholds
    for i in range(len(distance_thresholds) - 1):
        higher_threshold = distance_thresholds[i]
        lower_threshold = distance_thresholds[i + 1]
        
        higher_clusters = all_clusters[higher_threshold]
        lower_clusters = all_clusters[lower_threshold]
        
        # For each higher threshold cluster, find its children in the lower threshold
        for higher_cluster_id, higher_samples in higher_clusters.items():
            higher_node_id = cluster_node_ids[(higher_threshold, higher_cluster_id)]
            
            # Find all lower clusters that contain samples from this higher cluster
            children = set()
            for lower_cluster_id, lower_samples in lower_clusters.items():
                # Use set intersection to check for overlap more efficiently
                if set(higher_samples).intersection(set(lower_samples)):
                    lower_node_id = cluster_node_ids[(lower_threshold, lower_cluster_id)]
                    children.add(lower_node_id)
            
            # Connect the higher cluster to its children
            for child_id in children:
                G.add_edge(higher_node_id, child_id)
    
    # Verify that the tree is properly connected
    if nx.number_weakly_connected_components(G) > 1:
        print("Warning: Tree has disconnected components. Adding a super-root to connect them.")
        components = list(nx.weakly_connected_components(G))
        super_root_id = cluster_id_counter
        
        # Add a super-root
        G.add_node(super_root_id, type='super_root', size=n_samples)
        
        # Connect each component to the super-root
        for component in components:
            # Find nodes with no incoming edges in this component
            for node in component:
                if G.in_degree(node) == 0 and node < n_samples:
                    G.add_edge(super_root_id, node)
                elif G.in_degree(node) == 0:
                    G.add_edge(super_root_id, node)
    
    # Compute subtree sizes and membership for all nodes
    print("Computing subtree sizes and membership...")
    root_node = get_root(G)
    compute_subtree_sizes_and_membership(G, root_node, examples_df=examples_df)
    
    return G, cluster_node_ids


def run_hierarchical_clustering(
    centroids_file_path,
    output_dir_path,
    min_clusters=2,
    max_clusters=None,
    min_cluster_size=2,
    method='ward',
    metric='euclidean',
    examples_df=None,
    min_hierarchical_levels=1
):
    """
    Perform hierarchical clustering on pre-computed centroids (from KMeans)
    and find optimal clustering levels.
    
    Parameters:
    -----------
    centroids_file_path : str
        Path to the NumPy file containing the centroids from KMeans clustering
    output_dir_path : str
        Directory to save the output files
    min_clusters : int, default=2
        Minimum number of clusters to consider
    max_clusters : int, default=None
        Maximum number of clusters to consider
    min_cluster_size : int, default=2
        Minimum size a cluster must have to be considered valid
    method : str, default='ward'
        Linkage method for hierarchical clustering
    metric : str, default='euclidean'
        Distance metric for clustering
    examples_dataframe_path : str, optional
        Path to the CSV file containing original datapoints with cluster assignments
    min_hierarchical_levels : int, default=1
        Minimum number of hierarchical levels to aim for.
    """
    # Basic input validation
    if not os.path.exists(centroids_file_path):
        raise FileNotFoundError(f"Centroids file not found: {centroids_file_path}")
    
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2")
    
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    
    if method not in ["ward", "complete", "average", "single"]:
        raise ValueError(f"Invalid method: {method}. Must be one of: ward, complete, average, single")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Load centroids file
    print(f"Loading centroids from: {centroids_file_path}")
    try:
        centroids = np.load(centroids_file_path)
    except Exception as e:
        raise ValueError(f"Error loading centroids file: {e}")
    
    # Check for NaN or infinite values in centroids
    if np.isnan(centroids).any() or np.isinf(centroids).any():
        raise ValueError("Centroids contain NaN or infinite values")
    
    # Check for zero variance features
    zero_var_features = np.where(np.var(centroids, axis=0) == 0)[0]
    if len(zero_var_features) > 0:
        print(f"Warning: Found {len(zero_var_features)} features with zero variance. These may affect clustering quality.")
    
    n_centroids = len(centroids)
    print(f"Loaded {n_centroids} centroids with dimension {centroids.shape[1]}")
    
    # Validate max_clusters if provided
    if max_clusters is not None:
        if max_clusters <= min_clusters:
            raise ValueError(f"max_clusters ({max_clusters}) must be greater than min_clusters ({min_clusters})")
        if max_clusters >= n_centroids:
            raise ValueError(f"max_clusters ({max_clusters}) must be less than number of centroids ({n_centroids})")
    
    # Run hierarchical clustering
    print("Performing hierarchical clustering and finding optimal thresholds...")
    result = optimal_hierarchical_cuts(
        centroids,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        min_cluster_size=min_cluster_size,
        method=method,
        metric=metric,
        n_samples=n_centroids,
        verbose=True,
        min_hierarchical_levels=min_hierarchical_levels
    )
    
    # Validate clustering results
    if len(result['optimal_thresholds']) < min_hierarchical_levels:
        print(f"Warning: Found only {len(result['optimal_thresholds'])} hierarchical levels, which is less than the requested minimum of {min_hierarchical_levels}")
    
    # Check for degenerate clusters (clusters with size < min_cluster_size)
    for i, threshold in enumerate(result['optimal_thresholds']):
        cluster_labels = fcluster(result['linkage_matrix'], threshold, criterion='distance')
        cluster_sizes = np.bincount(cluster_labels)
        small_clusters = np.sum(cluster_sizes[1:] < min_cluster_size)
        if small_clusters > 0:
            print(f"Warning: Level {i+1} has {small_clusters} clusters smaller than min_cluster_size={min_cluster_size}")
    
    # Create the hierarchical tree after optimization
    print("Creating hierarchical tree from optimal thresholds...")
    tree, cluster_node_ids = create_hierarchical_tree(
        centroids,
        result['linkage_matrix'],
        n_centroids,
        result['optimal_thresholds'],
        result['model'],
        examples_df
    )
    
    # Add the tree to the result
    result['tree'] = tree
    result['cluster_node_ids'] = cluster_node_ids
    
    # Save results
    output_base = Path(output_dir_path)
    
    # Save thresholds and scores
    thresholds_df = pd.DataFrame({
        'threshold': result['optimal_thresholds'],
        'n_clusters': result['n_clusters'],
        'silhouette_score': result['silhouette_scores']
    })
    thresholds_df.to_csv(output_base / 'optimal_thresholds.csv', index=False)
    print(f"Saved optimal thresholds to: {output_base / 'optimal_thresholds.csv'}")
    
    # Save hierarchical tree
    tree_path = output_base / 'hierarchical_tree.gml'
    save_hierarchical_tree(result['tree'], tree_path)
    print(f"Saved hierarchical tree to: {tree_path}")
    
    # For each threshold, save cluster assignments
    linkage_m = result['linkage_matrix']
    threshold_to_fcluster_to_graph_node_map = result['cluster_node_ids']

    for i, threshold in enumerate(result['optimal_thresholds']):
        n_clusters_at_level = result['n_clusters'][i]
        
        # Get fcluster labels for each original sample (centroid) at this threshold
        fcluster_labels_for_samples = fcluster(linkage_m, threshold, criterion='distance')
        
        # Map each sample's fcluster label to its corresponding graph_node_id
        graph_node_ids_for_samples = []
        for f_label in fcluster_labels_for_samples:
            graph_node = threshold_to_fcluster_to_graph_node_map.get((threshold, f_label))
            if graph_node is None:
                print(f"Warning: Could not map fcluster label {f_label} at threshold {threshold} to a graph node ID. Appending None.")
                graph_node_ids_for_samples.append(None) 
            else:
                graph_node_ids_for_samples.append(int(graph_node))
        
        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'centroid_id': np.arange(n_centroids), # K-Means centroid ID (0-indexed)
            'fcluster_label': fcluster_labels_for_samples, # fcluster output (1-indexed usually)
            'graph_node_id': graph_node_ids_for_samples # Actual tree node ID for the cluster
        })
        cluster_path = output_base / f'clusters_level_{i+1}_{n_clusters_at_level}_clusters.csv'
        cluster_df.to_csv(cluster_path, index=False)
        print(f"Saved level {i+1} clustering ({n_clusters_at_level} clusters) to: {cluster_path}")
        
    print("Hierarchical clustering completed successfully.")
    return result


def custom_examples_loader(
        data_level_labels_path=None, 
        examples_dataframe_path=None,
        initial_labels_embeddings_path=None,
        use_discretization=False,
        experiment=None
):
    """
    Load examples from a custom source. Must output a dataframe with the following columns:
    - sentences: the text of the example
    - label: the label of the example
    - cluster: the cluster of the example
    - description: the description of the example
    """
    def file_reader(path):
        if '.csv' in path:
            return pd.read_csv(path)
        elif '.json' in path:
            return pd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        

    # Load discretization resources if needed
    text_embeddings = None
    if use_discretization:
        if initial_labels_embeddings_path is None:
            raise ValueError("initial_labels_embeddings_path must be provided when use_discretization is True")
        
        # Load the embeddings
        print(f"Loading embeddings from: {initial_labels_embeddings_path}")
        text_embeddings = np.load(initial_labels_embeddings_path)
        if isinstance(text_embeddings, np.lib.npyio.NpzFile):
            text_embeddings = text_embeddings['embeddings']

    if data_level_labels_path is not None:
        data_level_labels_df = file_reader(data_level_labels_path)
    if examples_dataframe_path is not None:
        examples_df = file_reader(examples_dataframe_path)
    if data_level_labels_path is None and examples_dataframe_path is None:
        return None

    if experiment == 'news-discourse':
        examples_df = (
            examples_df
                .rename(columns={'sentence': 'sentences'})
                .assign(doc_id=lambda df: df['doc_id'].str.split('_').str.get(1).astype(int))
                .assign(sentence_idx=lambda df: df.reset_index().groupby('doc_id')['index'].rank(method='dense').astype(int))
         )
        data_level_labels_df = (
            data_level_labels_df
                .assign(doc_id=lambda df: df['custom_id'].str.split('__').str.get(0).astype(int))
                .drop(columns='custom_id')
        )
        return examples_df.drop(columns='label').merge(data_level_labels_df, on=['doc_id', 'sentence_idx']), text_embeddings
    
    elif experiment == 'editorial':
        return examples_df, text_embeddings
    
    elif 'reasoning' in experiment:
        data_level_labels_df = (
            data_level_labels_df
                .loc[lambda df: df['output'].notna()]
                .assign(output_chunks=lambda df: df['output'].str.split(':'))
                .assign(label=lambda df: df['output_chunks'].apply(lambda x: x[0].replace('"', '').strip()))
                .assign(description=lambda df: df['output_chunks'].apply(lambda x: ':'.join(x[1:]).strip()))
                .drop(columns=['output', 'output_chunks'])
                .rename(columns={'index': 'datapoint_index'})
        )

        if examples_dataframe_path is None:
            return data_level_labels_df.assign(sentences=np.nan), text_embeddings
        
        if 'response' in examples_df.columns:
            examples_df = (
                examples_df
                    .explode('response')
                    .reset_index(drop=True).reset_index()
                    .assign(chunk_idx=lambda df:df.groupby('index')['level_0'].rank(method='dense').astype(int)-1)
                    .assign(index=lambda df: df['index'] + '__reasoning-step-' + df['chunk_idx'].astype(str))
                    .rename(columns={'response': 'sentences', 'index': 'datapoint_index'})
                )
            example_cols_to_keep = ['sentences', 'problem', 'datapoint_index']
        else:
            examples_df = examples_df.rename(columns={'rollouts': 'sentences', 'index': 'datapoint_index'})
            example_cols_to_keep = ['sentences', 'datapoint_index']

        output_df = (
            data_level_labels_df
                .merge(examples_df[example_cols_to_keep].drop_duplicates(), on='datapoint_index', how='left')
        )
        if '__' in experiment:
            subsection = experiment.split('__')[1]
            output_df = output_df.loc[lambda df: df['datapoint_index'].str.split('__').str.get(0) == subsection]
            if use_discretization:
                text_embeddings = text_embeddings[output_df.index.tolist()]
            output_df = output_df.reset_index(drop=True)
        return output_df, text_embeddings

    elif experiment == 'emotions':
        data_level_labels_df = (
            data_level_labels_df
                .assign(id=lambda df: df.apply(lambda x: x['custom_id'].split('__')[x['sentence_idx'] - 1], axis=1))
        )
        examples_df = examples_df[['text', 'id']].drop_duplicates().rename(columns={'text': 'sentences'})
        return data_level_labels_df.merge(examples_df, on='id'), text_embeddings
    
    elif experiment == 'hate-speech':
        examples_df = (
            examples_df.assign(key= lambda df: df['Hate_ID'] + '_' + df['Reply_ID'])
                .rename(columns={'Reply_body': 'sentences'})
        )
        data_level_labels_df = (
            data_level_labels_df
                .assign(key=lambda df: df.apply(lambda x: x['custom_id'].split('__')[x['sentence_idx'] - 1], axis=1))
        )
        return data_level_labels_df.merge(examples_df.drop_duplicates(subset='key'), on='key'), text_embeddings
    else:
        raise ValueError(f"Experiment {experiment} not supported")



def label_and_visualize_tree(
    tree,
    initial_cluster_labels_path,
    output_dir_path,
    leaf_node_counts=None,
    min_descendants=2,
    max_depth=8,
    num_samples_to_label=10,
    visualize=True,
    examples_metadata=None,
    use_discretization=False,
    text_embeddings=None,
    discretization_goal="determine the common theme or purpose",
    # experimental variations
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
    Label and visualize a hierarchical tree.
    
    Parameters:
    -----------
    tree : networkx.DiGraph
        The hierarchical tree to label and visualize
    initial_cluster_labels_path : str
        Path to the CSV file containing initial cluster labels
    output_dir_path : str
        Directory to save the labeled tree and visualization
    leaf_node_counts : dict, optional
        Dictionary mapping node IDs to their respective counts
    min_descendants : int, default=2
        Minimum number of descendants required for a node to be retained
    max_depth : int, default=8
        Maximum depth of the tree to consider during pruning
    num_samples_to_label : int, default=10
        Number of samples to use for labeling each inner node
    visualize : bool, default=True
        Whether to generate and save a visualization of the labeled tree
    examples_metadata : dictionary
        contains keys: examples_df, num_examples_per_node, cluster_col, sentence_col
        * examples_df: the examples dataframe
        * num_examples_per_node: Number of example datapoints to use for each node (in addition to the labels, to keep it more specific.)
        * cluster_col: column name of the cluster column in the examples dataframe
        * sentence_col: column name of the sentence column in the examples dataframe
    use_discretization : bool, default=False
        Whether to use discretization for labeling
    discretization_goal : str, default="determine the common theme or purpose"
        The goal or purpose for discretization
    initial_labels_embeddings_path : str, optional
        Path to the initial labels embeddings file (required if use_discretization is True)
        
    Returns:
    --------
    tuple
        (labeled_tree, pruned_tree, inner_node_label_dict)
    """
    # Save labeled tree
    output_base = Path(output_dir_path)
    labeled_tree_path = output_base / 'labeled_hierarchical_tree.gml'
    pruned_tree_path = output_base / 'pruned_hierarchical_tree.gml'
    inner_node_labels_path = output_base / 'inner_node_labels.csv'
    
    # Load initial cluster labels
    print(f"Loading initial cluster labels from: {initial_cluster_labels_path}")
    initial_labels_df = pd.read_csv(initial_cluster_labels_path)
    # Ensure the labels dataframe has node_id and label columns
    if 'node_id' not in initial_labels_df.columns or 'label' not in initial_labels_df.columns:
        raise ValueError("Initial labels dataframe must have 'node_id' and 'label' columns")
    
    # Rename and process the dataframe
    initial_labels_df = initial_labels_df.rename(columns={'node_id': 'orig_leaf_node_id'})
    
    # Add labels to leaf nodes in the tree
    for _, row in initial_labels_df.iterrows():
        node_id = int(row['orig_leaf_node_id'])
        for n in tree.nodes():
            if tree.nodes[n].get('orig_leaf_node_id') == node_id:
                nx.set_node_attributes(tree, {n: row['label']}, 'label')
                break
    
    
    # Create combined texts from initial labels
    combined_cluster_texts = initial_labels_df.apply(
        lambda x: f"\"{x['label']}\": {x.get('description', '')}", axis=1
    ).tolist()
    
    print(f"Loaded {len(combined_cluster_texts)} texts for discretization")

    
    # Prune the tree based on subtree sizes (which were already computed in create_hierarchical_tree)
    print(f"Pruning tree with min_descendants={min_descendants}, max_depth={max_depth}...")
    pruned_tree = prune_tree_by_subtree_size(tree, min_descendants=min_descendants, max_depth=max_depth)
    
    # Label the pruned tree
    print("Labeling the pruned tree...")
    labeled_tree, inner_node_label_dict = label_hierarchical_tree(
        pruned_tree,
        leaf_node_counts=leaf_node_counts,
        num_samples_to_label=num_samples_to_label,
        examples_dataframe=examples_metadata['examples_df'],
        examples_metadata=examples_metadata,
        use_discretization=use_discretization,
        text_embeddings=text_embeddings,
        combined_cluster_texts=combined_cluster_texts,
        # experimental variations
        use_labels_and_descriptions=use_labels_and_descriptions,
        use_descriptions_and_examples=use_descriptions_and_examples,
        use_child_nodes=use_child_nodes,
        output_labels_and_descriptions=output_labels_and_descriptions,
        num_iterations_to_run_labeling=num_iterations_to_run_labeling,
        labeling_model=labeling_model,
        checking_model=checking_model,
        # discretization parameters
        batch_size=batch_size,
        num_summary_candidates_to_generate=num_summary_candidates_to_generate,
        num_datapoints_to_use_for_scoring=num_datapoints_to_use_for_scoring,
        num_leaf_samples_to_use_for_generation=num_leaf_samples_to_use_for_generation,
        discretization_goal=discretization_goal
    )
    save_hierarchical_tree(labeled_tree, labeled_tree_path)
    print(f"Saved labeled tree to: {labeled_tree_path}")
    
    # Save inner node labels
    inner_node_label_df = pd.DataFrame([
        {'node_id': node_id, 'label': label_info['label'], 'description': label_info['description']}
        for node_id, label_info in inner_node_label_dict.items()
    ])
    inner_node_label_df.to_csv(inner_node_labels_path, index=False)
    print(f"Saved inner node labels to: {inner_node_labels_path}")
    
    # Visualize the labeled tree if requested
    if visualize:
        # Set up visualization parameters
        plot_path = output_base / 'labeled_tree_visualization.png'
        
        # Create figure with appropriate size
        plt.figure(figsize=(16, 12))
        
        # Use plot_graph function to visualize
        plot_graph(
            labeled_tree,
            with_labels=True,
            with_node_id=True,
            with_sizes=True,
            label_name='label',
            node_color='lightblue',
            node_max_size=50,
            node_min_size=5,
            font_size=10
        )
        
        # Save the plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved tree visualization to: {plot_path}")
    
    print("Tree labeling and visualization completed successfully.")
    return labeled_tree, pruned_tree, inner_node_label_dict


def get_experiment_variant_name(args):
    """
    Generate a descriptive name for the experiment variant based on the flags used.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    str
        A descriptive name for the experiment variant
    """
    variant_parts = []
    
    # Add discretization variants
    if args.use_discretization:
        variant_parts.append("discretized")
        if args.use_labels_and_descriptions:
            variant_parts.append("labels_descriptions")
        if args.use_descriptions_and_examples:
            variant_parts.append("examples")
        if args.use_child_nodes:
            variant_parts.append("child-nodes")
        if args.output_labels_and_descriptions:
            variant_parts.append("output-labels-desc")
    
    # If no variants were added, return "standard"
    if not variant_parts:
        return "standard"
    
    return "__".join(variant_parts)

def main():
    parser = argparse.ArgumentParser(description="Perform hierarchical clustering on KMeans centroids.")
    parser.add_argument("centroids_file", help="Path to the file containing KMeans centroids (NumPy format)")
    parser.add_argument("output_dir", help="Directory to save the output files")
    parser.add_argument("--min_clusters", type=int, default=2, help="Minimum number of clusters")
    parser.add_argument("--max_clusters", type=int, default=None, help="Maximum number of clusters")
    parser.add_argument("--min_cluster_size", type=int, default=2, help="Minimum size a cluster must have")
    parser.add_argument("--method", default="ward", choices=["ward", "complete", "average", "single"], 
                        help="Linkage method for hierarchical clustering")
    parser.add_argument("--metric", default="euclidean", help="Distance metric for clustering")
    
    # Add argument for min_hierarchical_levels
    parser.add_argument("--min_hierarchical_levels", type=int, default=6, 
                        help="Minimum number of hierarchical levels to aim for (default: 6)")

    # Add arguments for labeling and visualization
    parser.add_argument("--label_tree", action="store_true", help="Label the hierarchical tree using utils_trees functions")
    parser.add_argument("--initial_cluster_labels_path", help="Path to the CSV file containing initial cluster labels")
    parser.add_argument("--min_descendants", type=int, default=2, help="Minimum descendants for tree pruning")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum depth for tree pruning")
    parser.add_argument("--num_samples_per_node_for_labeling", type=int, default=10, 
                       help="Number of samples to use for labeling each inner node in the tree")
    parser.add_argument("--no_visualize", action="store_true", help="Skip tree visualization")

    # Add arguments for examples
    parser.add_argument("--experiment", type=str, default="editorial", help="Experiment name")
    parser.add_argument("--datapoint_level_labels_path", help="Path to the CSV file containing low-level labels for each datapoint")
    parser.add_argument("--raw_data_examples_df_path", help="Path to the CSV file containing example sentences for each cluster")
    parser.add_argument("--num_examples_per_node", type=int, default=10, help="Number of examples to show per node in visualization")
    parser.add_argument("--examples_cluster_col", type=str, default="cluster", help="Column name for cluster IDs in examples_df")
    parser.add_argument("--examples_sentence_col", type=str, default="description", help="Column name for description in examples_df")

    # Add arguments for discretization
    parser.add_argument("--use_discretization", action="store_true", help="Use discretization approach for labeling")
    parser.add_argument("--discretization_goal", type=str, default="determine the common theme or purpose", 
                       help="Goal for discretization (e.g., 'determine the argumentation strategy')")
    parser.add_argument("--initial_datapoint_level_label_embeddings", 
                       help="Path to the embeddings for the labels of the datapoints (required if use_discretization is True)")
    parser.add_argument("--num_leaf_samples_to_use_for_generation", type=int, default=30,
                       help="How many datapoints to sample for LLM prompting")
    parser.add_argument("--num_summary_candidates_to_generate", type=int, default=10,
                       help="How many summary candidates to generate per slot")
    parser.add_argument("--num_datapoints_to_use_for_scoring", type=int, default=300,
                       help="How many datapoints to sample for scoring")
    parser.add_argument("--batch_size", type=int, default=30,
                       help="How many summaries to score at a time")

    # experimental variations
    parser.add_argument("--use_labels_and_descriptions", action="store_true", 
                       help="Use labels and descriptions for summary generation")
    parser.add_argument("--use_descriptions_and_examples", action="store_true", 
                       help="Use descriptions and examples for summary generation")
    parser.add_argument("--use_child_nodes", action="store_true", 
                       help="Use child nodes for summary generation")
    parser.add_argument("--output_labels_and_descriptions", action="store_true", 
                       help="Output labels and descriptions for summary generation")
    parser.add_argument("--num_iterations_to_run_labeling", type=int, default=2, 
                       help="Number of iterations to run labeling for discretization")
    
    # models
    parser.add_argument("--labeling_model", type=str, default='gpt-4o-mini', help="Model to use for labeling nodes")
    parser.add_argument("--checking_model", type=str, default='gpt-4o-mini', help="Model to use for checking labels")
    args = parser.parse_args()

    # Generate descriptive output directory name
    experiment_variant = get_experiment_variant_name(args)
    output_dir = f"{args.output_dir}__{experiment_variant}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using output directory: {output_dir}")

    # Load examples dataframe if provided
    examples_result = custom_examples_loader(
        data_level_labels_path=args.datapoint_level_labels_path,
        examples_dataframe_path=args.raw_data_examples_df_path,
        initial_labels_embeddings_path=args.initial_datapoint_level_label_embeddings,
        use_discretization=args.use_discretization,
        experiment=args.experiment
    )
    if examples_result is None:
        examples_df, text_embeddings = None, None
    else:
        examples_df, text_embeddings = examples_result

    # Run hierarchical clustering
    result = run_hierarchical_clustering(
        args.centroids_file,
        str(output_dir),  # Convert Path to string
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        min_cluster_size=args.min_cluster_size,
        method=args.method,
        metric=args.metric,
        examples_df=examples_df,
        min_hierarchical_levels=args.min_hierarchical_levels
    )
    
    # Label and visualize the tree if requested
    if args.label_tree:
        if not args.initial_cluster_labels_path:
            raise ValueError("Initial labels must be provided to label the tree.")
        
        print("Labeling and visualizing tree...")
        label_and_visualize_tree(
            tree=result['tree'],
            initial_cluster_labels_path=args.initial_cluster_labels_path,
            output_dir_path=output_dir,  # Use the new output directory
            leaf_node_counts=result['cluster_node_ids'],
            min_descendants=args.min_descendants,
            max_depth=args.max_depth,
            num_samples_to_label=args.num_samples_per_node_for_labeling,
            visualize=(not args.no_visualize),
            examples_metadata={
                'examples_df': examples_df,
                'num_examples_per_node': args.num_examples_per_node,
                'cluster_col': args.examples_cluster_col,
                'sentence_col': args.examples_sentence_col
            },
            use_discretization=args.use_discretization,
            text_embeddings=text_embeddings,
            discretization_goal=args.discretization_goal,
            # experimental variations
            use_labels_and_descriptions=args.use_labels_and_descriptions,
            use_descriptions_and_examples=args.use_descriptions_and_examples,
            use_child_nodes=args.use_child_nodes,
            output_labels_and_descriptions=args.output_labels_and_descriptions,
            num_iterations_to_run_labeling=args.num_iterations_to_run_labeling,
            labeling_model=args.labeling_model,
            checking_model=args.checking_model,
            # discretization parameters
            batch_size=args.batch_size,
            num_summary_candidates_to_generate=args.num_summary_candidates_to_generate,
            num_datapoints_to_use_for_scoring=args.num_datapoints_to_use_for_scoring,
            num_leaf_samples_to_use_for_generation=args.num_leaf_samples_to_use_for_generation
        )

if __name__ == "__main__":
    main() 

"""
    python src/step_6__agglomerative_clustering.py \
        experiments/editorial/models/cluster_centroids.npy \
        experiments/editorial/hierarchy_results \
        --min_clusters 2 \
        --max_clusters 15 \
        --min_cluster_size 2 \
        --method ward \
        --metric euclidean \
        --label_tree \
        --initial_labels experiments/editorial/models/cluster_labels.csv \
        --min_descendants 2 \
        --max_depth 8 \
        --num_samples 10  \
        --examples_df experiments/editorial/models/all_extracted_discourse_with_clusters.csv \
        --num_examples_per_node 10 \
        --examples_cluster_col cluster \
        --examples_sentence_col sentences

    # Example with discretization:
    python src/step_6__agglomerative_clustering.py \
        experiments/editorial/models/cluster_centroids.npy \
        experiments/editorial/hierarchy_results_discretized \
        --min_clusters 2 \
        --max_clusters 15 \
        --min_cluster_size 2 \
        --method ward \
        --metric euclidean \
        --label_tree \
        --initial_labels experiments/editorial/models/cluster_labels.csv \
        --min_descendants 2 \
        --max_depth 8 \
        --num_samples 10  \
        --examples_df experiments/editorial/models/all_extracted_discourse_with_clusters.csv \
        --num_examples_per_node 10 \
        --examples_cluster_col cluster \
        --examples_sentence_col sentences \
        --use_discretization \
        --use_labels_and_descriptions \
        --use_descriptions_and_examples \
        --use_child_nodes \
        --output_labels_and_descriptions \
        --discretization_goal "determine the argumentation strategy" \
        --initial_datapoint_level_label_embeddings experiments/editorial/initial_label_embeddings_cache.npy
"""
