import argparse
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from pathlib import Path
import sys
from tqdm import tqdm

from em_scores import compute_document_level_scores, compute_corpus_level_scores

# --- EM Refinement Constants ---
# For decide_schema_updates
SIMILARITY_THRESHOLD_MERGE = 0.9  # Cosine similarity for merging cluster centroids
MIN_CLUSTER_SIZE_REMOVE = 5     # Minimum number of items for a cluster to be kept
BASELINE_LL_FACTOR_REMOVE = 1.01 # Factor for L_z vs L_baseline for removal (L_z < L_baseline * FACTOR)
BASELINE_LL_FACTOR_SPLIT = 0.9   # Factor for L_z vs L_baseline for splitting (L_z < L_baseline * FACTOR)
CONFIDENCE_THRESHOLD_SPLIT = 0.4 # Minimum posterior confidence C(z) for splitting
# For em_schema_refinement (Add step)
LOW_CONFIDENCE_THRESHOLD_ADD = 0.1 # Max p(z|x) for a text to be considered poorly explained
MIN_POORLY_EXPLAINED_FOR_ADD = 10  # Min number of poorly explained texts to trigger new cluster formation
ITEMS_PER_NEW_CLUSTER_ADD = 50   # Heuristic: 1 new cluster per this many poorly explained items

DEFAULT_BASELINE_SAMPLE_SIZE = 500 # Default sample size for baseline P(x) calculation
DEFAULT_MAX_TEXTS_PER_CLUSTER_METRICS = -1 # Default max texts per cluster for metrics; -1 means no limit

try:
    from .utils_probability_calibrator import ProbabilityCalibrator, initialize_probability_calibrator
except ImportError:
    try:
        from utils_probability_calibrator import ProbabilityCalibrator, initialize_probability_calibrator
    except ImportError:
        raise ImportError("Failed to import ProbabilityCalibrator and initialize_probability_calibrator. Please check your project structure and PYTHONPATH.")


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def determine_target_clustering_info(agglomerative_output_dir: Path, agglomerative_level_spec: int | None, num_agglomerative_clusters_spec: int | None) -> tuple[int, int, Path]:
    """
    Determines the target agglomerative clustering level, number of clusters,
    and the path to the corresponding cluster assignment file.
    """
    optimal_thresholds_path = agglomerative_output_dir / "optimal_thresholds.csv"
    if not optimal_thresholds_path.exists():
        raise FileNotFoundError(f"optimal_thresholds.csv not found in {agglomerative_output_dir}")

    optimal_thresholds_df = pd.read_csv(optimal_thresholds_path)
    if 'level' not in optimal_thresholds_df.columns:
        # Assuming levels are 1-indexed based on row order if not present
        optimal_thresholds_df['level'] = range(1, len(optimal_thresholds_df) + 1)

    target_level = None
    target_n_clusters = None

    if agglomerative_level_spec is not None:
        target_level = agglomerative_level_spec
        level_info = optimal_thresholds_df[optimal_thresholds_df['level'] == target_level]
        if level_info.empty:
            raise ValueError(f"Level {target_level} not found in {optimal_thresholds_path}. Available levels: {sorted(optimal_thresholds_df['level'].unique())}")
        target_n_clusters = level_info['n_clusters'].iloc[0]
    elif num_agglomerative_clusters_spec is not None:
        target_n_clusters_requested = num_agglomerative_clusters_spec
        n_clusters_info = optimal_thresholds_df[optimal_thresholds_df['n_clusters'] == target_n_clusters_requested]
        
        if n_clusters_info.empty:
            logging.warning(f"Exact number of clusters {target_n_clusters_requested} not found in {optimal_thresholds_path}.")
            # Find the closest number of clusters
            available_n_clusters = sorted(optimal_thresholds_df['n_clusters'].unique())
            if not available_n_clusters:
                raise ValueError(f"No cluster counts available in {optimal_thresholds_path} to find a closest match for {target_n_clusters_requested}.")
            
            closest_n_clusters = min(available_n_clusters, key=lambda x: abs(x - target_n_clusters_requested))
            logging.warning(f"Using the closest available number of clusters: {closest_n_clusters}.")
            target_n_clusters = closest_n_clusters
            # Now get the info for this closest_n_clusters
            n_clusters_info = optimal_thresholds_df[optimal_thresholds_df['n_clusters'] == target_n_clusters]
            # If multiple levels have this closest_n_clusters, pick the first (usually lowest level/highest threshold)
            if len(n_clusters_info) > 1:
                logging.warning(f"Multiple levels found for the closest cluster count {target_n_clusters} (e.g., levels {n_clusters_info['level'].tolist()}). Using the first one: level {n_clusters_info['level'].iloc[0]}.")
            target_level = n_clusters_info['level'].iloc[0]
        else:
            # Exact match found
            target_n_clusters = target_n_clusters_requested
            if len(n_clusters_info) > 1:
                logging.warning(f"Multiple levels found for {target_n_clusters} clusters. Using the first one (level {n_clusters_info['level'].iloc[0]}).")
            target_level = n_clusters_info['level'].iloc[0]
    
    if target_level is None or target_n_clusters is None:
        raise ValueError("Could not determine target level and number of clusters from arguments.")

    target_level = int(target_level)
    target_n_clusters = int(target_n_clusters)

    selected_level_assignments_filename = f"clusters_level_{target_level}_{target_n_clusters}_clusters.csv"
    selected_level_assignments_path = agglomerative_output_dir / selected_level_assignments_filename

    if not selected_level_assignments_path.exists():
        raise FileNotFoundError(f"Cluster assignment file not found: {selected_level_assignments_path}")
    
    logging.info(f"Targeting agglomerative clustering: Level {target_level}, N_Clusters {target_n_clusters}")
    logging.info(f"Using cluster assignment file: {selected_level_assignments_path}")
    return target_level, target_n_clusters, selected_level_assignments_path


def load_and_prepare_data(sentence_data_path: Path, selected_level_assignments_path: Path, agglomerative_labels_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Loads sentence data, cluster assignments, and cluster labels, then merges them.
    Returns the merged DataFrame and the list of choices (textual labels) for the calibrator.
    """
    logging.info(f"Loading sentence data from: {sentence_data_path}")
    sentences_df = pd.read_csv(sentence_data_path)
    if 'cluster' in sentences_df.columns:
        sentences_df.rename(columns={'cluster': 'centroid_id'}, inplace=True)
    elif 'centroid_id' not in sentences_df.columns:
        raise KeyError(f"Expected 'cluster' or 'centroid_id' column in {sentence_data_path}")

    logging.info(f"Loading agglomerative cluster labels from: {agglomerative_labels_path}")
    agg_labels_df = pd.read_csv(agglomerative_labels_path)  # columns: node_id, label, description

    logging.info(f"Loading selected level assignments from: {selected_level_assignments_path}")
    level_assignments_df = pd.read_csv(selected_level_assignments_path)  # columns: centroid_id, fcluster_label, graph_node_id

    # Ensure the necessary columns are present
    if 'centroid_id' not in level_assignments_df.columns or \
       'graph_node_id' not in level_assignments_df.columns:
        raise KeyError(f"File {selected_level_assignments_path} must contain 'centroid_id' and 'graph_node_id' columns. Found: {level_assignments_df.columns.tolist()}")

    merged_df = pd.merge(sentences_df, level_assignments_df, on='centroid_id', how='inner')
    if merged_df.empty:
        raise ValueError("No data after merging sentences with cluster assignments. Check 'centroid_id' alignment and file contents.")
    
    # Use graph_node_id as the primary cluster identifier moving forward for label mapping
    merged_df.rename(columns={'graph_node_id': 'cluster_id'}, inplace=True)
    merged_df['cluster_id'] = merged_df['cluster_id'].astype(int) # This is now the graph_node_id

    # Map agglomerative cluster IDs (which are now graph_node_ids) to their textual labels
    unique_agg_cluster_ids_at_level = sorted(merged_df['cluster_id'].unique())
    
    # Filter agg_labels_df for relevant node_ids and ensure 'label' column exists
    if 'node_id' not in agg_labels_df.columns or 'label' not in agg_labels_df.columns:
        raise KeyError(f"Agglomerative labels file {agglomerative_labels_path} must contain 'node_id' and 'label' columns.")

    level_textual_labels_df = (
        agg_labels_df
            .loc[lambda df: df['node_id'].isin(unique_agg_cluster_ids_at_level)]
            .copy()
    )
    if level_textual_labels_df.empty and unique_agg_cluster_ids_at_level:
         logging.warning(f"No textual labels found in {agglomerative_labels_path} for the agglomerative cluster IDs {unique_agg_cluster_ids_at_level} active at the selected level. Calibrator choices will be based on node_ids if textual labels are missing.")


    # Sort by node_id for consistent order of choices_list
    level_textual_labels_df.sort_values(by='node_id', inplace=True)
    id_to_label_map = pd.Series(level_textual_labels_df.label.values, index=level_textual_labels_df.node_id).to_dict()
    
    merged_df['agglomerative_label'] = merged_df['cluster_id'].map(id_to_label_map)

    # Ensure choices are strings. If a label is missing, use the cluster_id as a string.
    choices_list = []
    for cid in sorted(unique_agg_cluster_ids_at_level):
        label = id_to_label_map.get(cid)
        if label is None:
            logging.warning(f"No textual label for agglomerative cluster_id {cid}. Using ID as label.")
            choices_list.append(str(cid))
            if 'agglomerative_label' in merged_df.columns and merged_df.loc[merged_df['cluster_id'] == cid, 'agglomerative_label'].isnull().any():
                 merged_df.loc[merged_df['cluster_id'] == cid, 'agglomerative_label'] = str(cid)
        else:
            choices_list.append(str(label))


    if not choices_list and unique_agg_cluster_ids_at_level: # only raise if there were clusters but no labels formed
        raise ValueError(f"Could not form a list of choices (textual labels) for the calibrator from {agglomerative_labels_path}.")
    if not unique_agg_cluster_ids_at_level:
        raise ValueError("No unique agglomerative cluster IDs found in the data for the selected level. Cannot proceed.")

    logging.info(f"Agglomerative cluster choices for ProbabilityCalibrator: {choices_list}")
    return merged_df, choices_list


def compute_sentence_embeddings(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
    """Returns a numpy array (len(texts), dim) of embeddings using SentenceTransformer."""
    from sentence_transformers import SentenceTransformer
    logging.info(f"Loading SentenceTransformer model: {model_name}")
    st_model = SentenceTransformer(model_name)
    embeddings = st_model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def score_dataframe_sentences(calibrator: ProbabilityCalibrator, data_df: pd.DataFrame, text_column: str, choices_list_for_output: list[str]) -> pd.DataFrame:
    """Scores sentences in the DataFrame using the provided calibrator."""
    logging.info(f"Scoring sentences from column: '{text_column}'...")
    results_list = []

    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Scoring sentences"):
        sentence_text = str(row[text_column])
        try:
            probabilities = calibrator.calibrate_p_z_given_X(sentence_text)
        except Exception as e:
            logging.error(f"Error scoring sentence: '{sentence_text[:100]}...'. Error: {e}")
            # Fill probabilities with NaN or a specific error marker if scoring fails for a row
            probabilities = [np.nan] * len(choices_list_for_output)

        res_row = {
            'sentence_text': sentence_text,
            'original_kmeans_cluster_id': row.get('centroid_id', np.nan),
            'selected_agglomerative_cluster_id': row.get('cluster_id', np.nan),
            'selected_agglomerative_cluster_label': row.get('agglomerative_label', 'N/A')
        }
        for i, choice_label in enumerate(choices_list_for_output):
            sane_choice_label = "".join(c if c.isalnum() else "_" for c in str(choice_label))
            # Ensure index i is within bounds of probabilities list
            res_row[f'prob_{sane_choice_label}'] = probabilities[i] if i < len(probabilities) else np.nan
        results_list.append(res_row)

    return pd.DataFrame(results_list)


def compute_cluster_metrics(
        merged_df: pd.DataFrame, 
        embeddings: np.ndarray, 
        idx_to_emb_pos_map: dict,
        prob_calibrator: ProbabilityCalibrator, 
        choices_list: list[str], 
        baseline_log_px: float, 
        verbose: bool = False,
        text_column: str = 'sentence_text',
        max_texts_per_cluster_metrics: int = DEFAULT_MAX_TEXTS_PER_CLUSTER_METRICS
    ) -> dict:
    """Compute per-cluster metrics: average log-likelihood, posterior confidence, variance."""
    logging.info("Starting computation of cluster metrics...")
    cluster_metrics = {}
    choice_to_idx = {c: i for i, c in enumerate(choices_list)}

    # Determine the iterator for clusters, potentially with tqdm
    cluster_groups = merged_df.groupby('cluster_id')
    if verbose:
        cluster_iterator = tqdm(cluster_groups, desc="Computing cluster metrics", unit="cluster")
    else:
        cluster_iterator = cluster_groups

    for cluster_id, group in cluster_iterator:
        indices = group.index.tolist()
        if not indices:
            logging.warning(f"Cluster {cluster_id} has no data points. Skipping metric calculation for it.")
            continue
        
        # Map original index labels to 0-based positions in the embeddings array
        embedding_positions = []
        valid_group_indices_for_embedding = [] # Store original indices that have a map entry
        for original_idx in group.index:
            if original_idx in idx_to_emb_pos_map:
                embedding_positions.append(idx_to_emb_pos_map[original_idx])
                valid_group_indices_for_embedding.append(original_idx)
            else:
                logging.warning(f"Original index {original_idx} from cluster {cluster_id} not found in idx_to_emb_pos_map. Skipping this datapoint for embedding.")

        if not embedding_positions:
            logging.warning(f"Cluster {cluster_id} has no valid embeddings after mapping. Assigning default metrics.")
            # Ensure label is available for consistent metric structure
            current_label_for_metric = group['agglomerative_label'].iloc[0] if not group.empty and 'agglomerative_label' in group.columns else f"cluster_{cluster_id}"
            cluster_metrics[cluster_id] = {
                'L_z': float('-inf'), 'C_z': 0.0, 'V_z': 0.0, 
                'size': len(indices), # Original size before mapping issues
                'mapped_size': 0, # Number of items for which embeddings were found
                'embedding_centroid': np.zeros(embeddings.shape[1] if embeddings.ndim > 1 else 0),
                'label': current_label_for_metric
            }
            continue
            
        texts_for_metrics = group.loc[valid_group_indices_for_embedding, text_column].tolist()
        emb_for_metrics = embeddings[embedding_positions]
        current_mapped_size = len(texts_for_metrics)

        if max_texts_per_cluster_metrics > 0 and current_mapped_size > max_texts_per_cluster_metrics:
            logging.info(f"Cluster {cluster_id} ('{group['agglomerative_label'].iloc[0] if not group.empty else 'N/A'}') has {current_mapped_size} texts, sampling to {max_texts_per_cluster_metrics} for metric calculation.")
            sample_indices = np.random.choice(current_mapped_size, size=max_texts_per_cluster_metrics, replace=False)
            texts_for_metrics = [texts_for_metrics[i] for i in sample_indices]
            emb_for_metrics = emb_for_metrics[sample_indices] # Sample corresponding embeddings
            # Update current_mapped_size to reflect the sample size for metrics
            # The 'mapped_size' in the final output will reflect this sampling if it occurs

        e_z = emb_for_metrics.mean(axis=0)
        variances = ((emb_for_metrics - e_z) ** 2).sum(axis=1)
        V_z = variances.mean()

        log_likelihoods = []
        posterior_probs = []
        label = group.loc[valid_group_indices_for_embedding, 'agglomerative_label'].iloc[0] if valid_group_indices_for_embedding else f"cluster_{cluster_id}"
        label_idx = choice_to_idx.get(label)

        if label_idx is None:
            logging.warning(f"Label '{label}' for cluster {cluster_id} not in calibrator choices {choices_list}. Assigning default likelihood/confidence.")
            L_z = float('-inf')
            C_z = 0.0
        else:
            text_iterator = texts_for_metrics # Use the (potentially sampled) texts
            if verbose:
                # TQDM should now reflect the size of texts_for_metrics
                text_iterator = tqdm(texts_for_metrics, desc=f"Scoring texts in cluster {cluster_id} ('{label[:20]}...')", unit="text", leave=False, total=len(texts_for_metrics))
            
            for txt in text_iterator:
                log_p_z_given_x = prob_calibrator.compute_log_p_z_given_X(txt)
                log_likelihoods.append(float(log_p_z_given_x[label_idx]))
                posterior_probs.append(float(torch.exp(log_p_z_given_x[label_idx])))
            L_z = np.mean(log_likelihoods) if log_likelihoods else float('-inf')
            C_z = np.mean(posterior_probs) if posterior_probs else 0.0

        cluster_metrics[cluster_id] = {
            'L_z': L_z,
            'C_z': C_z,
            'V_z': V_z,
            'size': len(indices), 
            'mapped_size': len(texts_for_metrics), # This now reflects sampled size if sampling occurred
            'embedding_centroid': e_z,
            'label': label
        }
        logging.debug(f"Cluster {cluster_id} ('{label}', size {len(indices)}, mapped_size {len(texts_for_metrics)}): L_z={L_z:.4f}, C_z={C_z:.4f}, V_z={V_z:.4f}")
    
    cluster_metrics['_baseline_log_px'] = baseline_log_px # Use a key that won't be mistaken for a cluster ID
    logging.info(f"Finished computation of cluster metrics. Baseline log p(x) = {baseline_log_px:.4f}")
    return cluster_metrics


def decide_schema_updates(cluster_metrics: dict):
    """Determine which clusters to split, merge, remove. Returns dictionaries describing actions."""
    logging.info("Starting schema update decisions...")
    actions = {'split': [], 'merge': [], 'remove': []}
    # Extract baseline and actual cluster data separately
    baseline_log_px = cluster_metrics.pop('_baseline_log_px', 0.0)
    
    # Filter out any non-cluster data that might have been passed in cluster_metrics keys
    valid_cluster_ids = [k for k, v in cluster_metrics.items() if isinstance(v, dict) and 'L_z' in v]
    if not valid_cluster_ids:
        logging.warning("No valid cluster data found in cluster_metrics for schema update decisions.")
        return actions

    median_V_z = np.median([cluster_metrics[cid]['V_z'] for cid in valid_cluster_ids if 'V_z' in cluster_metrics[cid]])
    logging.info(f"Using L_baseline={baseline_log_px:.4f}, Median V_z={median_V_z:.4f} for decisions.")

    # --- SPLIT & REMOVE ---
    for cid in valid_cluster_ids:
        m = cluster_metrics[cid]
        # Removal Check
        if m['size'] < MIN_CLUSTER_SIZE_REMOVE and m['L_z'] < baseline_log_px * BASELINE_LL_FACTOR_REMOVE:
            actions['remove'].append(cid)
            logging.info(f"  Suggest REMOVE for cluster {cid} ('{m.get('label', 'N/A')}'): size {m['size']} < {MIN_CLUSTER_SIZE_REMOVE} AND L_z {m['L_z']:.4f} < baseline_thresh {baseline_log_px * BASELINE_LL_FACTOR_REMOVE:.4f}.")
            continue # If removed, don't consider for split
        # Split Check
        if m['L_z'] < baseline_log_px * BASELINE_LL_FACTOR_SPLIT and \
           m['C_z'] < CONFIDENCE_THRESHOLD_SPLIT and \
           m['V_z'] > median_V_z:
            actions['split'].append(cid)
            logging.info(f"  Suggest SPLIT for cluster {cid} ('{m.get('label', 'N/A')}'): L_z {m['L_z']:.4f} < thresh {baseline_log_px * BASELINE_LL_FACTOR_SPLIT:.4f} AND C_z {m['C_z']:.4f} < {CONFIDENCE_THRESHOLD_SPLIT} AND V_z {m['V_z']:.4f} > median_V_z {median_V_z:.4f}.")

    # --- MERGE ---
    # Filter out clusters already marked for removal or split before considering for merge
    eligible_for_merge_ids = [cid for cid in valid_cluster_ids if cid not in actions['remove'] and cid not in actions['split']]
    if len(eligible_for_merge_ids) < 2:
        logging.info("Not enough eligible clusters to consider merging.")
    else:
        centroids = np.stack([cluster_metrics[cid]['embedding_centroid'] for cid in eligible_for_merge_ids])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        cent_norm = centroids / (norms + 1e-8)
        sim_matrix = cent_norm @ cent_norm.T
        n = len(eligible_for_merge_ids)
        merged_already = set()
        for i in range(n):
            if eligible_for_merge_ids[i] in merged_already:
                continue
            for j in range(i + 1, n):
                if eligible_for_merge_ids[j] in merged_already:
                    continue
                
                id_i = eligible_for_merge_ids[i]
                id_j = eligible_for_merge_ids[j]
                
                if sim_matrix[i, j] >= SIMILARITY_THRESHOLD_MERGE:
                    m_i = cluster_metrics[id_i]
                    m_j = cluster_metrics[id_j]
                    diff_L = abs(m_i['L_z'] - m_j['L_z'])
                    diff_C = abs(m_i['C_z'] - m_j['C_z'])
                    # Heuristic: check if L and C are 'similar enough' (e.g. within 10% of each other or small absolute diff)
                    if diff_L < 0.1 * max(abs(m_i['L_z']), abs(m_j['L_z']), 0.1) and \
                       diff_C < 0.1 * max(m_i['C_z'], m_j['C_z'], 0.1):
                        actions['merge'].append(tuple(sorted((id_i, id_j)))) # Store sorted tuple to avoid duplicates like (B,A) if (A,B) decided
                        merged_already.add(id_i)
                        merged_already.add(id_j)
                        logging.info(f"  Suggest MERGE for clusters {id_i} ('{m_i.get('label', 'N/A')}') and {id_j} ('{m_j.get('label', 'N/A')}'): Similarity {sim_matrix[i,j]:.4f} >= {SIMILARITY_THRESHOLD_MERGE}, L_diff {diff_L:.4f}, C_diff {diff_C:.4f}.")
                        break # Merge id_i with one cluster, then move to next i
    
    # Deduplicate merge pairs (if (A,B) and (B,A) somehow got in)
    actions['merge'] = sorted(list(set(actions['merge'])))
    logging.info(f"Finished schema update decisions. Actions: {actions}")
    return actions


def apply_schema_updates(merged_df: pd.DataFrame, embeddings: np.ndarray, actions: dict, next_new_cluster_id: int) -> tuple[pd.DataFrame, int]:
    """Apply split, merge, remove actions to merged_df and return updated df and next available cluster id."""
    from sklearn.cluster import KMeans
    logging.info(f"Starting to apply schema updates. Initial #clusters: {merged_df['cluster_id'].nunique()}")
    current_df = merged_df.copy() # Work on a copy

    # Remove clusters
    if actions['remove']:
        logging.info(f"Removing {len(actions['remove'])} clusters: {actions['remove']}")
        current_df = current_df[~current_df['cluster_id'].isin(actions['remove'])].copy()
        logging.info(f"  #clusters after removal: {current_df['cluster_id'].nunique()}")

    # Merge clusters: map second id to first id in each pair
    if actions['merge']:
        logging.info(f"Merging {len(actions['merge'])} pairs: {actions['merge']}")
        parent = {cid: cid for cid in current_df['cluster_id'].unique()}
        def find_set(item):
            if parent[item] == item:
                return item
            parent[item] = find_set(parent[item])
            return parent[item]
        def unite_sets(a, b):
            a_root = find_set(a)
            b_root = find_set(b)
            if a_root != b_root:
                parent[b_root] = a_root
        
        for c1, c2 in actions['merge']:
            if c1 in parent and c2 in parent:
                 unite_sets(c1, c2)
            else:
                logging.warning(f"Skipping merge of ({c1},{c2}), one or both clusters no longer exist or were not in initial parent map.")
        current_df['cluster_id'] = current_df['cluster_id'].apply(lambda x: find_set(x) if x in parent else x)
        logging.info(f"  #clusters after merge: {current_df['cluster_id'].nunique()}")

    # Split clusters
    if actions['split']:
        logging.info(f"Splitting {len(actions['split'])} clusters: {actions['split']}")
        for cid_to_split in actions['split']:
            if cid_to_split not in current_df['cluster_id'].unique():
                logging.warning(f"Cluster {cid_to_split} marked for split, but it no longer exists (possibly merged). Skipping split.")
                continue
            subset_indices = current_df[current_df['cluster_id'] == cid_to_split].index
            if len(subset_indices) < 2:
                logging.warning(f"Cluster {cid_to_split} has < 2 items after other ops. Cannot split.")
                continue
            emb_subset = embeddings[subset_indices.to_numpy()]
            kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
            try:
                labels_split = kmeans.fit_predict(emb_subset)
            except Exception as e:
                logging.error(f"KMeans failed for splitting cluster {cid_to_split}: {e}. Skipping split.")
                continue
            new_split_id = next_new_cluster_id
            next_new_cluster_id += 1
            split_assignment_map = {}
            for i, original_df_idx in enumerate(subset_indices):
                if labels_split[i] == 0:
                    split_assignment_map[original_df_idx] = cid_to_split
                else:
                    split_assignment_map[original_df_idx] = new_split_id
            for original_df_idx, new_cluster_assignment in split_assignment_map.items():
                current_df.loc[original_df_idx, 'cluster_id'] = new_cluster_assignment
            logging.info(f"  Split cluster {cid_to_split} into {cid_to_split} and {new_split_id}. New #clusters: {current_df['cluster_id'].nunique()}")

    logging.info(f"Finished applying schema updates. Final #clusters: {current_df['cluster_id'].nunique()}")
    return current_df, next_new_cluster_id


def em_schema_refinement(
        merged_df: pd.DataFrame,
        prob_calibrator: ProbabilityCalibrator, 
        choices_list: list[str], 
        text_column_name: str,
        model_identifier: str, 
        model_type: str, 
        max_iters: int = 3, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        baseline_sample_size: int = DEFAULT_BASELINE_SAMPLE_SIZE,
        verbose: bool = False,
        max_texts_per_cluster_metrics: int = DEFAULT_MAX_TEXTS_PER_CLUSTER_METRICS
    ):
    """Run an EM-like schema refinement loop, returning updated merged_df and history of schema changes."""
    logging.info(f"Starting EM-style schema refinement for {max_iters} iterations...")
    logging.info(f"Computing initial sentence embeddings using {embedding_model_name}...")
    
    # Create a mapping from the original DataFrame index to 0-based positions for the embeddings array
    # This assumes current_merged_df's index at this point is the reference for all_embeddings
    idx_to_emb_pos_map = {original_idx: pos for pos, original_idx in enumerate(merged_df.index)}
    all_embeddings = compute_sentence_embeddings(merged_df[text_column_name].tolist(), model_name=embedding_model_name)

    current_merged_df = merged_df.copy()
    current_choices_list = list(choices_list)
    current_prob_calibrator = prob_calibrator
    schema_history = []
    if current_merged_df['cluster_id'].empty:
        logging.warning("Initial merged_df has no cluster_id assignments. Starting new cluster IDs from 0.")
        next_new_cluster_id = 0
    else:
        next_new_cluster_id = current_merged_df['cluster_id'].max() + 1
    logging.info("Calculating baseline log p(x) for the dataset...")
    sample_size_for_baseline = min(baseline_sample_size, len(current_merged_df))
    baseline_log_px = 0.0 # Initialize baseline_log_px
    if sample_size_for_baseline > 0:
        texts_for_baseline_sample = current_merged_df[text_column_name].sample(sample_size_for_baseline).tolist()
        baseline_log_px = current_prob_calibrator.compute_average_log_px_from_sample(texts_for_baseline_sample)
        
        logging.info(f"Baseline log p(x) = {baseline_log_px:.4f} (based on {sample_size_for_baseline} samples)")

        # NEW: E-step scoring
        doc_scores = compute_document_level_scores(
            df=current_merged_df,
            text_col=text_column_name,
            cluster_col="agglomerative_label",
            choices=current_choices_list,
            prob_calibrator=current_prob_calibrator,
            embeddings=all_embeddings if 'all_embeddings' in locals() else None,
            p_z_prior=getattr(current_prob_calibrator, "p_z", None),  # uses dataset prior if available
        )
        # Persist per-row pmax / assignments for analysis
        current_merged_df["pmax_explained"] = doc_scores["row_pmax"]     
        current_merged_df["z_hat_explainer"] = doc_scores["row_z_hat"]   

        # Optionally compute corpus scores (treat all as test or pass a mask)
        corpus_scores = compute_corpus_level_scores(
            log_px_given_z_hat=doc_scores["row_log_px_given_z_norm"][np.arange(len(current_merged_df)),
                                                                    current_merged_df["z_hat_explainer"].values],
            token_counts=None,             # plug true token counts if you track them
            k_complexity=len(current_choices_list),  # simple complexity proxy
            is_test_mask=None,
        )  

        logging.info(f"[E-step] L_baseline={doc_scores['L_baseline']:.4f}, "
                    f"logL_total={corpus_scores['logL_cond_total']:.2f}, "
                    f"AIC={corpus_scores['AIC']}, BIC={corpus_scores['BIC']}, "
                    f"PPL={corpus_scores['perplexity']:.3f}")  
    else:
        logging.warning("  Dataset is empty or too small for baseline sample, cannot compute baseline log p(x). Using {baseline_log_px}.")

    for iter_idx in range(max_iters):
        logging.info(f"=== EM Iteration {iter_idx + 1}/{max_iters} ===")
        iteration_actions = {}
        if current_merged_df.empty:
            logging.warning(f"Iteration {iter_idx + 1}: Dataframe is empty. Stopping EM refinement.")
            break
        logging.info(f"Iteration {iter_idx + 1}: Computing cluster metrics for {current_merged_df['cluster_id'].nunique()} clusters.")
        cluster_metrics = compute_cluster_metrics(
            current_merged_df, 
            all_embeddings, 
            idx_to_emb_pos_map,
            current_prob_calibrator, 
            current_choices_list, 
            baseline_log_px,
            verbose=verbose,
            text_column=text_column_name,
            max_texts_per_cluster_metrics=max_texts_per_cluster_metrics
        )
        actions = decide_schema_updates(cluster_metrics)
        iteration_actions.update(actions)
        logging.info(f"Iteration {iter_idx + 1}: Actions decided from metrics (Split/Merge/Remove): {actions}. 'Add' actions to be determined next.")
        current_merged_df, next_new_cluster_id = apply_schema_updates(current_merged_df, all_embeddings, actions, next_new_cluster_id)
        
        # Log P(z|X) for each datapoint if verbose
        if verbose:
            logging.info(f"Iteration {iter_idx + 1} - Datapoint Posterior Probabilities (P(z|X)) for {current_merged_df['cluster_id'].nunique()} clusters:")
            if not current_merged_df.empty and current_choices_list:
                # Create a temporary calibrator with the current choices for this logging section if it's different
                # This ensures probabilities are logged against the correct set of choices for the current iteration state
                temp_calibrator_for_logging = current_prob_calibrator 
                # Check if current_prob_calibrator's choices match current_choices_list, if not, reinitialize (should generally match)
                if set(temp_calibrator_for_logging.choices) != set(current_choices_list):
                    logging.debug("Re-initializing temp calibrator for P(z|X) logging due to choice mismatch.")
                    temp_calibrator_for_logging = initialize_probability_calibrator(
                        model_identifier=model_identifier,
                        model_type=model_type,
                        choices=current_choices_list, # Use the most up-to-date choices
                        num_trials=current_prob_calibrator.num_trials,
                        scorer_type=current_prob_calibrator.scorer_type,
                        verbose=verbose
                    )

                for original_idx in current_merged_df.index:
                    sentence_text = str(current_merged_df.loc[original_idx, text_column_name])
                    # Get P(z|X) using the potentially re-initialized calibrator for the current choices
                    probabilities_for_log = temp_calibrator_for_logging.calibrate_p_z_given_X(sentence_text)
                    prob_map_str = ", ".join([
                        f"'{label}': {prob:.4f}" 
                        for label, prob in zip(temp_calibrator_for_logging.choices, probabilities_for_log)
                    ])
                    logging.info(f"  Datapoint (idx {original_idx}, text: '{sentence_text[:80]}...'): {{{prob_map_str}}}")
            else:
                logging.info("  No data or choices to log P(z|X).")

        logging.info(f"Iteration {iter_idx + 1}: Identifying poorly explained texts for potential new clusters.")
        poor_indices = []
        for original_idx in current_merged_df.index:
            sentence_text = current_merged_df.loc[original_idx, text_column_name]
            probabilities = current_prob_calibrator.calibrate_p_z_given_X(sentence_text)
            if not probabilities.any() or probabilities.max() < LOW_CONFIDENCE_THRESHOLD_ADD:
                poor_indices.append(original_idx)
        added_clusters_info = []
        if len(poor_indices) >= MIN_POORLY_EXPLAINED_FOR_ADD:
            logging.info(f"  Found {len(poor_indices)} texts poorly explained (max p(z|x) < {LOW_CONFIDENCE_THRESHOLD_ADD}). Attempting to form new clusters.")
            emb_poor = all_embeddings[poor_indices]
            num_new_clusters_to_add = max(1, len(poor_indices) // ITEMS_PER_NEW_CLUSTER_ADD)
            logging.info(f"  Attempting to create {num_new_clusters_to_add} new clusters from these texts.")
            from sklearn.cluster import KMeans
            if len(emb_poor) < num_new_clusters_to_add:
                 logging.warning(f"  Number of poorly explained texts ({len(emb_poor)}) is less than target new clusters ({num_new_clusters_to_add}). Adjusting to {len(emb_poor)} clusters.")
                 num_new_clusters_to_add = len(emb_poor)
            if num_new_clusters_to_add > 0:
                kmeans_poor = KMeans(n_clusters=num_new_clusters_to_add, n_init='auto', random_state=42)
                try:
                    new_labels_for_poor_texts = kmeans_poor.fit_predict(emb_poor)
                    for k_new_cluster in range(num_new_clusters_to_add):
                        new_cluster_original_indices = [poor_indices[i] for i, lab in enumerate(new_labels_for_poor_texts) if lab == k_new_cluster]
                        if new_cluster_original_indices:
                            assigned_new_id = next_new_cluster_id
                            next_new_cluster_id += 1
                            current_merged_df.loc[new_cluster_original_indices, 'cluster_id'] = assigned_new_id
                            current_merged_df.loc[new_cluster_original_indices, 'agglomerative_label'] = f"new_cluster_{assigned_new_id}"
                            added_clusters_info.append({'id': assigned_new_id, 'size': len(new_cluster_original_indices)})
                            logging.info(f"    Added new cluster {assigned_new_id} with {len(new_cluster_original_indices)} texts.")
                except Exception as e:
                    logging.error(f"  KMeans failed for adding new clusters from poorly explained texts: {e}")
            else:
                logging.info("  No new clusters to add based on poor text count or KMeans failure.")
        else:
            logging.info(f"  Found {len(poor_indices)} poorly explained texts (threshold: {MIN_POORLY_EXPLAINED_FOR_ADD}). Not adding new clusters based on this criterion.")
        iteration_actions['add'] = added_clusters_info
        logging.info(f"Iteration {iter_idx + 1}: 'Add' actions decided: {added_clusters_info}. Total actions for iteration: {iteration_actions}")
        schema_history.append(iteration_actions)
        if current_merged_df.empty:
            logging.warning(f"Iteration {iter_idx + 1}: Dataframe became empty after updates. Stopping EM refinement.")
            break
        unique_cluster_ids_after_iter = sorted(current_merged_df['cluster_id'].unique())
        if not unique_cluster_ids_after_iter:
            logging.warning(f"Iteration {iter_idx + 1}: No clusters remaining after updates. Stopping EM refinement.")
            break
        current_choices_list = []
        temp_label_map = {}
        for cid_val in unique_cluster_ids_after_iter:
            existing_label_series = current_merged_df[current_merged_df['cluster_id'] == cid_val]['agglomerative_label']
            if not existing_label_series.empty and pd.notna(existing_label_series.iloc[0]):
                label = existing_label_series.iloc[0]
            else:
                label = f"cluster_{cid_val}"
            current_choices_list.append(label)
            temp_label_map[cid_val] = label
        current_merged_df['agglomerative_label'] = current_merged_df['cluster_id'].map(temp_label_map)
        logging.info(f"Iteration {iter_idx + 1}: Re-initializing ProbabilityCalibrator with {len(current_choices_list)} choices: {current_choices_list}")
        current_prob_calibrator = initialize_probability_calibrator(
            model_identifier=model_identifier,
            model_type=model_type,
            choices=current_choices_list,
            num_trials=current_prob_calibrator.num_trials,
            scorer_type=current_prob_calibrator.scorer_type,
            verbose=verbose
        )
        logging.info(f"=== EM Iteration {iter_idx + 1} Complete. #Clusters: {len(unique_cluster_ids_after_iter)} ===")

        # Log P(z) for the current schema labels if verbose
        if verbose:
            logging.info(f"Iteration {iter_idx + 1} - Schema Label Priors (P(z)) after updates:")
            if not current_merged_df.empty and 'agglomerative_label' in current_merged_df.columns and current_merged_df['agglomerative_label'].notna().any():
                label_proportions = current_merged_df['agglomerative_label'].value_counts(normalize=True)
                for label, prob in label_proportions.items():
                    logging.info(f"  P('{label}') = {prob:.4f}")
            else:
                logging.info("  No data or agglomerative_labels to compute P(z) from current_merged_df.")

    final_iter_count = iter_idx + 1 if max_iters > 0 and 'iter_idx' in locals() and iter_idx is not None else 0
    logging.info(f"EM schema refinement finished after {final_iter_count} iterations.")
    return current_merged_df, schema_history, current_choices_list, current_prob_calibrator


def main(args):
    setup_logging()

    if args.model_type == "openai":
        raise ValueError(
            "The 'openai' model_type in this script relies on logprob scoring methods "
            "(via ChatCompletions with max_tokens=0 as implemented in assess_clusters.py) "
            "that are not suitable for reliably scoring full prompt likelihoods as needed here. "
            "Please use 'hf' or 'together' model types."
        )

    # 1. Construct paths
    experiment_base_dir = Path(args.experiment_dir)
    models_dir = experiment_base_dir / "models"
    agglomerative_output_dir = experiment_base_dir / "hierarchy_results"
    sentence_data_path = models_dir / "all_extracted_discourse_with_clusters.csv" # Input for examples_df
    agglomerative_labels_path = agglomerative_output_dir / "inner_node_labels.csv" # Input for initial_labels

    # 2. Determine target agglomerative clustering level/file
    _, _, selected_level_assignments_path = determine_target_clustering_info(
        agglomerative_output_dir,
        args.agglomerative_level,
        args.num_agglomerative_clusters
    )

    # 3. Load and prepare data for scoring
    merged_df, choices_list = load_and_prepare_data(
        sentence_data_path,
        selected_level_assignments_path,
        agglomerative_labels_path
    )

    # 3.5 Subsample data if requested
    if args.num_datapoints_per_cluster is not None:
        logging.info(f"Subsampling to {args.num_datapoints_per_cluster} datapoints per cluster.")
        merged_df = merged_df.groupby('cluster_id').sample(n=args.num_datapoints_per_cluster, replace=True)
        logging.info(f"  Number of datapoints after subsampling: {len(merged_df)}")


    # 4. Initialize ProbabilityCalibrator
    prob_calibrator = initialize_probability_calibrator(
        model_identifier=args.model_name,
        model_type=args.model_type,
        choices=choices_list,
        num_trials=args.num_trials,
        scorer_type=args.scorer_type,
        verbose=args.verbose
    )

    # 4.5 EM-style schema refinement (optional)
    if args.num_iterations > 0:
        merged_df, schema_history, choices_list, prob_calibrator = em_schema_refinement(
            merged_df,
            prob_calibrator,
            choices_list,
            text_column_name=args.sentence_column_name,
            model_identifier=args.model_name,
            model_type=args.model_type,
            max_iters=args.num_iterations,
            embedding_model_name=args.embedding_model_name,
            baseline_sample_size=args.baseline_sample_size,
            verbose=args.verbose,
            max_texts_per_cluster_metrics=args.max_texts_per_cluster_metrics
        )
        logging.info(f"EM schema refinement completed. Schema history: {schema_history}")

    # 5. Score Sentences
    if args.sentence_column_name not in merged_df.columns:
        raise KeyError(f"Specified sentence column '{args.sentence_column_name}' not found in the data.")
    
    results_df = score_dataframe_sentences(
        prob_calibrator,
        merged_df,
        args.sentence_column_name,
        choices_list
    )

    # 6. Save Results
    logging.info(f"Saving results to: {args.output_file}")
    output_path_obj = Path(args.output_file)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path_obj, index=False)
    logging.info(f"Processing complete. Output saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads agglomerative clustering outputs, scores sentences using ProbabilityCalibrator.")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment directory (e.g., experiments/editorial).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agglomerative_level", type=int, help="The level in the agglomerative hierarchy (1-indexed).")
    group.add_argument("--num_agglomerative_clusters", type=int, help="The desired number of agglomerative clusters.")

    parser.add_argument("--model_type", type=str, default="hf", choices=["hf", "openai", "together", "vllm"], help="Type of model to use (Hugging Face, OpenAI API, TogetherAI API).")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name/path of the model (HuggingFace ID, or API model name like 'gpt-4o-mini').")
    parser.add_argument("--sentence_column_name", type=str, default="sentences", help="Name of the column containing sentences/text to score in the input data.")
    parser.add_argument("--num_trials", type=int, default=3, help="Number of trials for ProbabilityCalibrator.")
    parser.add_argument("--scorer_type", type=str, default="batch", 
                        choices=["single", "batch"], 
                        help="Type of logprob scorer to use ('single' or 'batch'). Batch is usually more efficient if supported by backend.")
    parser.add_argument("--num_iterations", type=int, default=3, help="Number of iterations for EM-style schema refinement.")
    parser.add_argument("--embedding_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Name of the embedding model for EM-style schema refinement.")
    parser.add_argument("--num_datapoints_per_cluster", type=int, default=None, help="Number of datapoints to sample per cluster.")
    parser.add_argument("--baseline_sample_size", type=int, default=DEFAULT_BASELINE_SAMPLE_SIZE, help="Sample size for calculating baseline P(x) during EM refinement.")
    parser.add_argument("--max_texts_per_cluster_metrics", type=int, default=DEFAULT_MAX_TEXTS_PER_CLUSTER_METRICS, help="Max texts per cluster to use for metrics calculation (-1 for no limit).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of probabilities at each EM iteration.")

    args = parser.parse_args()
    main(args)


"""
Example command based on run_everything.sh outputs:

python src/run_em_algorithm.py \
    --experiment_dir "experiments/editorial" \
    --output_file "experiments/editorial/em_refined_scores.csv" \
    --num_agglomerative_clusters 10 \
    --model_type "together" \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" \
    --sentence_column_name "sentences" \
    --num_trials 3 \
    --scorer_type "batch" \
    --num_iterations 3 \
    --embedding_model_name "sentence-transformers/all-MiniLM-L6-v2" \
    --num_datapoints_per_cluster 50
"""

"""
Example command using vLLM (ensure you have a vLLM-compatible model and environment):

python src/run_em_algorithm.py \
    --experiment_dir "experiments/editorial" \
    --output_file "experiments/editorial/em_refined_scores_vllm.csv" \
    --num_agglomerative_clusters 10 \
    --model_type "vllm" \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --sentence_column_name "sentences" \
    --num_trials 3 \
    --scorer_type "batch" \
    --num_iterations 3 \
    --embedding_model_name "sentence-transformers/all-MiniLM-L6-v2" \
    --num_datapoints_per_cluster 50 \
    --verbose
"""