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
from dataclasses import dataclass

from em_scores import compute_document_level_scores, compute_corpus_level_scores, bayes_log_px_given_z
from em_diagnostics_utils import compute_posterior_diagnostics
from schema_utils import (
    SchemaState,
    build_schema_state_from_df,
    propose_label_from_texts,
    propose_label_and_keywords_from_texts,
    detect_near_duplicate_labels,
    sanitize_text,
    base_label,
    build_label_string,
)

# --- EM Refinement Constants ---
# For decide_schema_updates
SIMILARITY_THRESHOLD_MERGE = 0.8  # Cosine similarity for merging cluster centroids
MIN_CLUSTER_SIZE_REMOVE = 5     # Minimum number of items for a cluster to be kept
BASELINE_LL_FACTOR_REMOVE = 1.01 # Factor for L_z vs L_baseline for removal (L_z < L_baseline * FACTOR)
BASELINE_LL_FACTOR_SPLIT = 0.9   # Factor for L_z vs L_baseline for splitting (L_z < L_baseline * FACTOR)
CONFIDENCE_THRESHOLD_SPLIT = 0.4 # Minimum posterior confidence C(z) for splitting
# For em_schema_refinement (Add step)
LOW_CONFIDENCE_THRESHOLD_ADD = 0.2 # Max p(z|x) for a text to be considered poorly explained
MIN_POORLY_EXPLAINED_FOR_ADD = 5  # Min number of poorly explained texts to trigger new cluster formation
ITEMS_PER_NEW_CLUSTER_ADD = 50   # Heuristic: 1 new cluster per this many poorly explained items
MAX_NEW_CLUSTERS_PER_ITER = 3    # Hard cap to prevent avalanche
MAX_TOTAL_CLUSTERS = 200         # Hard cap on total clusters
MIN_DIVERSE_GROUPS_ADD = 2       # Require at least this many coherent groups before adding
MIN_GROUP_SIZE_ADD = 5           # Min size per new cluster group
ENTROPY_THRESHOLD_ADD = 1.0      # Require posterior entropy above this to add
DUP_LABEL_SIM_THRESHOLD = 0.90   # Near-duplicate label similarity threshold
ADD_COHESION_THRESHOLD = 0.20    # Min avg cosine similarity within new group


@dataclass
class SchemaRefinementConfig:
    enable_splits: bool = False
    max_splits_per_iter: int = 1
    min_cluster_size_for_split: int = 20
    split_L_margin: float = 8.0
    split_C_thresh: float = 0.20
    min_gap_median_for_split: float = 0.15
    cooldown_iters_after_split: int = 1

DEFAULT_BASELINE_SAMPLE_SIZE = 500 # Default sample size for baseline P(x) calculation
DEFAULT_MAX_TEXTS_PER_CLUSTER_METRICS = -1 # Default max texts per cluster for metrics; -1 means no limit

try:
    from .utils_probability_calibrator import ProbabilityCalibrator, initialize_probability_calibrator
except ImportError:
    try:
        from utils_probability_calibrator import ProbabilityCalibrator, initialize_probability_calibrator
    except ImportError:
        raise ImportError("Failed to import ProbabilityCalibrator and initialize_probability_calibrator. Please check your project structure and PYTHONPATH.")

def rebuild_calibrator_with_existing_backend(
    existing_calibrator: ProbabilityCalibrator,
    new_choices: list[str],
    verbose: bool = False,
) -> ProbabilityCalibrator:
    """
    Rebuild a calibrator with new choices while reusing the existing scorer functions.
    This avoids reloading the underlying model (important for vLLM GPU memory).
    """
    return ProbabilityCalibrator(
        choices=new_choices,
        logprob_scorer=existing_calibrator.logprob_scorer,
        full_logprob_fn=existing_calibrator.full_logprob_fn,
        num_trials=existing_calibrator.num_trials,
        content_free_input=existing_calibrator.content_free_input,
        alpha=existing_calibrator.alpha,
        verbose=verbose,
        batch_prompts=existing_calibrator.batch_prompts,
        batch_permutations=existing_calibrator.batch_permutations,
    )


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def build_unique_label_map(
    cluster_ids: list[int],
    raw_label_map: dict[int, str],
    fallback_prefix: str = "cluster",
) -> tuple[dict[int, str], list[str]]:
    """
    Ensure one unique textual label per cluster id.
    If duplicate labels occur, append a stable cluster-id suffix.
    """
    unique_label_map: dict[int, str] = {}
    seen_labels: dict[str, int] = {}
    choices: list[str] = []

    for cid in sorted(cluster_ids):
        base_label = sanitize_text(raw_label_map.get(cid, f"{fallback_prefix}_{cid}"))
        if not base_label:
            base_label = f"{fallback_prefix}_{cid}"
        if base_label in seen_labels:
            seen_labels[base_label] += 1
            unique_label = f"{base_label}__c{cid}"
            logging.warning(
                "Duplicate cluster label '%s' detected. Using unique label '%s' for cluster_id=%s.",
                base_label,
                unique_label,
                cid,
            )
        else:
            seen_labels[base_label] = 1
            unique_label = base_label
        unique_label_map[cid] = unique_label
        choices.append(build_label_string(unique_label))

    return unique_label_map, choices

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


def resolve_agglomerative_output_dir(experiment_base_dir: Path) -> Path:
    default_dir = experiment_base_dir / "hierarchy_results"
    if default_dir.exists():
        return default_dir
    standard_dir = experiment_base_dir / "hierarchy_results__standard"
    if standard_dir.exists():
        logging.warning(f"Using agglomerative output dir: {standard_dir} (default hierarchy_results not found).")
        return standard_dir
    candidates = sorted(experiment_base_dir.glob("hierarchy_results__*"))
    if candidates:
        logging.warning(f"Using agglomerative output dir: {candidates[0]} (default hierarchy_results not found).")
        return candidates[0]
    return default_dir


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

    # Ensure each active cluster has a label, then force uniqueness (required for categorical priors/calibrator choices).
    raw_label_map = {}
    for cid in unique_agg_cluster_ids_at_level:
        label = id_to_label_map.get(cid)
        if label is None:
            logging.warning(f"No textual label for agglomerative cluster_id {cid}. Using ID as label.")
            raw_label_map[cid] = str(cid)
        else:
            raw_label_map[cid] = str(label)

    unique_label_map, choices_list = build_unique_label_map(unique_agg_cluster_ids_at_level, raw_label_map, fallback_prefix="cluster")
    merged_df['agglomerative_label'] = merged_df['cluster_id'].map(unique_label_map)


    if not choices_list and unique_agg_cluster_ids_at_level: # only raise if there were clusters but no labels formed
        raise ValueError(f"Could not form a list of choices (textual labels) for the calibrator from {agglomerative_labels_path}.")
    if not unique_agg_cluster_ids_at_level:
        raise ValueError("No unique agglomerative cluster IDs found in the data for the selected level. Cannot proceed.")

    logging.info(f"Agglomerative cluster choices for ProbabilityCalibrator: {choices_list}")
    return merged_df, choices_list


def compute_sentence_embeddings(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64, show_progress: bool = False):
    """Returns a numpy array (len(texts), dim) of embeddings using SentenceTransformer."""
    from sentence_transformers import SentenceTransformer
    logging.info(f"Loading SentenceTransformer model: {model_name}")
    st_model = SentenceTransformer(model_name)
    embeddings = st_model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=show_progress)
    return embeddings


def score_dataframe_sentences(calibrator: ProbabilityCalibrator, data_df: pd.DataFrame, text_column: str, choices_list_for_output: list[str], show_progress: bool = False) -> pd.DataFrame:
    """Scores sentences in the DataFrame using the provided calibrator."""
    logging.info(f"Scoring sentences from column: '{text_column}'...")
    results_list = []

    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Scoring sentences", disable=not show_progress):
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
        max_texts_per_cluster_metrics: int = DEFAULT_MAX_TEXTS_PER_CLUSTER_METRICS,
        p_z_prior: np.ndarray | None = None
    ) -> dict:
    """Compute per-cluster metrics: average log-likelihood, posterior confidence, variance."""
    logging.info("Starting computation of cluster metrics...")
    cluster_metrics = {}
    all_assigned_log_likelihoods = []
    choice_to_idx = {c: i for i, c in enumerate(choices_list)}
    labels_as_str = merged_df['agglomerative_label'].astype(str)
    choices_str = [str(c) for c in choices_list]
    if p_z_prior is None:
        counts = pd.Categorical(labels_as_str, categories=choices_str).value_counts()
        p_z_prior = counts.to_numpy(dtype=float)
        p_z_prior = p_z_prior / p_z_prior.sum() if p_z_prior.sum() > 0 else np.ones(len(choices_str)) / len(choices_str)
    else:
        p_z_prior = np.array(p_z_prior, dtype=float)
        if p_z_prior.ndim != 1 or len(p_z_prior) != len(choices_list):
            logging.warning(
                "p_z_prior length/shape mismatch (got %s, expected %s). "
                "Recomputing prior from current labels.",
                p_z_prior.shape,
                (len(choices_list),),
            )
            counts = pd.Categorical(labels_as_str, categories=choices_str).value_counts()
            p_z_prior = counts.to_numpy(dtype=float)
            p_z_prior = p_z_prior / p_z_prior.sum() if p_z_prior.sum() > 0 else np.ones(len(choices_str)) / len(choices_str)

    # Determine the iterator for clusters, potentially with tqdm
    cluster_groups = merged_df.groupby('cluster_id')
    if verbose:
        cluster_iterator = tqdm(cluster_groups, desc="Computing cluster metrics", unit="cluster", disable=False)
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
        gap_vals = []
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
                p_z_given_x = prob_calibrator.calibrate_p_z_given_X(txt)
                if getattr(prob_calibrator, "full_logprob_fn", None) is None:
                    raise ValueError("full_logprob_fn is required to compute log p(x|z) for cluster metrics.")
                log_px = float(prob_calibrator.full_logprob_fn(str(txt)))
                log_px_given_z = bayes_log_px_given_z(np.array(p_z_given_x, dtype=float), log_px, p_z_prior)
                ll = float(log_px_given_z[label_idx])
                log_likelihoods.append(ll)
                all_assigned_log_likelihoods.append(ll)
                posterior_probs.append(float(p_z_given_x[label_idx]))
                # top1-top2 gap for split gating
                pzx = np.array(p_z_given_x, dtype=float)
                if pzx.size > 1:
                    order = np.sort(pzx)[::-1]
                    gap_vals.append(float(order[0] - order[1]))
            L_z = np.mean(log_likelihoods) if log_likelihoods else float('-inf')
            C_z = np.mean(posterior_probs) if posterior_probs else 0.0

        cluster_metrics[cluster_id] = {
            'L_z': L_z,
            'C_z': C_z,
            'V_z': V_z,
            'gap_median': float(np.median(gap_vals)) if gap_vals else 0.0,
            'size': len(indices), 
            'mapped_size': len(texts_for_metrics), # This now reflects sampled size if sampling occurred
            'embedding_centroid': e_z,
            'label': label
        }
        logging.debug(f"Cluster {cluster_id} ('{label}', size {len(indices)}, mapped_size {len(texts_for_metrics)}): L_z={L_z:.4f}, C_z={C_z:.4f}, V_z={V_z:.4f}")
    
    cluster_metrics['_baseline_log_px'] = baseline_log_px # cross-entropy style log p(x), legacy
    if all_assigned_log_likelihoods:
        cluster_metrics['_baseline_log_pxz_assigned'] = float(np.mean(all_assigned_log_likelihoods))
    else:
        cluster_metrics['_baseline_log_pxz_assigned'] = float('-inf')
    logging.info(f"Finished computation of cluster metrics. Baseline log p(x) = {baseline_log_px:.4f}")
    return cluster_metrics


def decide_schema_updates(
    cluster_metrics: dict,
    config: SchemaRefinementConfig,
    split_cooldown: dict[int, int],
    iter_idx: int,
    verbose: bool = False,
    log_top_pairs: int = 0,
):
    """Determine which clusters to split, merge, remove. Returns dictionaries describing actions."""
    logging.info("Starting schema update decisions...")
    actions = {'split': [], 'merge': [], 'remove': []}
    # Extract baseline and actual cluster data separately
    baseline_log_px = cluster_metrics.pop('_baseline_log_px', 0.0)
    baseline_log_pxz_assigned = cluster_metrics.pop('_baseline_log_pxz_assigned', baseline_log_px)
    decision_baseline = baseline_log_pxz_assigned if np.isfinite(baseline_log_pxz_assigned) else baseline_log_px
    
    # Filter out any non-cluster data that might have been passed in cluster_metrics keys
    valid_cluster_ids = [k for k, v in cluster_metrics.items() if isinstance(v, dict) and 'L_z' in v]
    if not valid_cluster_ids:
        logging.warning("No valid cluster data found in cluster_metrics for schema update decisions.")
        return actions

    median_V_z = np.median([cluster_metrics[cid]['V_z'] for cid in valid_cluster_ids if 'V_z' in cluster_metrics[cid]])
    logging.info(
        f"Using decision baseline L_baseline_assigned={decision_baseline:.4f} (and log p(x) baseline={baseline_log_px:.4f}), Median V_z={median_V_z:.4f} for decisions. "
        f"Thresholds: remove if size<{MIN_CLUSTER_SIZE_REMOVE} and L_z<{decision_baseline * BASELINE_LL_FACTOR_REMOVE:.4f}; "
        f"split if enabled (gated by size, L_z margin, C_z, gap_median, cooldown). "
        f"merge if cosine_sim>={SIMILARITY_THRESHOLD_MERGE:.2f}."
    )

    # Track margins to understand how close we are to thresholds
    split_margins = []
    remove_margins = []

    # --- SPLIT & REMOVE ---
    for cid in valid_cluster_ids:
        m = cluster_metrics[cid]
        # Per-cluster threshold diagnostics
        logging.info(
            f"Cluster {cid} ('{m.get('label','N/A')}'): "
            f"L_z={m['L_z']:.4f}, C_z={m['C_z']:.4f}, V_z={m['V_z']:.4f}, size={m['size']}. "
            f"remove_thresh={decision_baseline * BASELINE_LL_FACTOR_REMOVE:.4f}, "
            f"split_thresh={decision_baseline * BASELINE_LL_FACTOR_SPLIT:.4f}"
        )
        # Removal Check
        if m['size'] < MIN_CLUSTER_SIZE_REMOVE and m['L_z'] < decision_baseline * BASELINE_LL_FACTOR_REMOVE:
            actions['remove'].append(cid)
            logging.info(f"  Suggest REMOVE for cluster {cid} ('{m.get('label', 'N/A')}'): size {m['size']} < {MIN_CLUSTER_SIZE_REMOVE} AND L_z {m['L_z']:.4f} < baseline_thresh {decision_baseline * BASELINE_LL_FACTOR_REMOVE:.4f}.")
            continue # If removed, don't consider for split
        remove_margins.append(m['L_z'] - (decision_baseline * BASELINE_LL_FACTOR_REMOVE))
        # Split Check (gated)
        if config.enable_splits:
            cooldown_until = split_cooldown.get(cid)
            in_cooldown = cooldown_until is not None and iter_idx <= cooldown_until
            if (
                (not in_cooldown)
                and m['size'] >= config.min_cluster_size_for_split
                and m['L_z'] < (decision_baseline - config.split_L_margin)
                and m['C_z'] < config.split_C_thresh
                and m.get('gap_median', 0.0) >= config.min_gap_median_for_split
            ):
                actions['split'].append(cid)
                logging.info(
                    f"  Suggest SPLIT for cluster {cid} ('{m.get('label', 'N/A')}'): "
                    f"L_z {m['L_z']:.4f} < {decision_baseline - config.split_L_margin:.4f}, "
                    f"C_z {m['C_z']:.4f} < {config.split_C_thresh}, "
                    f"gap_median {m.get('gap_median', 0.0):.4f} >= {config.min_gap_median_for_split:.4f}."
                )
        split_margins.append(m['L_z'] - (decision_baseline * BASELINE_LL_FACTOR_SPLIT))

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
        top_pairs = []
        for i in range(n):
            if eligible_for_merge_ids[i] in merged_already:
                continue
            for j in range(i + 1, n):
                if eligible_for_merge_ids[j] in merged_already:
                    continue
                
                id_i = eligible_for_merge_ids[i]
                id_j = eligible_for_merge_ids[j]
                
                sim_val = sim_matrix[i, j]
                if log_top_pairs > 0:
                    top_pairs.append((sim_val, id_i, id_j))
                if sim_val >= SIMILARITY_THRESHOLD_MERGE:
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
                        logging.info(f"  Suggest MERGE for clusters {id_i} ('{m_i.get('label', 'N/A')}') and {id_j} ('{m_j.get('label', 'N/A')}'): Similarity {sim_val:.4f} >= {SIMILARITY_THRESHOLD_MERGE}, L_diff {diff_L:.4f}, C_diff {diff_C:.4f}.")
                        break # Merge id_i with one cluster, then move to next i
    
    # Deduplicate merge pairs (if (A,B) and (B,A) somehow got in)
    actions['merge'] = sorted(list(set(actions['merge'])))
    if verbose:
        if remove_margins:
            logging.info(f"Remove margin (L_z - remove_thresh) min/median/max: "
                         f"{np.min(remove_margins):.4f}/{np.median(remove_margins):.4f}/{np.max(remove_margins):.4f}")
        if split_margins:
            logging.info(f"Split margin (L_z - split_thresh) min/median/max: "
                         f"{np.min(split_margins):.4f}/{np.median(split_margins):.4f}/{np.max(split_margins):.4f}")
    if log_top_pairs > 0 and top_pairs:
        top_pairs.sort(reverse=True, key=lambda x: x[0])
        logging.info(f"Top-{log_top_pairs} merge candidate similarities:")
        for sim_val, id_i, id_j in top_pairs[:log_top_pairs]:
            m_i = cluster_metrics[id_i]
            m_j = cluster_metrics[id_j]
            diff_L = abs(m_i['L_z'] - m_j['L_z'])
            diff_C = abs(m_i['C_z'] - m_j['C_z'])
            logging.info(
                f"  pair ({id_i}, {id_j}) sim={sim_val:.4f} "
                f"L_diff={diff_L:.4f} C_diff={diff_C:.4f} "
                f"labels=('{m_i.get('label','N/A')}', '{m_j.get('label','N/A')}')"
            )
    logging.info(f"Finished schema update decisions. Actions: {actions}")
    if config.enable_splits and actions['split']:
        actions['split'] = sorted(actions['split'], key=lambda cid: cluster_metrics[cid]['L_z'])[: config.max_splits_per_iter]
    else:
        actions['split'] = []

    return actions


def _plot_histogram(values, title, xlabel, output_path, bins=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if values is None or len(values) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="#4C78A8", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _canonical_label_key(label: str) -> str:
    if label is None:
        return ""
    s = sanitize_text(label).lower()
    # drop suffixes like __c123
    if "__c" in s:
        s = s.split("__c")[0]
    # drop keywords after ":"
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    return s


def _label_cluster_vllm(texts: list[str], model_name: str) -> tuple[str, str, list[str]] | None:
    """
    Label a cluster using vLLM. Returns (name, description, keywords).
    """
    try:
        from utils_vllm_client import load_model as load_vllm_model
        from vllm import SamplingParams
    except Exception:
        return None


def _select_poor_candidates(
    pmax_list: list[float],
    entropy_list: list[float],
    assigned_rank_list: list[int | None],
    min_poorly_explained_for_add: int,
) -> list[int]:
    N = len(pmax_list)
    if N == 0:
        return []
    K = max(min_poorly_explained_for_add, int(0.05 * N))
    order = np.argsort(pmax_list)
    bottom_k = order[:K]
    selected = []
    for idx in bottom_k:
        ent = entropy_list[idx]
        rank = assigned_rank_list[idx]
        if ent >= ENTROPY_THRESHOLD_ADD and (rank is None or rank > 1):
            selected.append(idx)
    return selected

    examples = [t.replace("\n", " ").strip() for t in texts if t][:8]
    if not examples:
        return None
    prompt = (
        "Given these example sentences, propose a short label (2-5 words), "
        "a one-sentence description, and up to 6 keywords. "
        "Return JSON with keys: name, description, keywords.\n\n"
        "Examples:\n- " + "\n- ".join(examples)
    )

    sampling_params = SamplingParams(temperature=0.2, max_tokens=256)
    tokenizer, model = load_vllm_model(model_name)
    outputs = model.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text if outputs else ""
    import json
    try:
        data = json.loads(text)
        name = sanitize_text(data.get("name", ""))
        desc = sanitize_text(data.get("description", ""))
        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [sanitize_text(k) for k in keywords if sanitize_text(k)]
        return name, desc, keywords
    except Exception:
        return None


def _rename_duplicate_labels(
    df: pd.DataFrame,
    text_col: str,
    label_col: str = "agglomerative_label",
    cluster_col: str = "cluster_id",
    max_texts: int = 40,
) -> int:
    """Rename exact-duplicate labels at the CLUSTER level. Returns number of labels changed."""
    changed = 0
    cluster_labels = df.groupby(cluster_col)[label_col].first().astype(str)
    dup_labels = cluster_labels[cluster_labels.duplicated(keep=False)].unique().tolist()
    if not dup_labels:
        return 0

    for dup_label in dup_labels:
        dup_cluster_ids = cluster_labels[cluster_labels == dup_label].index.tolist()
        for cid in dup_cluster_ids:
            texts = df.loc[df[cluster_col] == cid, text_col].astype(str).tolist()
            if max_texts > 0 and len(texts) > max_texts:
                texts = texts[:max_texts]
            name, keywords = propose_label_and_keywords_from_texts(texts, parent_label=base_label(dup_label))
            new_label = build_label_string(name, keywords)
            prev_label = str(cluster_labels.loc[cid])
            if new_label != prev_label:
                df.loc[df[cluster_col] == cid, label_col] = new_label
                changed += 1
    return changed


def run_probability_diagnostics(
    prob_calibrator,
    merged_df: pd.DataFrame,
    text_column_name: str,
    choices_list: list[str],
    output_dir: Path,
    iteration_idx: int,
    sample_size: int = 200,
    bins: int = 40,
):
    """
    Sample probabilities and write histograms/CSV for calibration inspection.

    Metrics:
      - pmax_posterior: max_j p(z_j | x)
      - assigned_pzx: p(z_assigned | x) (posterior at the current assigned label)
      - pmax_lift: max softmax(log p(x|z)) which equals normalized p(z|x)/p(z)
    """
    if text_column_name not in merged_df.columns:
        logging.warning(f"Diagnostics skipped: text column '{text_column_name}' not found.")
        return {}

    if not choices_list:
        logging.warning("Diagnostics skipped: empty choices list.")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13 + iteration_idx)

    sample_size = min(sample_size, len(merged_df))
    sample_indices = rng.choice(merged_df.index.to_numpy(), size=sample_size, replace=False)
    sample_df = merged_df.loc[sample_indices]

    # p(z) prior: prefer calibrator.p_z if available, else from empirical cluster counts
    p_z_prior = getattr(prob_calibrator, "p_z", None)
    if p_z_prior is None or len(p_z_prior) != len(choices_list):
        if p_z_prior is not None:
            logging.warning(
                "Diagnostics: p_z_prior length mismatch (got %s, expected %s). Recomputing prior from labels.",
                len(p_z_prior),
                len(choices_list),
            )
        if "agglomerative_label" in merged_df.columns:
            counts = merged_df["agglomerative_label"].value_counts()
            p_z_prior = np.array([counts.get(c, 0) for c in choices_list], dtype=float)
            p_z_prior = p_z_prior / p_z_prior.sum() if p_z_prior.sum() > 0 else np.ones(len(choices_list)) / len(choices_list)
        else:
            p_z_prior = np.ones(len(choices_list)) / len(choices_list)

    label_to_idx = {str(lbl): i for i, lbl in enumerate(choices_list)}
    canonical_to_choice = {_canonical_label_key(str(lbl)): str(lbl) for lbl in choices_list}

    pmax_vals = []
    entropy_vals = []
    top_gap_vals = []
    pxz_max_vals = []
    assigned_pzx_vals = []
    log_px_vals = []

    rows = []
    for _, row in sample_df.iterrows():
        txt = row[text_column_name]
        pzx = np.array(prob_calibrator.calibrate_p_z_given_X(txt), dtype=float)
        if pzx.size == 0:
            continue
        pzx = np.clip(pzx, 1e-12, 1.0)
        pzx = pzx / pzx.sum()

        log_px = None
        if getattr(prob_calibrator, "full_logprob_fn", None) is not None:
            try:
                log_px = float(prob_calibrator.full_logprob_fn(str(txt)))
                log_px_vals.append(log_px)
            except Exception as exc:
                logging.warning(f"full_logprob_fn failed on diagnostics sample: {exc}")

        assigned_label_raw = row.get("agglomerative_label")
        assigned_label = assigned_label_raw
        assigned_in_choices_raw = str(assigned_label_raw) in label_to_idx if assigned_label_raw is not None else False
        assigned_label_canonical = _canonical_label_key(str(assigned_label_raw)) if assigned_label_raw is not None else ""
        assigned_in_choices_canonical = assigned_label_canonical in canonical_to_choice
        formatting_mismatch = False
        if not assigned_in_choices_raw and assigned_in_choices_canonical:
            assigned_label = canonical_to_choice[assigned_label_canonical]
            formatting_mismatch = True
        diag = compute_posterior_diagnostics(
            pzx=pzx,
            choices_list=choices_list,
            p_z_prior=p_z_prior,
            assigned_label=assigned_label,
        )

        pmax_vals.append(diag["pmax_posterior"])
        entropy_vals.append(diag["entropy"])
        top_gap_vals.append(diag["top1_top2_gap"])
        pxz_max_vals.append(diag["pmax_lift"])
        if np.isfinite(diag["assigned_pzx"]):
            assigned_pzx_vals.append(diag["assigned_pzx"])

        rows.append({
            "index": int(row.name),
            "cluster_id": row.get("cluster_id"),
            **diag,
            "assigned_label": assigned_label_raw,
            "assigned_label_used": assigned_label,
            "assigned_label_canonical": assigned_label_canonical,
            "assigned_in_choices_raw": assigned_in_choices_raw,
            "assigned_in_choices_canonical": assigned_in_choices_canonical,
            "formatting_mismatch": formatting_mismatch,
            "pred_label": diag.get("top1_label"),
            "log_px": log_px,
        })

    diag_df = pd.DataFrame(rows)
    diag_csv = output_dir / f"em_diag_iter_{iteration_idx:02d}.csv"
    diag_df.to_csv(diag_csv, index=False)
    logging.info(
        f"[Diagnostics] Iter {iteration_idx}: wrote {len(diag_df)} sampled rows to {diag_csv}"
    )
    if len(diag_df) > 0:
        logging.info(
            "[Diagnostics] Iter %s pmax p10/p50/p90=%.4f/%.4f/%.4f; "
            "gap p10/p50/p90=%.4f/%.4f/%.4f; entropy p10/p50/p90=%.4f/%.4f/%.4f",
            iteration_idx,
            np.percentile(diag_df["pmax_posterior"], 10),
            np.percentile(diag_df["pmax_posterior"], 50),
            np.percentile(diag_df["pmax_posterior"], 90),
            np.percentile(diag_df["top1_top2_gap"], 10),
            np.percentile(diag_df["top1_top2_gap"], 50),
            np.percentile(diag_df["top1_top2_gap"], 90),
            np.percentile(diag_df["entropy"], 10),
            np.percentile(diag_df["entropy"], 50),
            np.percentile(diag_df["entropy"], 90),
        )

    _plot_histogram(pmax_vals, f"p(z|x) max (iter {iteration_idx})", "pmax_posterior", output_dir / f"hist_pmax_iter_{iteration_idx:02d}.png", bins=bins)
    _plot_histogram(entropy_vals, f"p(z|x) entropy (iter {iteration_idx})", "entropy", output_dir / f"hist_entropy_iter_{iteration_idx:02d}.png", bins=bins)
    _plot_histogram(top_gap_vals, f"p(z|x) top1-top2 gap (iter {iteration_idx})", "gap", output_dir / f"hist_topgap_iter_{iteration_idx:02d}.png", bins=bins)
    _plot_histogram(pxz_max_vals, f"p(z|x)/p(z) max (iter {iteration_idx})", "pmax_lift", output_dir / f"hist_pxz_max_iter_{iteration_idx:02d}.png", bins=bins)
    _plot_histogram(assigned_pzx_vals, f"p(z|x) assigned label (iter {iteration_idx})", "assigned_pzx", output_dir / f"hist_assigned_pzx_iter_{iteration_idx:02d}.png", bins=bins)
    if log_px_vals:
        _plot_histogram(log_px_vals, f"log p(x) (iter {iteration_idx})", "log_px", output_dir / f"hist_log_px_iter_{iteration_idx:02d}.png", bins=bins)
    _plot_histogram(p_z_prior, f"p(z) prior (iter {iteration_idx})", "p_z", output_dir / f"hist_pz_prior_iter_{iteration_idx:02d}.png", bins=min(bins, len(p_z_prior)))

    # Posterior sanity per label (sample up to 10 assigned items per label)
    sanity_rows = []
    for label in choices_list:
        assigned_rows = merged_df[merged_df["agglomerative_label"].astype(str) == str(label)]
        if assigned_rows.empty:
            continue
        sample_assigned = assigned_rows.sample(n=min(10, len(assigned_rows)), random_state=42)
        ranks = []
        pzs = []
        gaps = []
        for _, r in sample_assigned.iterrows():
            pzx = np.array(prob_calibrator.calibrate_p_z_given_X(r[text_column_name]), dtype=float)
            if pzx.size == 0:
                continue
            diag = compute_posterior_diagnostics(
                pzx=pzx,
                choices_list=choices_list,
                p_z_prior=p_z_prior,
                assigned_label=str(label),
            )
            ranks.append(diag["assigned_rank"] if diag["assigned_rank"] is not None else np.nan)
            pzs.append(diag["assigned_pzx"])
            gaps.append(diag["top1_top2_gap"])
        if pzs:
            sanity_rows.append({
                "label": label,
                "n_samples": len(pzs),
                "fraction_top1": float(np.mean([r == 1 for r in ranks if not np.isnan(r)])) if ranks else np.nan,
                "avg_rank": float(np.nanmean(ranks)) if ranks else np.nan,
                "avg_p_assigned": float(np.mean(pzs)),
                "avg_gap": float(np.mean(gaps)),
            })

    if sanity_rows:
        sanity_df = pd.DataFrame(sanity_rows)
        sanity_csv = output_dir / f"posterior_sanity_iter_{iteration_idx:02d}.csv"
        sanity_df.to_csv(sanity_csv, index=False)

    mismatch_rate = None
    if "is_assigned_top1" in diag_df.columns:
        valid = diag_df["is_assigned_top1"].dropna()
        if len(valid) > 0:
            mismatch_rate = float((valid == False).mean())

    return {
        "mismatch_rate": mismatch_rate,
        "pmax_median": float(np.median(pmax_vals)) if pmax_vals else None,
        "entropy_median": float(np.median(entropy_vals)) if entropy_vals else None,
    }


def apply_schema_updates(
    merged_df: pd.DataFrame,
    embeddings: np.ndarray,
    idx_to_emb_pos_map: dict,
    actions: dict,
    next_new_cluster_id: int,
) -> tuple[pd.DataFrame, int, list[tuple[int, int]]]:
    """Apply split, merge, remove actions to merged_df and return updated df, next available cluster id, and split pairs."""
    from sklearn.cluster import KMeans
    pre_clusters = merged_df['cluster_id'].nunique()
    logging.info(f"Starting to apply schema updates. Initial #clusters: {pre_clusters}")
    current_df = merged_df.copy() # Work on a copy
    split_pairs: list[tuple[int, int]] = []

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

    count_before_split = current_df['cluster_id'].nunique()
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

            valid_subset_indices = []
            emb_positions = []
            for idx in subset_indices:
                if idx in idx_to_emb_pos_map:
                    valid_subset_indices.append(idx)
                    emb_positions.append(idx_to_emb_pos_map[idx])
            if len(emb_positions) < 2:
                logging.warning(f"Cluster {cid_to_split} has < 2 valid embeddings. Cannot split.")
                continue

            emb_subset = embeddings[emb_positions]
            kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
            try:
                labels_split = kmeans.fit_predict(emb_subset)
            except Exception as e:
                logging.error(f"KMeans failed for splitting cluster {cid_to_split}: {e}. Skipping split.")
                continue
            new_split_id = next_new_cluster_id
            next_new_cluster_id += 1
            split_assignment_map = {}
            for i, original_df_idx in enumerate(valid_subset_indices):
                if labels_split[i] == 0:
                    split_assignment_map[original_df_idx] = cid_to_split
                else:
                    split_assignment_map[original_df_idx] = new_split_id
            for original_df_idx, new_cluster_assignment in split_assignment_map.items():
                current_df.loc[original_df_idx, 'cluster_id'] = new_cluster_assignment
            split_pairs.append((cid_to_split, new_split_id))
            logging.info(f"  Split cluster {cid_to_split} into {cid_to_split} and {new_split_id}. New #clusters: {current_df['cluster_id'].nunique()}")

    post_clusters = current_df['cluster_id'].nunique()
    logging.info(f"Finished applying schema updates. Final #clusters: {post_clusters}")
    if actions.get('split'):
        expected = count_before_split + len(actions['split'])
        assert post_clusters == expected, f"Split count mismatch: expected {expected}, got {post_clusters}"
    return current_df, next_new_cluster_id, split_pairs


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
        max_texts_per_cluster_metrics: int = DEFAULT_MAX_TEXTS_PER_CLUSTER_METRICS,
        show_progress: bool = False,
        log_iteration_metrics: bool = False,
        log_top_pairs: int = 0,
        diagnostics_sample_size: int = 200,
        diagnostics_bins: int = 40,
        diagnostics_dir: Path | None = None,
        low_confidence_threshold_add: float = LOW_CONFIDENCE_THRESHOLD_ADD,
        min_poorly_explained_for_add: int = MIN_POORLY_EXPLAINED_FOR_ADD,
        items_per_new_cluster_add: int = ITEMS_PER_NEW_CLUSTER_ADD,
        config: SchemaRefinementConfig | None = None,
): 
    """Run an EM-like schema refinement loop, returning updated merged_df and history of schema changes."""
    if config is None:
        config = SchemaRefinementConfig()
    logging.info(f"Starting EM-style schema refinement for {max_iters} iterations...")
    logging.info(f"Computing initial sentence embeddings using {embedding_model_name}...")
    
    # Create a mapping from the original DataFrame index to 0-based positions for the embeddings array
    # This assumes current_merged_df's index at this point is the reference for all_embeddings
    idx_to_emb_pos_map = {original_idx: pos for pos, original_idx in enumerate(merged_df.index)}
    all_embeddings = compute_sentence_embeddings(
        merged_df[text_column_name].tolist(),
        model_name=embedding_model_name,
        show_progress=show_progress,
    )

    current_merged_df = merged_df.copy()
    schema_state = build_schema_state_from_df(current_merged_df)
    current_merged_df['agglomerative_label'] = current_merged_df['cluster_id'].map(schema_state.label_map)
    current_choices_list = list(schema_state.choices_list)
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
            p_z_prior=schema_state.p_z_prior,
        )
        # Persist per-row pmax / assignments for analysis
        current_merged_df["pmax_explained"] = doc_scores["row_pmax_posterior"]     
        current_merged_df["z_hat_explainer"] = doc_scores["row_z_hat"]   

        # Optionally compute corpus scores (treat all as test or pass a mask)
        corpus_scores = compute_corpus_level_scores(
            log_px_given_z_hat=doc_scores["row_log_px_given_z"][np.arange(len(current_merged_df)),
                                                                    current_merged_df["z_hat_explainer"].values],
            token_counts=None,             # plug true token counts if you track them
            k_complexity=len(current_choices_list),  # simple complexity proxy
            is_test_mask=None,
            q_ij=doc_scores["PZX"],
            log_px_given_z=doc_scores["row_log_px_given_z"],
            log_pz=np.log(np.clip(doc_scores["pz_prior"], 1e-30, None)),
        )  

        logging.info(f"[E-step] L_baseline={doc_scores['L_baseline']:.4f}, "
                    f"logL_total={corpus_scores['logL_cond_total']:.2f}, "
                    f"AIC={corpus_scores['AIC']}, BIC={corpus_scores['BIC']}, "
                    f"PPL={corpus_scores['perplexity']:.3f}, "
                    f"ELBO={corpus_scores['ELBO']}")  
    else:
        logging.warning(f"  Dataset is empty or too small for baseline sample, cannot compute baseline log p(x). Using {baseline_log_px}.")

    split_cooldown: dict[int, int] = {}
    add_cooldown_until = 0
    for iter_idx in range(max_iters):
        logging.info(f"=== EM Iteration {iter_idx + 1}/{max_iters} ===")
        iteration_actions = {}
        dup_renamed = 0
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
            max_texts_per_cluster_metrics=max_texts_per_cluster_metrics,
            p_z_prior=schema_state.p_z_prior
        )
        # Diagnostics (probability calibration)
        diag_summary = None
        if diagnostics_dir is not None:
            diag_summary = run_probability_diagnostics(
                current_prob_calibrator,
                current_merged_df,
                text_column_name,
                current_choices_list,
                diagnostics_dir,
                iter_idx + 1,
                sample_size=diagnostics_sample_size,
                bins=diagnostics_bins,
            )
            if diag_summary and diag_summary.get("mismatch_rate") is not None:
                logging.info(
                    "Diagnostics summary (iter %s): mismatch_rate=%.3f, pmax_median=%.3f, entropy_median=%.3f",
                    iter_idx + 1,
                    diag_summary["mismatch_rate"],
                    diag_summary.get("pmax_median", float("nan")),
                    diag_summary.get("entropy_median", float("nan")),
                )

        actions = decide_schema_updates(cluster_metrics, config, split_cooldown, iter_idx + 1, verbose=verbose, log_top_pairs=log_top_pairs)
        iteration_actions.update(actions)
        logging.info(f"Iteration {iter_idx + 1}: Actions decided from metrics (Split/Merge/Remove): {actions}. 'Add' actions to be determined next.")
        current_merged_df, next_new_cluster_id, split_pairs = apply_schema_updates(
            current_merged_df,
            all_embeddings,
            idx_to_emb_pos_map,
            actions,
            next_new_cluster_id,
        )
        if split_pairs:
            for parent_id, _ in split_pairs:
                split_cooldown[parent_id] = (iter_idx + 1) + config.cooldown_iters_after_split

        # Rename split children (and parents) with meaningful labels based on exemplar texts
        split_child_ids = {child for _, child in split_pairs}
        if split_pairs and config.enable_splits:
            for parent_id, child_id in split_pairs:
                parent_label = schema_state.label_map.get(parent_id, f"cluster_{parent_id}")
                parent_texts = current_merged_df.loc[current_merged_df["cluster_id"] == parent_id, text_column_name].astype(str).tolist()
                child_texts = current_merged_df.loc[current_merged_df["cluster_id"] == child_id, text_column_name].astype(str).tolist()
                parent_base = base_label(parent_label)

                new_parent_label = None
                new_child_label = None
                if config.enable_splits and model_type == "vllm":
                    parent_llm = _label_cluster_vllm(parent_texts, model_identifier)
                    child_llm = _label_cluster_vllm(child_texts, model_identifier)
                    if parent_llm:
                        name, _, keywords = parent_llm
                        new_parent_label = build_label_string(name or parent_base, keywords)
                    if child_llm:
                        name, _, keywords = child_llm
                        new_child_label = build_label_string(name or parent_base, keywords)

                if not new_parent_label:
                    name, keywords = propose_label_and_keywords_from_texts(parent_texts, parent_label=parent_base)
                    new_parent_label = build_label_string(name, keywords)
                if not new_child_label:
                    name, keywords = propose_label_and_keywords_from_texts(child_texts, parent_label=parent_base)
                    new_child_label = build_label_string(name, keywords)
                if new_child_label == new_parent_label:
                    new_child_label = build_label_string(f"{new_child_label} B")

                current_merged_df.loc[current_merged_df["cluster_id"] == parent_id, "agglomerative_label"] = new_parent_label
                current_merged_df.loc[current_merged_df["cluster_id"] == child_id, "agglomerative_label"] = new_child_label

        # Rename any exact duplicates to avoid suffixes like __cXYZ
        dup_renamed = _rename_duplicate_labels(current_merged_df, text_column_name)

        # Near-duplicate guardrail: revert split children if labels are too similar
        duplicates_prevented = False
        if split_pairs:
            labels_now = current_merged_df.groupby("cluster_id")["agglomerative_label"].first().astype(str).tolist()
            dup_pairs = detect_near_duplicate_labels(labels_now, threshold=DUP_LABEL_SIM_THRESHOLD)
            if dup_pairs:
                duplicates_prevented = True
                logging.warning(
                    "Near-duplicate labels detected after split; reverting split children: %s",
                    split_child_ids,
                )
                for parent_id, child_id in split_pairs:
                    current_merged_df.loc[current_merged_df["cluster_id"] == child_id, "cluster_id"] = parent_id
                # Rebuild labels after reverting
                current_merged_df["agglomerative_label"] = current_merged_df["cluster_id"].map(
                    current_merged_df.groupby("cluster_id")["agglomerative_label"].first().to_dict()
                )
                split_child_ids = set()
        
        # Log P(z|X) for each datapoint if verbose
        if verbose:
            logging.info(f"Iteration {iter_idx + 1} - Datapoint Posterior Probabilities (P(z|X)) for {current_merged_df['cluster_id'].nunique()} clusters:")
            if not current_merged_df.empty and current_choices_list:
                # Create a temporary calibrator with the current choices for this logging section if it's different
                # This ensures probabilities are logged against the correct set of choices for the current iteration state
                temp_calibrator_for_logging = current_prob_calibrator 
                # Check if current_prob_calibrator's choices match current_choices_list, if not, reinitialize (should generally match)
                if list(temp_calibrator_for_logging.choices) != list(current_choices_list):
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
        add_prev_cluster_ids = None
        schema_unstable = bool(actions.get("split") or actions.get("merge") or actions.get("remove") or duplicates_prevented)
        added_clusters_info = []
        if schema_unstable:
            logging.info("  Skipping add-step due to schema instability (splits/merges/removes or duplicate guardrail).")
        elif iter_idx + 1 <= add_cooldown_until:
            logging.info("  Skipping add-step due to cooldown (last add in iter %s).", add_cooldown_until - 1)
        else:
            poor_indices = []
            pmax_vals = []
            entropy_vals = []
            assigned_ranks = []
            all_indices = []
            for original_idx in current_merged_df.index:
                sentence_text = current_merged_df.loc[original_idx, text_column_name]
                probabilities = current_prob_calibrator.calibrate_p_z_given_X(sentence_text)
                pmax = float(np.max(probabilities)) if hasattr(probabilities, "__len__") and len(probabilities) > 0 else 0.0
                pmax_vals.append(pmax)
                all_indices.append(original_idx)

                diag = compute_posterior_diagnostics(
                    pzx=np.asarray(probabilities, dtype=float),
                    choices_list=current_choices_list,
                    p_z_prior=schema_state.p_z_prior if schema_state.p_z_prior is not None else np.ones(len(current_choices_list)) / max(1, len(current_choices_list)),
                    assigned_label=current_merged_df.loc[original_idx, "agglomerative_label"],
                )
                entropy_vals.append(diag["entropy"])
                assigned_ranks.append(diag["assigned_rank"])

            if pmax_vals:
                logging.info(
                    f"  pmax stats vs add-threshold {low_confidence_threshold_add}: "
                    f"min/median/max={np.min(pmax_vals):.4f}/{np.median(pmax_vals):.4f}/{np.max(pmax_vals):.4f}"
                )

            poor_pos = _select_poor_candidates(
                pmax_vals,
                entropy_vals,
                assigned_ranks,
                min_poorly_explained_for_add,
            )
            poor_indices = [all_indices[i] for i in poor_pos]
            logging.info(
                "  Add-step candidates: bottom-K=%s, filtered_poor=%s",
                max(min_poorly_explained_for_add, int(0.05 * len(current_merged_df))),
                len(poor_indices),
            )

            if len(current_choices_list) >= MAX_TOTAL_CLUSTERS:
                logging.info("  Max total clusters reached; skipping add-step.")
                poor_indices = []

            if len(poor_indices) >= min_poorly_explained_for_add:
                logging.info(f"  Found {len(poor_indices)} texts poorly explained. Attempting to form new clusters.")
                valid_poor_indices = []
                poor_emb_positions = []
                for idx in poor_indices:
                    if idx in idx_to_emb_pos_map:
                        valid_poor_indices.append(idx)
                        poor_emb_positions.append(idx_to_emb_pos_map[idx])
                    else:
                        logging.warning(f"  Original index {idx} not found in idx_to_emb_pos_map. Skipping.")

                if not poor_emb_positions:
                    logging.warning("  No valid embeddings for poorly explained texts. Skipping add step.")
                else:
                    prev_cluster_ids = current_merged_df.loc[valid_poor_indices, "cluster_id"].copy()
                    add_prev_cluster_ids = prev_cluster_ids
                    emb_poor = all_embeddings[poor_emb_positions]
                    num_new_clusters_to_add = max(1, len(valid_poor_indices) // items_per_new_cluster_add)
                    num_new_clusters_to_add = min(num_new_clusters_to_add, MAX_NEW_CLUSTERS_PER_ITER)
                    logging.info(f"  Attempting to create {num_new_clusters_to_add} new clusters from these texts.")
                    from sklearn.cluster import KMeans
                    if len(emb_poor) < num_new_clusters_to_add:
                        logging.warning(
                            f"  Number of poorly explained texts ({len(emb_poor)}) is less than target new clusters ({num_new_clusters_to_add}). Adjusting to {len(emb_poor)} clusters."
                        )
                        num_new_clusters_to_add = len(emb_poor)
                    try:
                        from sklearn.metrics.pairwise import cosine_similarity
                        created = 0
                        for k_try in [min(2, num_new_clusters_to_add), min(3, num_new_clusters_to_add)]:
                            if k_try < 2:
                                continue
                            kmeans_poor = KMeans(n_clusters=k_try, n_init='auto', random_state=42)
                            new_labels_for_poor_texts = kmeans_poor.fit_predict(emb_poor)
                            for k_new_cluster in range(k_try):
                                new_cluster_original_indices = [valid_poor_indices[i] for i, lab in enumerate(new_labels_for_poor_texts) if lab == k_new_cluster]
                                if len(new_cluster_original_indices) < MIN_GROUP_SIZE_ADD:
                                    continue
                                E = all_embeddings[[idx_to_emb_pos_map[i] for i in new_cluster_original_indices]]
                                centroid = E.mean(axis=0, keepdims=True)
                                sims = cosine_similarity(E, centroid).flatten()
                                if float(np.mean(sims)) < ADD_COHESION_THRESHOLD:
                                    continue
                                assigned_new_id = next_new_cluster_id
                                next_new_cluster_id += 1
                                current_merged_df.loc[new_cluster_original_indices, 'cluster_id'] = assigned_new_id
                                texts = current_merged_df.loc[new_cluster_original_indices, text_column_name].astype(str).tolist()
                                name, keywords = propose_label_and_keywords_from_texts(texts, parent_label=None)
                                new_label = build_label_string(name, keywords)
                                current_merged_df.loc[new_cluster_original_indices, 'agglomerative_label'] = new_label
                                added_clusters_info.append({'id': assigned_new_id, 'size': len(new_cluster_original_indices)})
                                logging.info(f"    Added new cluster {assigned_new_id} with {len(new_cluster_original_indices)} texts.")
                                created += 1
                                if created >= MAX_NEW_CLUSTERS_PER_ITER:
                                    break
                            if created > 0:
                                break
                        if created == 0:
                            logging.info("  Poorly explained set not diverse enough; skipping add-step.")
                            current_merged_df.loc[valid_poor_indices, "cluster_id"] = prev_cluster_ids.values
                            added_clusters_info = []
                    except Exception as e:
                        logging.error(f"  KMeans failed for adding new clusters from poorly explained texts: {e}")
            else:
                logging.info(f"  Found {len(poor_indices)} poorly explained texts (threshold: {min_poorly_explained_for_add}). Not adding new clusters based on this criterion.")
        # Guardrail: reject added clusters if they are near-duplicate labels
        if added_clusters_info:
            labels_now = current_merged_df.groupby("cluster_id")["agglomerative_label"].first().astype(str).tolist()
            dup_pairs = detect_near_duplicate_labels(labels_now, threshold=DUP_LABEL_SIM_THRESHOLD)
            if dup_pairs:
                logging.warning("Near-duplicate labels detected after add-step; reverting added clusters.")
                if add_prev_cluster_ids is not None:
                    current_merged_df.loc[add_prev_cluster_ids.index, "cluster_id"] = add_prev_cluster_ids.values
                added_clusters_info = []
            else:
                add_cooldown_until = (iter_idx + 1) + 1

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

        # Ensure no exact-duplicate labels remain.
        dup_renamed += _rename_duplicate_labels(current_merged_df, text_column_name)

        schema_state = build_schema_state_from_df(current_merged_df)
        current_merged_df['agglomerative_label'] = current_merged_df['cluster_id'].map(schema_state.label_map)
        current_choices_list = list(schema_state.choices_list)

        # IMPORTANT: keep calibrator choices aligned with current schema before any metric recomputation.
        if list(current_prob_calibrator.choices) != list(current_choices_list):
            logging.info(f"Iteration {iter_idx + 1}: Re-initializing ProbabilityCalibrator with {len(current_choices_list)} choices: {current_choices_list}")
            try:
                current_prob_calibrator = rebuild_calibrator_with_existing_backend(
                    current_prob_calibrator,
                    current_choices_list,
                    verbose=verbose,
                )
                logging.info("Iteration %s: Reused existing backend/scorer to refresh calibrator choices.", iter_idx + 1)
            except Exception as e:
                logging.warning(
                    f"Iteration {iter_idx + 1}: In-place calibrator rebuild failed ({e}). "
                    "Falling back to full re-initialization."
                )
                current_prob_calibrator = initialize_probability_calibrator(
                    model_identifier=model_identifier,
                    model_type=model_type,
                    choices=current_choices_list,
                    num_trials=current_prob_calibrator.num_trials,
                    scorer_type="batch" if current_prob_calibrator.batch_prompts is False else "single",
                    verbose=verbose
                )
        else:
            logging.info(f"Iteration {iter_idx + 1}: Choices unchanged; reusing existing ProbabilityCalibrator.")

        # Keep calibrator prior aligned with current schema
        if schema_state.p_z_prior is not None:
            current_prob_calibrator.p_z = schema_state.p_z_prior

        logging.info(
            "Schema diff (iter %s): #labels=%s, renamed_duplicates=%s, adds=%s, splits=%s, merges=%s, removes=%s",
            iter_idx + 1,
            len(current_choices_list),
            dup_renamed,
            len(iteration_actions.get("add", [])),
            len(iteration_actions.get("split", [])),
            len(iteration_actions.get("merge", [])),
            len(iteration_actions.get("remove", [])),
        )

        if log_iteration_metrics:
            doc_scores_iter = compute_document_level_scores(
                df=current_merged_df,
                text_col=text_column_name,
                cluster_col="agglomerative_label",
                choices=current_choices_list,
                prob_calibrator=current_prob_calibrator,
                embeddings=all_embeddings if 'all_embeddings' in locals() else None,
                p_z_prior=getattr(current_prob_calibrator, "p_z", None),
            )
            corpus_scores_iter = compute_corpus_level_scores(
                log_px_given_z_hat=doc_scores_iter["row_log_px_given_z"][np.arange(len(current_merged_df)),
                                                                            doc_scores_iter["row_z_hat"]],
                token_counts=None,
                k_complexity=len(current_choices_list),
                is_test_mask=None,
                q_ij=doc_scores_iter["PZX"],
                log_px_given_z=doc_scores_iter["row_log_px_given_z"],
                log_pz=np.log(np.clip(doc_scores_iter["pz_prior"], 1e-30, None)),
            )
            logging.info(
                f"[Iter {iter_idx + 1}] L_baseline={doc_scores_iter['L_baseline']:.4f}, "
                f"logL_total={corpus_scores_iter['logL_cond_total']:.2f}, "
                f"AIC={corpus_scores_iter['AIC']}, BIC={corpus_scores_iter['BIC']}, "
                f"PPL={corpus_scores_iter['perplexity']:.3f}, "
                f"ELBO={corpus_scores_iter['ELBO']}"
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
    agglomerative_output_dir = resolve_agglomerative_output_dir(experiment_base_dir)
    if args.sentence_data_path:
        sentence_data_path = Path(args.sentence_data_path)
    else:
        sentence_data_path = models_dir / "all_extracted_discourse_with_clusters_and_text.csv"
        if not sentence_data_path.exists():
            sentence_data_path = models_dir / "all_extracted_discourse_with_clusters.csv"
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
        # Reset index so embeddings align with row positions after subsampling.
        merged_df = merged_df.reset_index(drop=True)
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
        diagnostics_dir = Path(args.diagnostics_dir) if args.diagnostics_dir else Path(args.experiment_dir) / "em_diagnostics"
        config = SchemaRefinementConfig(
            enable_splits=args.enable_splits,
            max_splits_per_iter=args.max_splits_per_iter,
            min_cluster_size_for_split=args.min_cluster_size_for_split,
            split_L_margin=args.split_L_margin,
            split_C_thresh=args.split_C_thresh,
            min_gap_median_for_split=args.min_gap_median_for_split,
            cooldown_iters_after_split=args.cooldown_iters_after_split,
        )
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
            max_texts_per_cluster_metrics=args.max_texts_per_cluster_metrics,
            show_progress=args.show_progress,
            log_iteration_metrics=args.log_iteration_metrics,
            log_top_pairs=args.log_top_pairs,
            diagnostics_sample_size=args.diagnostics_sample_size,
            diagnostics_bins=args.diagnostics_bins,
            diagnostics_dir=diagnostics_dir,
            low_confidence_threshold_add=args.low_confidence_threshold_add,
            min_poorly_explained_for_add=args.min_poorly_explained_for_add,
            items_per_new_cluster_add=args.items_per_new_cluster_add,
            config=config,
        )
        logging.info(f"EM schema refinement completed. Schema history: {schema_history}")

    # 5. Score Sentences
    if args.sentence_column_name not in merged_df.columns:
        raise KeyError(f"Specified sentence column '{args.sentence_column_name}' not found in the data.")

    if args.log_sentence_samples > 0:
        sample_n = min(args.log_sentence_samples, len(merged_df))
        logging.info(
            "Logging %s sample sentences from column '%s' (showing first %s chars each).",
            sample_n,
            args.sentence_column_name,
            args.log_sentence_sample_chars,
        )
        for i, txt in enumerate(merged_df[args.sentence_column_name].astype(str).head(sample_n).tolist(), 1):
            logging.info("  [Sample %s] %s", i, txt[:args.log_sentence_sample_chars])
    
    results_df = score_dataframe_sentences(
        prob_calibrator,
        merged_df,
        args.sentence_column_name,
        choices_list,
        show_progress=args.show_progress,
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
    parser.add_argument("--sentence_data_path", type=str, default=None, help="Override path to the sentence data CSV (defaults to models/all_extracted_discourse_with_clusters*.csv).")
    parser.add_argument("--log_sentence_samples", type=int, default=0, help="Log N sample sentences from the input data after loading (default: 0).")
    parser.add_argument("--log_sentence_sample_chars", type=int, default=200, help="Max chars per logged sample sentence (default: 200).")
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
    parser.add_argument("--show_progress", action="store_true", help="Show tqdm progress bars.")
    parser.add_argument("--log_iteration_metrics", action="store_true", help="Log ELBO/AIC/BIC per iteration.")
    parser.add_argument("--log_top_pairs", type=int, default=0, help="Log top-N merge candidate similarities per iteration.")
    parser.add_argument("--diagnostics_sample_size", type=int, default=200, help="Number of rows to sample for probability diagnostics per iteration.")
    parser.add_argument("--diagnostics_bins", type=int, default=40, help="Histogram bins for diagnostics plots.")
    parser.add_argument("--diagnostics_dir", type=str, default="", help="Output directory for EM diagnostics (defaults to <experiment_dir>/em_diagnostics).")
    parser.add_argument("--low_confidence_threshold_add", type=float, default=LOW_CONFIDENCE_THRESHOLD_ADD, help="Add-step threshold on max p(z|x). Higher values mark more rows as poorly explained.")
    parser.add_argument("--min_poorly_explained_for_add", type=int, default=MIN_POORLY_EXPLAINED_FOR_ADD, help="Minimum poorly explained rows needed before creating new clusters.")
    parser.add_argument("--items_per_new_cluster_add", type=int, default=ITEMS_PER_NEW_CLUSTER_ADD, help="Heuristic scale: one new cluster per this many poorly explained rows.")
    # Split gating config
    parser.add_argument("--enable_splits", action="store_true", help="Enable split operations (default: disabled).")
    parser.add_argument("--max_splits_per_iter", type=int, default=1)
    parser.add_argument("--min_cluster_size_for_split", type=int, default=20)
    parser.add_argument("--split_L_margin", type=float, default=8.0)
    parser.add_argument("--split_C_thresh", type=float, default=0.20)
    parser.add_argument("--min_gap_median_for_split", type=float, default=0.15)
    parser.add_argument("--cooldown_iters_after_split", type=int, default=1)

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
