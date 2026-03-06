import argparse
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import re
from pathlib import Path
import sys
from tqdm import tqdm
from dataclasses import dataclass

from em_scores import (
    compute_document_level_scores,
    compute_corpus_level_scores_variants,
    bayes_log_px_given_z,
)
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
VALID_TUNE_OPERATIONS = ("all", "split", "merge", "remove", "add", "revise")
VALID_ACCEPT_METRICS = ("assigned", "soft", "oracle")
VALID_STRUCTURAL_LABEL_MODES = ("semantic", "parent_relative")
DUP_LABEL_SIM_THRESHOLD = 0.90   # Near-duplicate label similarity threshold


@dataclass
class SchemaRefinementConfig:
    tune_operation: str = "all"

    split_enabled: bool = True
    split_max_per_iter: int = 1
    split_min_cluster_size: int = 20
    split_ll_margin: float = 6.0
    split_confidence_max: float = 0.30
    split_gap_median_min: float = 0.10
    split_min_conditions: int = 1
    split_cooldown_iters: int = 1

    merge_enabled: bool = True
    merge_max_per_iter: int = 1
    merge_similarity_min: float = 0.15  # confusion-based threshold (symmetric avg p(z_j|x) between clusters)
    merge_l_diff_ratio_max: float = 0.10
    merge_c_diff_ratio_max: float = 0.10
    merge_min_conditions: int = 1

    remove_enabled: bool = True
    remove_max_per_iter: int = 0
    remove_min_cluster_size: int = 5
    remove_ll_factor: float = 1.01

    revise_enabled: bool = True
    revise_max_per_iter: int = 1
    revise_ll_margin: float = 4.0
    revise_confidence_max: float = 0.25
    revise_min_conditions: int = 1
    revise_cooldown_iters: int = 1

    add_enabled: bool = True
    add_low_confidence_max: float = 0.2
    add_low_confidence_quantile: float = 0.05
    add_min_poorly_explained: int = 5
    add_items_per_new_cluster: int = 50
    add_max_new_clusters_per_iter: int = 3
    add_max_total_clusters: int = 200
    add_min_group_size: int = 4
    add_entropy_min: float = 0.9
    add_cohesion_min: float = 0.15  # KL-divergence cohesion: exp(-mean_kl), 1.0=perfect, 0.15≈lenient
    add_cooldown_iters: int = 1

    rollback_on_metric_degrade: bool = True
    accept_metric: str = "assigned"
    accept_min_delta: float = 0.0
    noop_patience: int = 2
    best_operation_per_iteration: bool = False
    max_same_operation_streak: int = 0
    structural_label_mode: str = "semantic"
    split_label_pair_attempts: int = 3
    llm_label_max_attempts: int = 6

    def operation_enabled(self, operation_name: str) -> bool:
        if operation_name not in {"split", "merge", "remove", "add", "revise"}:
            raise ValueError(f"Unsupported operation name: {operation_name}")
        if self.tune_operation != "all" and self.tune_operation != operation_name:
            return False
        return bool(getattr(self, f"{operation_name}_enabled"))

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
    rebuilt = ProbabilityCalibrator(
        choices=new_choices,
        logprob_scorer=existing_calibrator.logprob_scorer,
        full_logprob_fn=existing_calibrator.full_logprob_fn,
        num_trials=existing_calibrator.num_trials,
        content_free_input=existing_calibrator.content_free_input,
        alpha=existing_calibrator.alpha,
        verbose=verbose,
        batch_prompts=existing_calibrator.batch_prompts,
        batch_permutations=existing_calibrator.batch_permutations,
        vllm_model=getattr(existing_calibrator, "vllm_model", None),
        vllm_tokenizer=getattr(existing_calibrator, "vllm_tokenizer", None),
    )
    # Preserve tokenizer handle for token-count-aware metrics when available.
    existing_tokenizer = getattr(existing_calibrator, "tokenizer", None)
    if existing_tokenizer is not None:
        setattr(rebuilt, "tokenizer", existing_tokenizer)
    return rebuilt


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
    If duplicate labels occur, append a human-readable variant suffix.
    """
    unique_label_map: dict[int, str] = {}
    used_labels: set[str] = set()
    choices: list[str] = []

    for cid in sorted(cluster_ids):
        base_label = sanitize_text(raw_label_map.get(cid, f"{fallback_prefix}_{cid}"))
        if not base_label:
            base_label = f"{fallback_prefix}_{cid}"
        unique_label = base_label
        if unique_label in used_labels:
            variant_idx = 2
            while f"{base_label} Variant {variant_idx}" in used_labels:
                variant_idx += 1
            unique_label = f"{base_label} Variant {variant_idx}"
            logging.warning(
                "Duplicate cluster label '%s' detected. Using unique label '%s' for cluster_id=%s.",
                base_label,
                unique_label,
                cid,
            )
        used_labels.add(unique_label)
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


# NOTE: compute_sentence_embeddings has been removed in LLM-only mode.
# All geometric operations now use LLM posteriors (PZX, log_px_given_z).


def estimate_token_counts(
    texts: list[str],
    tokenizer=None,
) -> np.ndarray:
    """
    Estimate token counts per text. Uses model tokenizer when available,
    otherwise falls back to whitespace token counts.
    """
    counts: list[float] = []
    use_model_tokenizer = tokenizer is not None and hasattr(tokenizer, "encode")
    for txt in texts:
        t = str(txt)
        if use_model_tokenizer:
            try:
                n_tokens = len(tokenizer.encode(t, add_special_tokens=False))
            except Exception:
                n_tokens = len(t.split())
        else:
            n_tokens = len(t.split())
        counts.append(float(max(1, n_tokens)))
    return np.array(counts, dtype=float)


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


###############################################################################
# LLM-ONLY GEOMETRIC HELPERS (no sentence-transformer embeddings)
###############################################################################

def split_cluster_by_posteriors(
    PZX_cluster: np.ndarray,
    log_px_given_z_cluster: np.ndarray,
    assigned_col_idx: int,
    min_runner_up_dominance: float = 0.3,
) -> np.ndarray:
    """Split a cluster using LLM posteriors (no sentence-transformer embeddings).

    Option B (primary): Split by runner-up disagreement.
      For each doc, find its best cluster AFTER masking the assigned cluster.
      Group docs by whether they prefer the dominant runner-up or not.

    Option A (fallback): KMeans(k=2) on the posterior vectors.

    Returns array of 0/1 labels for the 2-way split.
    """
    from sklearn.cluster import KMeans as _KMeans

    n = PZX_cluster.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    # --- Option B: runner-up split ---
    masked_ll = log_px_given_z_cluster.copy()
    masked_ll[:, assigned_col_idx] = -np.inf
    runner_ups = masked_ll.argmax(axis=1)
    unique_runners, counts = np.unique(runner_ups, return_counts=True)

    if len(unique_runners) >= 2:
        sorted_idx = np.argsort(counts)[::-1]
        top_runner = unique_runners[sorted_idx[0]]
        top_count = counts[sorted_idx[0]]
        dominance = top_count / n

        if min_runner_up_dominance <= dominance <= (1.0 - min_runner_up_dominance):
            labels = (runner_ups == top_runner).astype(int)
            if labels.sum() >= 2 and (n - labels.sum()) >= 2:
                logging.info(
                    "  Split by runner-up: dominant runner-up cluster idx %s has %.1f%% of docs.",
                    top_runner,
                    dominance * 100,
                )
                return labels

    # --- Option A fallback: KMeans on posteriors ---
    logging.info(
        "  Split by runner-up unclear; falling back to KMeans on posteriors."
    )
    kmeans = _KMeans(n_clusters=2, n_init="auto", random_state=42)
    return kmeans.fit_predict(PZX_cluster)


def compute_confusion_matrix_from_pzx(
    PZX: np.ndarray,
    assigned_idx: np.ndarray,
    K: int,
) -> np.ndarray:
    """Compute K×K confusion matrix from posteriors.

    Entry (i, j) = mean p(z_j|x) for x currently assigned to cluster i.
    High off-diagonal values mean the LLM confuses cluster i with cluster j.
    """
    confusion = np.zeros((K, K))
    for j in range(K):
        mask = assigned_idx == j
        if mask.any():
            confusion[j, :] = PZX[mask].mean(axis=0)
    return confusion


def kl_divergence_cohesion(
    PZX_subset: np.ndarray,
    eps: float = 1e-30,
) -> float:
    """Compute cohesion of a group via average KL divergence from group mean posterior.

    Low KL divergence = high cohesion (docs agree on posterior distribution).
    Returns exp(-mean_kl): 1.0 for perfect agreement, decaying toward 0.
    This value is directly comparable to cosine similarity in threshold checks.
    """
    if PZX_subset.shape[0] < 2:
        return 1.0

    mean_posterior = PZX_subset.mean(axis=0)
    mean_posterior = np.clip(mean_posterior, eps, None)
    mean_posterior = mean_posterior / mean_posterior.sum()

    kl_values = []
    for i in range(PZX_subset.shape[0]):
        p = np.clip(PZX_subset[i], eps, None)
        p = p / p.sum()
        kl_values.append(float(np.sum(p * np.log(p / mean_posterior))))
    mean_kl = float(np.mean(kl_values))
    return float(np.exp(-mean_kl))


###############################################################################

def compute_cluster_metrics(
        merged_df: pd.DataFrame,
        PZX: np.ndarray,
        row_log_px_given_z: np.ndarray,
        idx_to_pzx_row_map: dict,
        choices_list: list[str],
        baseline_log_px: float,
        verbose: bool = False,
        text_column: str = 'sentence_text',
        p_z_prior: np.ndarray | None = None
    ) -> dict:
    """Compute per-cluster metrics using pre-computed LLM posteriors (no sentence-transformer embeddings).

    Uses the PZX matrix and row_log_px_given_z from the E-step, avoiding
    redundant per-text calibrator calls and removing the need for sentence-
    transformer embeddings entirely.
    """
    logging.info("Starting computation of cluster metrics (LLM-only mode)...")
    cluster_metrics = {}
    all_assigned_log_likelihoods = []
    choice_to_idx = {c: i for i, c in enumerate(choices_list)}
    K = len(choices_list)

    cluster_groups = merged_df.groupby('cluster_id')
    if verbose:
        cluster_iterator = tqdm(cluster_groups, desc="Computing cluster metrics (LLM-only)", unit="cluster", disable=False)
    else:
        cluster_iterator = cluster_groups

    for cluster_id, group in cluster_iterator:
        indices = group.index.tolist()
        if not indices:
            logging.warning(f"Cluster {cluster_id} has no data points. Skipping.")
            continue

        label = (
            group['agglomerative_label'].iloc[0]
            if 'agglomerative_label' in group.columns and not group.empty
            else f"cluster_{cluster_id}"
        )
        label_idx = choice_to_idx.get(label)

        # Get PZX rows for this cluster
        valid_indices = [idx for idx in indices if idx in idx_to_pzx_row_map]
        if not valid_indices:
            cluster_metrics[cluster_id] = {
                'L_z': float('-inf'), 'C_z': 0.0, 'V_z': 0.0,
                'gap_median': 0.0,
                'size': len(indices), 'mapped_size': 0,
                'label': label,
            }
            continue

        pzx_rows = np.array([idx_to_pzx_row_map[idx] for idx in valid_indices])
        pzx_subset = PZX[pzx_rows]          # (n_j, K)
        ll_subset = row_log_px_given_z[pzx_rows]  # (n_j, K)

        if label_idx is None:
            logging.warning(
                "Label '%s' for cluster %s not in choices. Assigning default metrics.",
                label, cluster_id,
            )
            L_z = float('-inf')
            C_z = 0.0
        else:
            L_z = float(ll_subset[:, label_idx].mean())
            C_z = float(pzx_subset[:, label_idx].mean())
            all_assigned_log_likelihoods.extend(ll_subset[:, label_idx].tolist())

        # gap_median: median of (top1 - top2) posterior gap
        gap_vals = []
        for i in range(pzx_subset.shape[0]):
            pzx_row = pzx_subset[i]
            if pzx_row.size > 1:
                order = np.sort(pzx_row)[::-1]
                gap_vals.append(float(order[0] - order[1]))

        # V_z: mean posterior entropy (replaces ST embedding variance)
        entropy_vals = []
        for i in range(pzx_subset.shape[0]):
            p = np.clip(pzx_subset[i], 1e-30, None)
            p = p / p.sum()
            entropy_vals.append(float(-np.sum(p * np.log(p))))
        V_z = float(np.mean(entropy_vals)) if entropy_vals else 0.0

        cluster_metrics[cluster_id] = {
            'L_z': L_z,
            'C_z': C_z,
            'V_z': V_z,
            'gap_median': float(np.median(gap_vals)) if gap_vals else 0.0,
            'size': len(indices),
            'mapped_size': len(valid_indices),
            'label': label,
        }
        logging.debug(
            "Cluster %s ('%s', size %s, mapped %s): L_z=%.4f, C_z=%.4f, V_z=%.4f",
            cluster_id, label, len(indices), len(valid_indices), L_z, C_z, V_z,
        )

    # --- Build confusion matrix for merge decisions ---
    assigned_idx_arr = np.full(PZX.shape[0], -1, dtype=int)
    for cid, grp in merged_df.groupby('cluster_id'):
        lbl = (
            grp['agglomerative_label'].iloc[0]
            if 'agglomerative_label' in grp.columns and not grp.empty
            else f"cluster_{cid}"
        )
        j = choice_to_idx.get(lbl)
        if j is not None:
            for idx in grp.index:
                if idx in idx_to_pzx_row_map:
                    assigned_idx_arr[idx_to_pzx_row_map[idx]] = j

    confusion = compute_confusion_matrix_from_pzx(PZX, assigned_idx_arr, K)
    cluster_metrics['_confusion_matrix'] = confusion
    cluster_metrics['_confusion_cid_to_choice_idx'] = {
        int(cid): choice_to_idx.get(
            merged_df.loc[merged_df['cluster_id'] == cid, 'agglomerative_label'].iloc[0],
            -1,
        )
        for cid in merged_df['cluster_id'].unique()
    }

    cluster_metrics['_baseline_log_px'] = baseline_log_px
    if all_assigned_log_likelihoods:
        cluster_metrics['_baseline_log_pxz_assigned'] = float(np.mean(all_assigned_log_likelihoods))
    else:
        cluster_metrics['_baseline_log_pxz_assigned'] = float('-inf')
    logging.info(f"Finished computation of cluster metrics (LLM-only). Baseline log p(x) = {baseline_log_px:.4f}")
    return cluster_metrics


def decide_schema_updates(
    cluster_metrics: dict,
    config: SchemaRefinementConfig,
    split_cooldown: dict[int, int],
    revise_cooldown: dict[int, int],
    iter_idx: int,
    verbose: bool = False,
    log_top_pairs: int = 0,
):
    """Determine which clusters to split, merge, remove. Returns dictionaries describing actions."""
    logging.info("Starting schema update decisions...")
    actions = {'split': [], 'merge': [], 'remove': [], 'revise': []}
    split_enabled = config.operation_enabled("split")
    merge_enabled = config.operation_enabled("merge")
    remove_enabled = config.operation_enabled("remove")
    revise_enabled = config.operation_enabled("revise")
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
        "Using decision baseline L_baseline_assigned=%.4f (and log p(x) baseline=%.4f), Median V_z=%.4f. "
        "tune_operation=%s; enabled ops: split=%s merge=%s remove=%s revise=%s. "
        "remove: size<%s and L_z<%.4f. "
        "split: size>=%s, min_conditions=%s over [L_z<%.4f, C_z<%.4f, gap_median>=%.4f], cooldown=%s. "
        "merge: require cosine_sim>=%.2f AND min_conditions=%s over [cosine_sim, L_ratio<=%.3f, C_ratio<=%.3f]. "
        "revise: min_conditions=%s over [L_z<%.4f, C_z<%.4f], cooldown=%s.",
        decision_baseline,
        baseline_log_px,
        median_V_z,
        config.tune_operation,
        split_enabled,
        merge_enabled,
        remove_enabled,
        revise_enabled,
        config.remove_min_cluster_size,
        decision_baseline * config.remove_ll_factor,
        config.split_min_cluster_size,
        config.split_min_conditions,
        decision_baseline - config.split_ll_margin,
        config.split_confidence_max,
        config.split_gap_median_min,
        config.split_cooldown_iters,
        config.merge_similarity_min,
        config.merge_min_conditions,
        config.merge_l_diff_ratio_max,
        config.merge_c_diff_ratio_max,
        config.revise_min_conditions,
        decision_baseline - config.revise_ll_margin,
        config.revise_confidence_max,
        config.revise_cooldown_iters,
    )

    # Track margins to understand how close we are to thresholds
    split_margins = []
    remove_margins = []

    # --- SPLIT & REMOVE ---
    for cid in valid_cluster_ids:
        m = cluster_metrics[cid]
        # Per-cluster threshold diagnostics
        logging.info(
            "Cluster %s ('%s'): L_z=%.4f, C_z=%.4f, V_z=%.4f, size=%s. remove_thresh=%.4f, split_thresh=%.4f",
            cid,
            m.get('label', 'N/A'),
            m['L_z'],
            m['C_z'],
            m['V_z'],
            m['size'],
            decision_baseline * config.remove_ll_factor,
            decision_baseline - config.split_ll_margin,
        )
        # Removal Check
        if (
            remove_enabled
            and m['size'] < config.remove_min_cluster_size
            and m['L_z'] < decision_baseline * config.remove_ll_factor
        ):
            actions['remove'].append(cid)
            logging.info(
                "  Suggest REMOVE for cluster %s ('%s'): size %s < %s AND L_z %.4f < baseline_thresh %.4f.",
                cid,
                m.get('label', 'N/A'),
                m['size'],
                config.remove_min_cluster_size,
                m['L_z'],
                decision_baseline * config.remove_ll_factor,
            )
            continue # If removed, don't consider for split
        remove_margins.append(m['L_z'] - (decision_baseline * config.remove_ll_factor))
        # Split Check (gated)
        if split_enabled:
            cooldown_until = split_cooldown.get(cid)
            in_cooldown = cooldown_until is not None and iter_idx <= cooldown_until
            split_conditions = [
                m['L_z'] < (decision_baseline - config.split_ll_margin),
                m['C_z'] < config.split_confidence_max,
                m.get('gap_median', 0.0) >= config.split_gap_median_min,
            ]
            if (
                (not in_cooldown)
                and m['size'] >= config.split_min_cluster_size
                and int(sum(bool(x) for x in split_conditions)) >= max(1, int(config.split_min_conditions))
            ):
                actions['split'].append(cid)
                logging.info(
                    f"  Suggest SPLIT for cluster {cid} ('{m.get('label', 'N/A')}'): "
                    f"passed {sum(bool(x) for x in split_conditions)}/{len(split_conditions)} split conditions."
                )
        split_margins.append(m['L_z'] - (decision_baseline - config.split_ll_margin))

    # --- MERGE (LLM-only: confusion matrix from posteriors) ---
    top_pairs = []
    # Filter out clusters already marked for removal or split before considering for merge
    eligible_for_merge_ids = [cid for cid in valid_cluster_ids if cid not in actions['remove'] and cid not in actions['split']]
    confusion = cluster_metrics.get('_confusion_matrix')
    cid_to_choice_idx = cluster_metrics.get('_confusion_cid_to_choice_idx', {})
    if not merge_enabled:
        logging.info("Merge operation disabled for this run.")
    elif len(eligible_for_merge_ids) < 2:
        logging.info("Not enough eligible clusters to consider merging.")
    elif confusion is None:
        logging.warning("No confusion matrix available for merge decisions. Skipping merge.")
    else:
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

                ci = cid_to_choice_idx.get(id_i, -1)
                cj = cid_to_choice_idx.get(id_j, -1)
                if ci < 0 or cj < 0:
                    continue

                # Symmetric confusion: how much the LLM confuses i with j
                sim_val = float((confusion[ci, cj] + confusion[cj, ci]) / 2.0)
                if log_top_pairs > 0:
                    top_pairs.append((sim_val, id_i, id_j))
                m_i = cluster_metrics[id_i]
                m_j = cluster_metrics[id_j]
                diff_L = abs(m_i['L_z'] - m_j['L_z'])
                diff_C = abs(m_i['C_z'] - m_j['C_z'])
                diff_L_ratio = diff_L / max(abs(m_i['L_z']), abs(m_j['L_z']), 1e-8)
                diff_C_ratio = diff_C / max(m_i['C_z'], m_j['C_z'], 1e-8)
                # Structural guardrail: confusion must exceed threshold for any merge.
                if sim_val < config.merge_similarity_min:
                    continue
                merge_conditions = [
                    sim_val >= config.merge_similarity_min,
                    diff_L_ratio <= config.merge_l_diff_ratio_max,
                    diff_C_ratio <= config.merge_c_diff_ratio_max,
                ]
                if (
                    int(sum(bool(x) for x in merge_conditions)) >= max(1, int(config.merge_min_conditions))
                ):
                    actions['merge'].append(tuple(sorted((id_i, id_j))))
                    merged_already.add(id_i)
                    merged_already.add(id_j)
                    logging.info(
                        "  Suggest MERGE for clusters %s ('%s') and %s ('%s'): "
                        "passed %s/%s merge conditions (confusion=%.4f, L_diff_ratio=%.4f, C_diff_ratio=%.4f).",
                        id_i,
                        m_i.get('label', 'N/A'),
                        id_j,
                        m_j.get('label', 'N/A'),
                        sum(bool(x) for x in merge_conditions),
                        len(merge_conditions),
                        sim_val,
                        diff_L_ratio,
                        diff_C_ratio,
                    )
                    break # Merge id_i with one cluster, then move to next i
    
    # Deduplicate merge pairs (if (A,B) and (B,A) somehow got in)
    actions['merge'] = sorted(list(set(actions['merge'])))
    if config.merge_max_per_iter > 0:
        actions['merge'] = actions['merge'][:config.merge_max_per_iter]
    if config.remove_max_per_iter > 0 and actions['remove']:
        actions['remove'] = sorted(actions['remove'], key=lambda cid: cluster_metrics[cid]['L_z'])[:config.remove_max_per_iter]

    # --- REVISE ---
    merged_ids = {cid for pair in actions['merge'] for cid in pair}
    eligible_for_revise = [
        cid for cid in valid_cluster_ids
        if cid not in actions['remove'] and cid not in actions['split'] and cid not in merged_ids
    ]
    if not revise_enabled:
        logging.info("Revise operation disabled for this run.")
    elif not eligible_for_revise:
        logging.info("No eligible clusters to revise.")
    else:
        for cid in eligible_for_revise:
            m = cluster_metrics[cid]
            cooldown_until = revise_cooldown.get(cid)
            in_cooldown = cooldown_until is not None and iter_idx <= cooldown_until
            revise_conditions = [
                m['L_z'] < (decision_baseline - config.revise_ll_margin),
                m['C_z'] < config.revise_confidence_max,
            ]
            if (not in_cooldown) and int(sum(bool(x) for x in revise_conditions)) >= max(1, int(config.revise_min_conditions)):
                actions['revise'].append(cid)
                logging.info(
                    "  Suggest REVISE for cluster %s ('%s'): passed %s/%s revise conditions.",
                    cid,
                    m.get('label', 'N/A'),
                    sum(bool(x) for x in revise_conditions),
                    len(revise_conditions),
                )
    if config.revise_max_per_iter > 0 and actions['revise']:
        actions['revise'] = sorted(actions['revise'], key=lambda cid: cluster_metrics[cid]['L_z'])[:config.revise_max_per_iter]
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
    if split_enabled and actions['split']:
        actions['split'] = sorted(actions['split'], key=lambda cid: cluster_metrics[cid]['L_z'])
        if config.split_max_per_iter > 0:
            actions['split'] = actions['split'][:config.split_max_per_iter]
    else:
        actions['split'] = []
    if not merge_enabled:
        actions['merge'] = []
    if not remove_enabled:
        actions['remove'] = []
    if not revise_enabled:
        actions['revise'] = []
    logging.info(f"Finished schema update decisions. Actions: {actions}")

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
    # drop legacy synthetic suffixes if present
    if "__c" in s:
        s = s.split("__c")[0]
    # drop keywords after ":"
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    return s


def _sample_texts_for_labeling(texts: list[str], max_texts: int = 10, seed: int = 42) -> list[str]:
    cleaned = [t.replace("\n", " ").strip() for t in texts if t and str(t).strip()]
    if len(cleaned) <= max_texts:
        return cleaned
    rng = np.random.default_rng(seed + len(cleaned))
    idx = rng.choice(len(cleaned), size=max_texts, replace=False)
    idx = np.sort(idx)
    return [cleaned[i] for i in idx]


def _fallback_distinct_label_from_texts(
    texts: list[str],
    banned_names: list[str] | None = None,
) -> str | None:
    banned_keys = {_canonical_label_key(x) for x in (banned_names or []) if sanitize_text(x)}
    banned_keys.update({"topic", "misc", "subtopic", "variant"})

    for k in (4, 6, 8):
        name, keywords = propose_label_and_keywords_from_texts(texts, parent_label=None, max_keywords=k)
        candidate = build_label_string(name, keywords)
        if _canonical_label_key(candidate) not in banned_keys:
            return candidate

    # Last semantic fallback: build a name from top keywords directly.
    _, keywords = propose_label_and_keywords_from_texts(texts, parent_label=None, max_keywords=8)
    if keywords:
        name = " ".join(keywords[:2]) if len(keywords) >= 2 else keywords[0]
        candidate = build_label_string(name, keywords[2:8])
        if _canonical_label_key(candidate) not in banned_keys:
            return candidate
    return None


def _collect_existing_cluster_labels(
    df: pd.DataFrame,
    *,
    exclude_cluster_ids: set[int] | None = None,
    cluster_col: str = "cluster_id",
    label_col: str = "agglomerative_label",
) -> list[str]:
    if df.empty or cluster_col not in df.columns or label_col not in df.columns:
        return []
    exclude = {int(x) for x in (exclude_cluster_ids or set())}
    labels: list[str] = []
    grouped = df.groupby(cluster_col)[label_col].first().astype(str).to_dict()
    for cid, lbl in grouped.items():
        if int(cid) in exclude:
            continue
        labels.append(str(lbl))
    return labels


def _build_parent_relative_split_labels(
    parent_label: str,
    parent_id: int,
) -> tuple[str, str]:
    parent_base = base_label(parent_label) or "Cluster"
    new_parent_label = build_label_string(f"{parent_base} Parent Z{int(parent_id)}")
    new_child_label = build_label_string(f"{parent_base} Child Z{int(parent_id)}")
    if _canonical_label_key(new_parent_label) == _canonical_label_key(new_child_label):
        new_child_label = build_label_string(f"{parent_base} Child One Z{int(parent_id)}")
    return new_parent_label, new_child_label


def _build_parent_relative_merge_label(
    source_labels: list[str],
    root_id: int,
) -> str:
    if source_labels:
        first = base_label(source_labels[0]) or "Cluster"
        second = base_label(source_labels[1]) if len(source_labels) > 1 else ""
        stem = f"{first} {second}".strip()
    else:
        stem = "Merged Cluster"
    return build_label_string(f"{stem} Merged Z{int(root_id)}")


def _propose_split_labels_with_retries(
    *,
    parent_samples: list[str],
    child_samples: list[str],
    parent_label: str,
    model_identifier: str,
    tokenizer,
    model,
    context_prefix: str,
    existing_forbidden_names: list[str],
    pair_attempts: int,
    llm_label_max_attempts: int,
) -> tuple[str | None, str | None]:
    parent_base = base_label(parent_label) or "Cluster"
    pair_attempts = max(1, int(pair_attempts))
    banned_pool: list[str] = [n for n in existing_forbidden_names if sanitize_text(n)]
    seen_parent_child_keys: set[str] = set()

    for pair_try in range(1, pair_attempts + 1):
        parent_llm = _label_cluster_vllm(
            parent_samples,
            model_identifier,
            tokenizer=tokenizer,
            model=model,
            contrast_texts=child_samples,
            parent_label=parent_base,
            banned_names=banned_pool,
            context=f"{context_prefix}:pair{pair_try}:parent",
            max_attempts=llm_label_max_attempts,
        )
        new_parent_label = None
        if parent_llm:
            name, _, keywords = parent_llm
            new_parent_label = build_label_string(name or parent_base, keywords)

        child_banned = list(banned_pool)
        child_banned.extend([new_parent_label, parent_base])
        child_llm = _label_cluster_vllm(
            child_samples,
            model_identifier,
            tokenizer=tokenizer,
            model=model,
            contrast_texts=parent_samples,
            parent_label=parent_base,
            banned_names=[x for x in child_banned if x],
            context=f"{context_prefix}:pair{pair_try}:child",
            max_attempts=llm_label_max_attempts,
        )
        new_child_label = None
        if child_llm:
            name, _, keywords = child_llm
            new_child_label = build_label_string(name or "Split Child", keywords)

        parent_key = _canonical_label_key(new_parent_label)
        child_key = _canonical_label_key(new_child_label)
        pair_key = f"{parent_key}::{child_key}"
        if (
            new_parent_label
            and new_child_label
            and parent_key
            and child_key
            and parent_key != child_key
            and pair_key not in seen_parent_child_keys
        ):
            return new_parent_label, new_child_label

        seen_parent_child_keys.add(pair_key)
        if new_parent_label:
            banned_pool.append(new_parent_label)
        if new_child_label:
            banned_pool.append(new_child_label)

    return None, None


def _label_cluster_vllm(
    texts: list[str],
    model_name: str,
    tokenizer=None,
    model=None,
    contrast_texts: list[str] | None = None,
    parent_label: str | None = None,
    banned_names: list[str] | None = None,
    context: str | None = None,
    max_attempts: int = 4,
) -> tuple[str, str, list[str]] | None:
    """
    Label a cluster using vLLM. Returns (name, description, keywords).
    """
    try:
        from utils_vllm_client import robust_parse_outputs
        from vllm import SamplingParams
    except Exception:
        return None

    examples = [t.replace("\n", " ").strip() for t in texts if t][:8]
    if not examples:
        return None

    banned_name_keys = {
        _canonical_label_key(base_label(x))
        for x in (banned_names or [])
        if base_label(x)
    }
    parent_label = base_label(parent_label) if parent_label else None
    contrast_examples = [t.replace("\n", " ").strip() for t in (contrast_texts or []) if t][:8]

    try:
        if tokenizer is None or model is None:
            from utils_vllm_client import load_model as load_vllm_model

            tokenizer, model = load_vllm_model(model_name)
    except Exception as exc:
        logging.warning("vLLM cluster labeling model load failed%s: %s", f" ({context})" if context else "", exc)
        return None

    generic_tokens = ("subtopic", "variant", "topic", "misc", "other")
    rejection_reasons: list[str] = []
    dynamic_banned_keys = set(banned_name_keys)

    for attempt in range(1, max(1, max_attempts) + 1):
        # Slightly increase temperature on retries to escape repetitive generic names.
        attempt_temp = min(0.45, 0.10 + 0.10 * (attempt - 1))
        sampling_params = SamplingParams(temperature=attempt_temp, top_p=0.90, max_tokens=220)

        guidance_lines = [
            "Produce a semantically specific, standalone cluster label.",
            "Avoid generic labels such as Subtopic, Variant, Topic, Misc, Other.",
            "Name must be 2-5 words and should describe discourse function, not entities.",
            "Return ONLY valid JSON (no markdown, no preamble).",
        ]
        if parent_label:
            guidance_lines.append(f"Parent context label: '{parent_label}'.")
        if contrast_examples:
            guidance_lines.append("Ensure the name is distinct from the contrast cluster examples.")
        if dynamic_banned_keys:
            banned_list = ", ".join(sorted(dynamic_banned_keys))
            guidance_lines.append(f"Do NOT use any name equal to or derived from: {banned_list}.")

        prompt = (
            "Given these example sentences, generate a cluster label.\n\n"
            "Output schema (exact keys):\n"
            '{"name":"<2-5 words>","description":"<one sentence>","keywords":["k1","k2"]}\n\n'
            "Rules:\n- " + "\n- ".join(guidance_lines) + "\n\n"
            "Target cluster examples:\n- " + "\n- ".join(examples)
        )
        if contrast_examples:
            prompt += "\n\nContrast cluster examples:\n- " + "\n- ".join(contrast_examples)

        prompt_dict = [
            {"role": "system", "content": "You are an experienced analyst."},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(prompt_dict, tokenize=False, add_generation_prompt=True)

        try:
            outputs = model.generate([formatted_prompt], sampling_params=sampling_params, use_tqdm=False)
        except Exception as exc:
            rejection_reasons.append(f"attempt={attempt}:generate_error={exc}")
            continue

        raw_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        if not raw_text:
            rejection_reasons.append(f"attempt={attempt}:empty_output")
            continue

        data = robust_parse_outputs(raw_text)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]

        name = ""
        desc = ""
        keywords: list[str] = []

        if isinstance(data, dict):
            name = sanitize_text(data.get("name") or data.get("label") or data.get("title") or "")
            desc = sanitize_text(data.get("description", ""))
            keywords_raw = data.get("keywords", [])
            if isinstance(keywords_raw, str):
                keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
            elif isinstance(keywords_raw, list):
                keywords = [sanitize_text(k) for k in keywords_raw if sanitize_text(k)]
        else:
            # Parse-repair for partially structured outputs.
            name_match = re.search(r'"(?:name|label|title)"\s*:\s*"([^"]+)"', raw_text)
            desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', raw_text)
            kw_match = re.search(r'"keywords"\s*:\s*\[([^\]]*)\]', raw_text)
            if name_match:
                name = sanitize_text(name_match.group(1))
            if desc_match:
                desc = sanitize_text(desc_match.group(1))
            if kw_match:
                keywords = [
                    sanitize_text(tok.strip().strip('"').strip("'"))
                    for tok in kw_match.group(1).split(",")
                    if sanitize_text(tok.strip().strip('"').strip("'"))
                ]
            if not name:
                first_line = sanitize_text(raw_text.splitlines()[0] if raw_text.splitlines() else raw_text)
                first_line = first_line.replace('"', "").replace("{", "").replace("}", "")
                name = sanitize_text(first_line.split(":", 1)[0])

        name = base_label(name)
        canonical_name = _canonical_label_key(name)
        if not canonical_name:
            rejection_reasons.append(f"attempt={attempt}:missing_name")
            continue
        if canonical_name in dynamic_banned_keys:
            rejection_reasons.append(f"attempt={attempt}:banned_name={canonical_name}")
            continue
        if any(tok in canonical_name for tok in generic_tokens):
            dynamic_banned_keys.add(canonical_name)
            rejection_reasons.append(f"attempt={attempt}:generic_name={canonical_name}")
            continue
        if len(name.split()) < 2:
            # Avoid one-word labels that tend to be underspecified.
            dynamic_banned_keys.add(canonical_name)
            rejection_reasons.append(f"attempt={attempt}:too_short_name={canonical_name}")
            continue

        if not keywords:
            _, fallback_keywords = propose_label_and_keywords_from_texts(examples, parent_label=None, max_keywords=6)
            keywords = [sanitize_text(k) for k in fallback_keywords if sanitize_text(k)]
        keywords = keywords[:6]
        return name, desc, keywords

    if rejection_reasons:
        logging.warning(
            "vLLM cluster labeling exhausted retries%s. Reasons: %s",
            f" ({context})" if context else "",
            "; ".join(rejection_reasons[-max_attempts:]),
        )
    return None


def _enforce_unique_choice_labels(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    label_col: str = "agglomerative_label",
) -> int:
    """
    Force labels to be unique under the exact normalization used for choices.
    Returns number of cluster labels modified.
    """
    if df.empty or cluster_col not in df.columns or label_col not in df.columns:
        return 0

    cluster_ids = sorted(df[cluster_col].unique())
    cluster_labels = df.groupby(cluster_col)[label_col].first().astype(str).to_dict()

    seen_keys: set[str] = set()
    base_variant_counters: dict[str, int] = {}
    new_map: dict[int, str] = {}
    changed = 0
    for cid in cluster_ids:
        canonical = build_label_string(cluster_labels.get(cid, f"cluster_{cid}"))
        canonical_key = _canonical_label_key(canonical)
        if canonical_key in seen_keys:
            base = base_label(canonical) or "Misc"
            next_variant = max(2, base_variant_counters.get(base, 1) + 1)
            while True:
                candidate = build_label_string(f"{base} Variant {next_variant}")
                candidate_key = _canonical_label_key(candidate)
                if candidate_key not in seen_keys:
                    canonical = candidate
                    canonical_key = candidate_key
                    base_variant_counters[base] = next_variant
                    break
                next_variant += 1
            changed += 1
        seen_keys.add(canonical_key)
        base = base_label(canonical) or "Misc"
        base_variant_counters[base] = max(base_variant_counters.get(base, 1), 1)
        new_map[cid] = canonical

    if changed > 0:
        df[label_col] = df[cluster_col].map(new_map)
    return changed


def _select_poor_candidates(
    pmax_list: list[float],
    entropy_list: list[float],
    assigned_rank_list: list[int | None],
    min_poorly_explained_for_add: int,
    low_confidence_threshold_add: float,
    low_confidence_quantile_add: float,
    entropy_threshold_add: float,
) -> list[int]:
    N = len(pmax_list)
    if N == 0:
        return [], low_confidence_threshold_add
    K = max(min_poorly_explained_for_add, int(0.05 * N))
    order = np.argsort(pmax_list)
    bottom_k = order[:K]
    q = float(np.clip(low_confidence_quantile_add, 0.0, 1.0))
    quantile_threshold = float(np.quantile(np.asarray(pmax_list, dtype=float), q))
    dynamic_pmax_threshold = max(float(low_confidence_threshold_add), quantile_threshold)
    selected = []
    for idx in bottom_k:
        pmax = pmax_list[idx]
        ent = entropy_list[idx]
        rank = assigned_rank_list[idx]
        if (
            pmax <= dynamic_pmax_threshold
            and ent >= entropy_threshold_add
            and (rank is None or rank > 1)
        ):
            selected.append(idx)
    return selected, dynamic_pmax_threshold


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
    PZX: np.ndarray,
    row_log_px_given_z: np.ndarray,
    idx_to_pzx_row_map: dict,
    choices_list: list[str],
    actions: dict,
    next_new_cluster_id: int,
) -> tuple[pd.DataFrame, int, list[tuple[int, int]], list[tuple[int, list[int]]]]:
    """Apply split, merge, remove actions to merged_df using LLM posteriors (no ST embeddings)."""
    pre_clusters = merged_df['cluster_id'].nunique()
    logging.info(f"Starting to apply schema updates (LLM-only). Initial #clusters: {pre_clusters}")
    current_df = merged_df.copy()
    split_pairs: list[tuple[int, int]] = []
    merge_groups: list[tuple[int, list[int]]] = []

    choice_to_idx = {c: i for i, c in enumerate(choices_list)}

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
        groups: dict[int, list[int]] = {}
        for cid in sorted(parent.keys()):
            root = find_set(cid)
            groups.setdefault(root, []).append(cid)
        merge_groups = [(root, members) for root, members in groups.items() if len(members) > 1]
        if merge_groups:
            logging.info("  Merge groups formed: %s", merge_groups)
        current_df['cluster_id'] = current_df['cluster_id'].apply(lambda x: find_set(x) if x in parent else x)

    count_before_split = current_df['cluster_id'].nunique()
    logging.info(f"  #clusters after merge: {current_df['cluster_id'].nunique()}")

    # Split clusters using LLM posteriors
    if actions['split']:
        logging.info(f"Splitting {len(actions['split'])} clusters (LLM-only): {actions['split']}")
        for cid_to_split in actions['split']:
            if cid_to_split not in current_df['cluster_id'].unique():
                logging.warning(f"Cluster {cid_to_split} marked for split, but it no longer exists (possibly merged). Skipping split.")
                continue
            subset_indices = current_df[current_df['cluster_id'] == cid_to_split].index
            if len(subset_indices) < 2:
                logging.warning(f"Cluster {cid_to_split} has < 2 items after other ops. Cannot split.")
                continue

            valid_subset_indices = []
            pzx_positions = []
            for idx in subset_indices:
                if idx in idx_to_pzx_row_map:
                    valid_subset_indices.append(idx)
                    pzx_positions.append(idx_to_pzx_row_map[idx])
            if len(pzx_positions) < 2:
                logging.warning(f"Cluster {cid_to_split} has < 2 valid posterior rows. Cannot split.")
                continue

            # Determine the choice column index for this cluster
            cluster_label = current_df.loc[subset_indices[0], 'agglomerative_label'] if 'agglomerative_label' in current_df.columns else None
            assigned_col_idx = choice_to_idx.get(cluster_label, -1) if cluster_label else -1

            pzx_subset = PZX[pzx_positions]
            ll_subset = row_log_px_given_z[pzx_positions]
            try:
                labels_split = split_cluster_by_posteriors(
                    pzx_subset, ll_subset, assigned_col_idx,
                )
            except Exception as e:
                logging.error(f"Posterior-based split failed for cluster {cid_to_split}: {e}. Skipping split.")
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
    return current_df, next_new_cluster_id, split_pairs, merge_groups


def em_schema_refinement(
        merged_df: pd.DataFrame,
        prob_calibrator: ProbabilityCalibrator, 
        choices_list: list[str], 
        text_column_name: str,
        model_identifier: str, 
        model_type: str, 
        max_iters: int = 3, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",  # DEPRECATED: unused in LLM-only mode
        baseline_sample_size: int = DEFAULT_BASELINE_SAMPLE_SIZE,
        verbose: bool = False,
        max_texts_per_cluster_metrics: int = DEFAULT_MAX_TEXTS_PER_CLUSTER_METRICS,
        show_progress: bool = False,
        log_iteration_metrics: bool = False,
        log_top_pairs: int = 0,
        diagnostics_sample_size: int = 200,
        diagnostics_bins: int = 40,
        diagnostics_dir: Path | None = None,
        config: SchemaRefinementConfig | None = None,
): 
    """Run an EM-like schema refinement loop, returning updated merged_df and history of schema changes."""
    if config is None:
        config = SchemaRefinementConfig()
    if config.accept_metric not in VALID_ACCEPT_METRICS:
        raise ValueError(
            f"Invalid accept_metric '{config.accept_metric}'. Expected one of {VALID_ACCEPT_METRICS}."
        )
    if config.structural_label_mode not in VALID_STRUCTURAL_LABEL_MODES:
        raise ValueError(
            f"Invalid structural_label_mode '{config.structural_label_mode}'. "
            f"Expected one of {VALID_STRUCTURAL_LABEL_MODES}."
        )
    logging.info(
        "Operation config: tune_operation=%s | split=%s merge=%s remove=%s add=%s revise=%s | "
        "best_operation_per_iteration=%s max_same_operation_streak=%s | structural_label_mode=%s split_label_pair_attempts=%s llm_label_max_attempts=%s",
        config.tune_operation,
        config.operation_enabled("split"),
        config.operation_enabled("merge"),
        config.operation_enabled("remove"),
        config.operation_enabled("add"),
        config.operation_enabled("revise"),
        config.best_operation_per_iteration,
        config.max_same_operation_streak,
        config.structural_label_mode,
        config.split_label_pair_attempts,
        config.llm_label_max_attempts,
    )
    logging.info(f"Starting EM-style schema refinement for {max_iters} iterations (LLM-only mode, no sentence-transformer embeddings)...")

    # Build index mapping: DataFrame index → 0-based row position (stable across iterations)
    idx_to_pzx_row_map = {original_idx: pos for pos, original_idx in enumerate(merged_df.index)}

    current_merged_df = merged_df.copy()
    schema_state = build_schema_state_from_df(current_merged_df)
    current_merged_df['agglomerative_label'] = current_merged_df['cluster_id'].map(schema_state.label_map)
    current_choices_list = list(schema_state.choices_list)
    current_prob_calibrator = prob_calibrator
    token_count_tokenizer = getattr(current_prob_calibrator, "tokenizer", None) or getattr(current_prob_calibrator, "vllm_tokenizer", None)
    if token_count_tokenizer is None and model_type == "hf":
        try:
            token_count_tokenizer = AutoTokenizer.from_pretrained(model_identifier)
            logging.info("Loaded tokenizer '%s' for token-count-aware metrics.", model_identifier)
        except Exception as exc:
            logging.warning("Could not load tokenizer for token-count-aware metrics: %s. Falling back to whitespace token counts.", exc)
    elif token_count_tokenizer is None:
        logging.warning("No model tokenizer handle available for token counts. Falling back to whitespace token counts.")

    def _compute_variant_metrics(
        df_for_metrics: pd.DataFrame,
        schema_for_metrics: SchemaState,
        *,
        choices_override: list[str] | None = None,
        calibrator_override: ProbabilityCalibrator | None = None,
        tokenizer_override=None,
    ):
        choices_local = choices_override if choices_override is not None else current_choices_list
        calibrator_local = calibrator_override if calibrator_override is not None else current_prob_calibrator
        tokenizer_local = tokenizer_override if tokenizer_override is not None else token_count_tokenizer
        doc_scores_local = compute_document_level_scores(
            df=df_for_metrics,
            text_col=text_column_name,
            cluster_col="agglomerative_label",
            choices=choices_local,
            prob_calibrator=calibrator_local,
            embeddings=None,  # LLM-only mode: no sentence-transformer embeddings
            p_z_prior=schema_for_metrics.p_z_prior,
        )
        token_counts_local = estimate_token_counts(
            df_for_metrics[text_column_name].astype(str).tolist(),
            tokenizer=tokenizer_local,
        )
        corpus_scores_local = compute_corpus_level_scores_variants(
            doc_scores=doc_scores_local,
            token_counts=token_counts_local,
            k_complexity=len(choices_local),
            is_test_mask=None,
        )
        return doc_scores_local, token_counts_local, corpus_scores_local

    def _reassign_removed_datapoints(
        post_update_df: pd.DataFrame,
        pre_update_df: pd.DataFrame,
        removed_cluster_ids: list[int],
        base_calibrator: ProbabilityCalibrator,
        *,
        iter_number: int | None = None,
        log_details: bool = False,
    ) -> pd.DataFrame:
        """Reassign datapoints from removed clusters via argmax p(z|x) over remaining labels."""
        if not removed_cluster_ids:
            return post_update_df
        removed_rows = pre_update_df[pre_update_df["cluster_id"].isin(removed_cluster_ids)].copy()
        if removed_rows.empty:
            return post_update_df
        if post_update_df.empty:
            logging.warning(
                "Removed %s clusters but no remaining clusters to reassign %s datapoints; keeping current dataframe as-is.",
                len(removed_cluster_ids),
                len(removed_rows),
            )
            return post_update_df

        remaining_cluster_ids = sorted(post_update_df["cluster_id"].unique())
        remaining_label_map = (
            post_update_df.groupby("cluster_id")["agglomerative_label"].first().astype(str).to_dict()
        )
        unique_label_map, reassignment_choices = build_unique_label_map(
            remaining_cluster_ids,
            remaining_label_map,
            fallback_prefix="cluster",
        )
        choice_to_cluster_id = {
            build_label_string(unique_label_map[cid]): cid
            for cid in remaining_cluster_ids
        }

        reassignment_calibrator = base_calibrator
        try:
            if list(getattr(base_calibrator, "choices", [])) != list(reassignment_choices):
                reassignment_calibrator = rebuild_calibrator_with_existing_backend(
                    base_calibrator,
                    reassignment_choices,
                    verbose=False,
                )
        except Exception as exc:
            logging.warning(
                "Failed to rebuild calibrator for remove-reassignment choices; falling back to existing choices. Error: %s",
                exc,
            )

        # Keep prior aligned to remaining cluster proportions during reassignment.
        counts_by_cluster = post_update_df["cluster_id"].value_counts().to_dict()
        reassignment_prior = np.array(
            [counts_by_cluster.get(cid, 0) for cid in remaining_cluster_ids],
            dtype=float,
        )
        if reassignment_prior.sum() > 0:
            reassignment_prior = reassignment_prior / reassignment_prior.sum()
            try:
                reassignment_calibrator.p_z = reassignment_prior
            except Exception:
                pass

        reassigned_rows = removed_rows.copy()
        reassignment_events: list[dict] = []
        for row_idx, row in reassigned_rows.iterrows():
            old_cluster_id = int(row["cluster_id"])
            sentence_text = str(row[text_column_name])
            pzx = np.asarray(reassignment_calibrator.calibrate_p_z_given_X(sentence_text), dtype=float)
            if pzx.size == 0:
                target_cid = remaining_cluster_ids[0]
            else:
                best_choice = reassignment_choices[int(np.argmax(pzx))]
                target_cid = choice_to_cluster_id.get(best_choice, remaining_cluster_ids[0])
            reassigned_rows.at[row_idx, "cluster_id"] = target_cid
            reassigned_rows.at[row_idx, "agglomerative_label"] = remaining_label_map.get(
                target_cid,
                f"cluster_{target_cid}",
            )
            reassignment_events.append(
                {
                    "row_index": row_idx,
                    "old_cluster_id": old_cluster_id,
                    "new_cluster_id": int(target_cid),
                    "new_cluster_label": remaining_label_map.get(target_cid, f"cluster_{target_cid}"),
                    "sentence_preview": sentence_text[:200],
                }
            )

        combined = pd.concat([post_update_df, reassigned_rows], axis=0, sort=False)
        combined = combined.sort_index()
        logging.info(
            "Reassigned %s datapoints from removed clusters %s to remaining clusters.",
            len(reassigned_rows),
            sorted(set(int(x) for x in removed_cluster_ids)),
        )
        if reassignment_events:
            events_df = pd.DataFrame(reassignment_events)
            transition_counts = (
                events_df.groupby(["old_cluster_id", "new_cluster_id", "new_cluster_label"], dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values(["old_cluster_id", "count", "new_cluster_id"], ascending=[True, False, True])
            )
            for rec in transition_counts.to_dict(orient="records"):
                logging.info(
                    "  Remove reassignment: old_cluster=%s -> new_cluster=%s ('%s') count=%s",
                    rec["old_cluster_id"],
                    rec["new_cluster_id"],
                    rec["new_cluster_label"],
                    rec["count"],
                )
            if log_details:
                for rec in events_df.to_dict(orient="records"):
                    logging.info(
                        "  Remove reassignment detail: row=%s old_cluster=%s -> new_cluster=%s ('%s') text='%s'",
                        rec["row_index"],
                        rec["old_cluster_id"],
                        rec["new_cluster_id"],
                        rec["new_cluster_label"],
                        rec["sentence_preview"],
                    )
            if diagnostics_dir is not None and iter_number is not None:
                try:
                    out_path = diagnostics_dir / f"remove_reassignment_iter_{int(iter_number):02d}.csv"
                    events_df.to_csv(out_path, index=False)
                    logging.info("Saved remove reassignment details to: %s", out_path)
                except Exception as exc:
                    logging.warning("Failed to save remove reassignment details CSV: %s", exc)
        return combined

    schema_history = []
    if current_merged_df['cluster_id'].empty:
        logging.warning("Initial merged_df has no cluster_id assignments. Starting new cluster IDs from 0.")
        next_new_cluster_id = 0
    else:
        next_new_cluster_id = current_merged_df['cluster_id'].max() + 1
    logging.info("Calculating baseline log p(x) for the dataset...")
    sample_size_for_baseline = min(baseline_sample_size, len(current_merged_df))
    baseline_log_px = 0.0 # Initialize baseline_log_px
    accepted_avg_logL_per_token_by_variant = {m: None for m in VALID_ACCEPT_METRICS}
    if sample_size_for_baseline > 0:
        texts_for_baseline_sample = current_merged_df[text_column_name].sample(sample_size_for_baseline).tolist()
        baseline_log_px = current_prob_calibrator.compute_average_log_px_from_sample(texts_for_baseline_sample)
        
        logging.info(f"Baseline log p(x) = {baseline_log_px:.4f} (based on {sample_size_for_baseline} samples)")

        # NEW: E-step scoring
        doc_scores, token_counts, corpus_scores = _compute_variant_metrics(
            current_merged_df,
            schema_state,
        )
        # Persist per-row pmax / assignments for analysis
        current_merged_df["pmax_explained"] = doc_scores["row_pmax_posterior"]     
        current_merged_df["z_hat_explainer"] = doc_scores["row_z_hat"]   

        logging.info(f"[E-step] L_baseline={doc_scores['L_baseline']:.4f}, "
                    f"logL_total_oracle={corpus_scores['logL_total_oracle']:.2f}, "
                    f"avg_logL_tok_oracle={corpus_scores['avg_logL_per_token_oracle']:.6f}, "
                    f"avg_logL_tok_assigned={corpus_scores['avg_logL_per_token_assigned']:.6f}, "
                    f"avg_logL_tok_soft={corpus_scores['avg_logL_per_token_soft']:.6f}, "
                    f"AIC={corpus_scores['AIC']}, BIC={corpus_scores['BIC']}, "
                    f"PPL_tok_oracle={corpus_scores['perplexity_token_oracle']:.3f}, "
                    f"PPL_tok_assigned={corpus_scores['perplexity_token_assigned']:.3f}, "
                    f"PPL_tok_soft={corpus_scores['perplexity_token_soft']:.3f}, "
                    f"ELBO={corpus_scores['ELBO']}")  
        for variant in VALID_ACCEPT_METRICS:
            accepted_avg_logL_per_token_by_variant[variant] = float(
                corpus_scores[f"avg_logL_per_token_{variant}"]
            )
    else:
        logging.warning(f"  Dataset is empty or too small for baseline sample, cannot compute baseline log p(x). Using {baseline_log_px}.")

    split_cooldown: dict[int, int] = {}
    revise_cooldown: dict[int, int] = {}
    add_cooldown_until = 0
    no_op_iters = 0
    last_selected_operation: str | None = None
    same_operation_streak = 0
    iteration_metrics_rows: list[dict] = []
    if config.best_operation_per_iteration and config.tune_operation != "all":
        logging.warning(
            "best_operation_per_iteration is enabled but tune_operation=%s (not 'all'); best-op mode will be ignored.",
            config.tune_operation,
        )

    def _simulate_operation_delta(
        op_name: str,
        op_actions: dict,
        pre_df: pd.DataFrame,
        pre_schema_state: SchemaState,
        pre_choices: list[str],
        pre_prob_calibrator: ProbabilityCalibrator,
        pre_next_new_cluster_id: int,
        baseline_accept_metric_value: float,
    ) -> float | None:
        """
        Fast structural simulation to score a single candidate operation from the same pre-iteration state.
        This deliberately skips expensive relabeling/add-step and scores post-schema assignment quality directly.
        """
        if op_name not in {"split", "merge", "remove", "revise"}:
            return None
        if not op_actions.get(op_name):
            return None
        try:
            sim_df, _, _, _ = apply_schema_updates(
                pre_df.copy(),
                iter_PZX,
                iter_log_px_given_z,
                idx_to_pzx_row_map,
                current_choices_list,
                op_actions,
                pre_next_new_cluster_id,
            )
            if op_actions.get("remove"):
                sim_df = _reassign_removed_datapoints(
                    post_update_df=sim_df,
                    pre_update_df=pre_df,
                    removed_cluster_ids=list(op_actions.get("remove", [])),
                    base_calibrator=pre_prob_calibrator,
                )
            if sim_df.empty:
                return None
            _rename_duplicate_labels(sim_df, text_column_name)
            _enforce_unique_choice_labels(sim_df)
            sim_schema = build_schema_state_from_df(sim_df)
            sim_df["agglomerative_label"] = sim_df["cluster_id"].map(sim_schema.label_map)
            sim_choices = list(sim_schema.choices_list)
            sim_calibrator = pre_prob_calibrator
            if list(pre_choices) != list(sim_choices):
                sim_calibrator = rebuild_calibrator_with_existing_backend(
                    pre_prob_calibrator,
                    sim_choices,
                    verbose=False,
                )
            if sim_schema.p_z_prior is not None:
                sim_calibrator.p_z = sim_schema.p_z_prior
            sim_doc_scores = compute_document_level_scores(
                df=sim_df,
                text_col=text_column_name,
                cluster_col="agglomerative_label",
                choices=sim_choices,
                prob_calibrator=sim_calibrator,
                embeddings=None,  # LLM-only mode: no sentence-transformer embeddings
                p_z_prior=sim_schema.p_z_prior,
            )
            sim_token_counts = estimate_token_counts(
                sim_df[text_column_name].astype(str).tolist(),
                tokenizer=token_count_tokenizer,
            )
            sim_corpus_scores = compute_corpus_level_scores_variants(
                doc_scores=sim_doc_scores,
                token_counts=sim_token_counts,
                k_complexity=len(sim_choices),
                is_test_mask=None,
            )
            accept_metric_key = f"avg_logL_per_token_{config.accept_metric}"
            return float(sim_corpus_scores[accept_metric_key] - baseline_accept_metric_value)
        except Exception as exc:
            logging.warning(
                "[Iter %s] Candidate simulation failed for op '%s': %s",
                iter_idx + 1 if 'iter_idx' in locals() else -1,
                op_name,
                exc,
            )
            return None

    def _execute_candidate_full_pipeline(
        *,
        candidate_op_name: str,
        candidate_actions: dict,
        iter_number: int,
        pre_df: pd.DataFrame,
        pre_schema_state: SchemaState,
        pre_choices: list[str],
        pre_prob_calibrator: ProbabilityCalibrator,
        pre_tokenizer,
        pre_next_new_cluster_id: int,
        pre_split_cooldown: dict[int, int],
        pre_revise_cooldown: dict[int, int],
        pre_add_cooldown_until: int,
        baseline_accept_metric_value: float,
        pre_corpus_scores: dict | None,
    ) -> dict | None:
        """
        Execute a full candidate op path on a copied state and return its actual post-score delta.
        This mirrors the real execution path (apply/relabel/add/schema rebuild/re-score).
        """
        try:
            cand_df = pre_df.copy()
            cand_schema_state = pre_schema_state
            cand_choices = list(pre_choices)
            cand_next_new_cluster_id = int(pre_next_new_cluster_id)
            cand_split_cooldown = dict(pre_split_cooldown)
            cand_revise_cooldown = dict(pre_revise_cooldown)
            cand_add_cooldown_until = int(pre_add_cooldown_until)
            cand_tokenizer = pre_tokenizer

            try:
                cand_prob_calibrator = rebuild_calibrator_with_existing_backend(
                    pre_prob_calibrator,
                    cand_choices,
                    verbose=False,
                )
            except Exception:
                cand_prob_calibrator = pre_prob_calibrator

            cand_iteration_actions = {
                "split": list(candidate_actions.get("split", [])),
                "merge": list(candidate_actions.get("merge", [])),
                "remove": list(candidate_actions.get("remove", [])),
                "revise": list(candidate_actions.get("revise", [])),
                "selected_operation": candidate_op_name,
            }

            cand_df, cand_next_new_cluster_id, split_pairs, merge_groups = apply_schema_updates(
                cand_df,
                iter_PZX,
                iter_log_px_given_z,
                idx_to_pzx_row_map,
                pre_choices,
                {
                    "split": list(cand_iteration_actions.get("split", [])),
                    "merge": list(cand_iteration_actions.get("merge", [])),
                    "remove": list(cand_iteration_actions.get("remove", [])),
                    "revise": list(cand_iteration_actions.get("revise", [])),
                },
                cand_next_new_cluster_id,
            )

            if cand_iteration_actions.get("remove"):
                cand_df = _reassign_removed_datapoints(
                    post_update_df=cand_df,
                    pre_update_df=pre_df,
                    removed_cluster_ids=list(cand_iteration_actions.get("remove", [])),
                    base_calibrator=cand_prob_calibrator,
                    iter_number=None,
                    log_details=False,
                )

            split_child_ids = {child for _, child in split_pairs}
            if split_pairs and config.operation_enabled("split"):
                shared_vllm_tokenizer = getattr(cand_prob_calibrator, "vllm_tokenizer", None)
                shared_vllm_model = getattr(cand_prob_calibrator, "vllm_model", None)
                kept_split_pairs: list[tuple[int, int]] = []
                for parent_id, child_id in split_pairs:
                    parent_label = pre_schema_state.label_map.get(parent_id, f"cluster_{parent_id}")
                    parent_texts = cand_df.loc[cand_df["cluster_id"] == parent_id, text_column_name].astype(str).tolist()
                    child_texts = cand_df.loc[cand_df["cluster_id"] == child_id, text_column_name].astype(str).tolist()
                    parent_samples = _sample_texts_for_labeling(parent_texts, max_texts=10, seed=int(parent_id))
                    child_samples = _sample_texts_for_labeling(child_texts, max_texts=10, seed=int(child_id))
                    existing_forbidden_names = _collect_existing_cluster_labels(
                        cand_df,
                        exclude_cluster_ids={int(parent_id), int(child_id)},
                    )

                    if config.structural_label_mode == "parent_relative":
                        new_parent_label, new_child_label = _build_parent_relative_split_labels(
                            parent_label=parent_label,
                            parent_id=int(parent_id),
                        )
                    elif model_type == "vllm":
                        new_parent_label, new_child_label = _propose_split_labels_with_retries(
                            parent_samples=parent_samples,
                            child_samples=child_samples,
                            parent_label=parent_label,
                            model_identifier=model_identifier,
                            tokenizer=shared_vllm_tokenizer,
                            model=shared_vllm_model,
                            context_prefix=f"candidate:{candidate_op_name}:split:{parent_id}->{child_id}",
                            existing_forbidden_names=existing_forbidden_names,
                            pair_attempts=config.split_label_pair_attempts,
                            llm_label_max_attempts=config.llm_label_max_attempts,
                        )
                    else:
                        new_parent_label, new_child_label = _build_parent_relative_split_labels(
                            parent_label=parent_label,
                            parent_id=int(parent_id),
                        )

                    parent_key = _canonical_label_key(new_parent_label)
                    child_key = _canonical_label_key(new_child_label)
                    pair_near_dupe = False
                    if parent_key and child_key:
                        pair_near_dupe = bool(
                            detect_near_duplicate_labels(
                                [new_parent_label, new_child_label],
                                threshold=DUP_LABEL_SIM_THRESHOLD,
                            )
                        )
                    existing_label_keys = {
                        _canonical_label_key(lbl)
                        for cid, lbl in cand_df.groupby("cluster_id")["agglomerative_label"].first().items()
                        if int(cid) not in {int(parent_id), int(child_id)}
                    }
                    collides_existing_label = bool(
                        (parent_key and parent_key in existing_label_keys)
                        or (child_key and child_key in existing_label_keys)
                    )

                    if (
                        not new_parent_label
                        or not new_child_label
                        or not parent_key
                        or not child_key
                        or parent_key == child_key
                        or pair_near_dupe
                        or collides_existing_label
                    ):
                        logging.warning(
                            "[Iter %s] [Candidate %s] Reverting split (%s -> %s): invalid/distinct labels failed "
                            "(parent='%s', child='%s', collides_existing=%s).",
                            iter_number,
                            candidate_op_name,
                            parent_id,
                            child_id,
                            new_parent_label,
                            new_child_label,
                            collides_existing_label,
                        )
                        cand_df.loc[cand_df["cluster_id"] == child_id, "cluster_id"] = parent_id
                        cand_df.loc[cand_df["cluster_id"] == parent_id, "agglomerative_label"] = parent_label
                        continue

                    cand_df.loc[cand_df["cluster_id"] == parent_id, "agglomerative_label"] = new_parent_label
                    cand_df.loc[cand_df["cluster_id"] == child_id, "agglomerative_label"] = new_child_label
                    kept_split_pairs.append((parent_id, child_id))
                split_pairs = kept_split_pairs
                split_child_ids = {child for _, child in split_pairs}
                cand_iteration_actions["split"] = [p for p, _ in split_pairs]

            if merge_groups and config.operation_enabled("merge"):
                shared_vllm_tokenizer = getattr(cand_prob_calibrator, "vllm_tokenizer", None)
                shared_vllm_model = getattr(cand_prob_calibrator, "vllm_model", None)
                for root_id, member_ids in merge_groups:
                    if root_id not in set(cand_df["cluster_id"].unique()):
                        continue
                    source_labels = [
                        pre_schema_state.label_map.get(mid, f"cluster_{mid}")
                        for mid in member_ids
                    ]
                    merged_texts = cand_df.loc[
                        cand_df["cluster_id"] == root_id, text_column_name
                    ].astype(str).tolist()
                    merged_samples = _sample_texts_for_labeling(merged_texts, max_texts=12, seed=int(root_id))
                    new_merge_label = None
                    existing_forbidden_names = _collect_existing_cluster_labels(
                        cand_df,
                        exclude_cluster_ids={int(root_id)},
                    )
                    merge_banned_names = list(existing_forbidden_names) + list(source_labels)
                    if config.structural_label_mode == "parent_relative":
                        new_merge_label = _build_parent_relative_merge_label(
                            source_labels=[str(x) for x in source_labels],
                            root_id=int(root_id),
                        )
                    elif model_type == "vllm":
                        llm_label = _label_cluster_vllm(
                            merged_samples,
                            model_identifier,
                            tokenizer=shared_vllm_tokenizer,
                            model=shared_vllm_model,
                            parent_label=" + ".join(base_label(lbl) for lbl in source_labels[:2]),
                            banned_names=merge_banned_names,
                            context=f"candidate:{candidate_op_name}:merge_cluster:{root_id}",
                            max_attempts=config.llm_label_max_attempts,
                        )
                        if llm_label:
                            name, _, keywords = llm_label
                            new_merge_label = build_label_string(name, keywords)
                    if not new_merge_label:
                        new_merge_label = _fallback_distinct_label_from_texts(
                            merged_samples,
                            banned_names=merge_banned_names,
                        )
                        if not new_merge_label:
                            name, keywords = propose_label_and_keywords_from_texts(merged_samples, parent_label=None)
                            new_merge_label = build_label_string(name, keywords)
                    cand_df.loc[cand_df["cluster_id"] == root_id, "agglomerative_label"] = new_merge_label

            revised_clusters_applied: list[int] = []
            if cand_iteration_actions.get("revise") and config.operation_enabled("revise"):
                shared_vllm_tokenizer = getattr(cand_prob_calibrator, "vllm_tokenizer", None)
                shared_vllm_model = getattr(cand_prob_calibrator, "vllm_model", None)
                current_cluster_ids = set(cand_df["cluster_id"].unique())
                for cid in cand_iteration_actions["revise"]:
                    if cid not in current_cluster_ids:
                        continue
                    current_label = str(cand_df.loc[cand_df["cluster_id"] == cid, "agglomerative_label"].iloc[0])
                    revise_texts = cand_df.loc[cand_df["cluster_id"] == cid, text_column_name].astype(str).tolist()
                    revise_samples = _sample_texts_for_labeling(revise_texts, max_texts=12, seed=int(cid))
                    new_label = None
                    if model_type == "vllm":
                        llm_label = _label_cluster_vllm(
                            revise_samples,
                            model_identifier,
                            tokenizer=shared_vllm_tokenizer,
                            model=shared_vllm_model,
                            parent_label=base_label(current_label),
                            banned_names=[current_label],
                            context=f"candidate:{candidate_op_name}:revise_cluster:{cid}",
                            max_attempts=config.llm_label_max_attempts,
                        )
                        if llm_label:
                            name, _, keywords = llm_label
                            new_label = build_label_string(name, keywords)
                    if not new_label:
                        new_label = _fallback_distinct_label_from_texts(revise_samples, banned_names=[current_label])
                        if not new_label:
                            name, keywords = propose_label_and_keywords_from_texts(
                                revise_samples,
                                parent_label=base_label(current_label),
                            )
                            new_label = build_label_string(name, keywords)
                    cand_df.loc[cand_df["cluster_id"] == cid, "agglomerative_label"] = new_label
                    revised_clusters_applied.append(int(cid))
                cand_iteration_actions["revise"] = revised_clusters_applied
                for cid in revised_clusters_applied:
                    cand_revise_cooldown[cid] = iter_number + config.revise_cooldown_iters

            dup_renamed = _rename_duplicate_labels(cand_df, text_column_name)

            duplicates_prevented = False
            if split_pairs:
                still_kept_pairs: list[tuple[int, int]] = []
                for parent_id, child_id in split_pairs:
                    if child_id not in set(cand_df["cluster_id"].unique()):
                        continue
                    parent_label_now = str(cand_df.loc[cand_df["cluster_id"] == parent_id, "agglomerative_label"].iloc[0])
                    child_label_now = str(cand_df.loc[cand_df["cluster_id"] == child_id, "agglomerative_label"].iloc[0])
                    pair_dup = bool(
                        detect_near_duplicate_labels(
                            [parent_label_now, child_label_now],
                            threshold=DUP_LABEL_SIM_THRESHOLD,
                        )
                    )
                    if pair_dup or _canonical_label_key(parent_label_now) == _canonical_label_key(child_label_now):
                        duplicates_prevented = True
                        cand_df.loc[cand_df["cluster_id"] == child_id, "cluster_id"] = parent_id
                    else:
                        still_kept_pairs.append((parent_id, child_id))
                split_pairs = still_kept_pairs
                split_child_ids = {child for _, child in split_pairs}
                cand_iteration_actions["split"] = [p for p, _ in split_pairs]
                cand_df["agglomerative_label"] = cand_df["cluster_id"].map(
                    cand_df.groupby("cluster_id")["agglomerative_label"].first().to_dict()
                )
            if split_pairs:
                for parent_id, _ in split_pairs:
                    cand_split_cooldown[parent_id] = iter_number + config.split_cooldown_iters

            add_prev_cluster_ids = None
            schema_unstable = bool(
                cand_iteration_actions.get("split")
                or cand_iteration_actions.get("merge")
                or cand_iteration_actions.get("remove")
                or duplicates_prevented
            )
            added_clusters_info = []
            if not config.operation_enabled("add"):
                pass
            elif iter_number <= cand_add_cooldown_until:
                pass
            else:
                poor_indices = []
                pmax_vals = []
                entropy_vals = []
                assigned_ranks = []
                all_indices = []
                for original_idx in cand_df.index:
                    sentence_text = cand_df.loc[original_idx, text_column_name]
                    probabilities = cand_prob_calibrator.calibrate_p_z_given_X(sentence_text)
                    pmax = float(np.max(probabilities)) if hasattr(probabilities, "__len__") and len(probabilities) > 0 else 0.0
                    pmax_vals.append(pmax)
                    all_indices.append(original_idx)

                    diag = compute_posterior_diagnostics(
                        pzx=np.asarray(probabilities, dtype=float),
                        choices_list=cand_choices,
                        p_z_prior=(
                            cand_schema_state.p_z_prior
                            if cand_schema_state.p_z_prior is not None
                            else np.ones(len(cand_choices)) / max(1, len(cand_choices))
                        ),
                        assigned_label=cand_df.loc[original_idx, "agglomerative_label"],
                    )
                    entropy_vals.append(diag["entropy"])
                    assigned_ranks.append(diag["assigned_rank"])

                poor_pos, _ = _select_poor_candidates(
                    pmax_vals,
                    entropy_vals,
                    assigned_ranks,
                    config.add_min_poorly_explained,
                    config.add_low_confidence_max,
                    config.add_low_confidence_quantile,
                    config.add_entropy_min,
                )
                poor_indices = [all_indices[i] for i in poor_pos]

                if len(cand_choices) >= config.add_max_total_clusters:
                    poor_indices = []

                if len(poor_indices) >= config.add_min_poorly_explained:
                    # Collect posterior vectors for poorly-explained docs
                    valid_poor_indices = list(poor_indices)  # all indices are valid (no embedding map needed)
                    poor_pzx = np.vstack([
                        cand_prob_calibrator.calibrate_p_z_given_X(
                            cand_df.loc[idx, text_column_name]
                        )
                        for idx in valid_poor_indices
                    ]) if valid_poor_indices else np.empty((0, len(cand_choices)))

                    if len(valid_poor_indices) >= config.add_min_group_size:
                        prev_cluster_ids = cand_df.loc[valid_poor_indices, "cluster_id"].copy()
                        add_prev_cluster_ids = prev_cluster_ids
                        num_new_clusters_to_add = max(1, len(valid_poor_indices) // config.add_items_per_new_cluster)
                        num_new_clusters_to_add = min(num_new_clusters_to_add, config.add_max_new_clusters_per_iter)
                        from sklearn.cluster import KMeans
                        if len(poor_pzx) < num_new_clusters_to_add:
                            num_new_clusters_to_add = len(poor_pzx)
                        try:
                            created = 0
                            for k_try in [min(2, num_new_clusters_to_add), min(3, num_new_clusters_to_add)]:
                                if k_try < 2:
                                    continue
                                kmeans_poor = KMeans(n_clusters=k_try, n_init='auto', random_state=42)
                                new_labels_for_poor_texts = kmeans_poor.fit_predict(poor_pzx)
                                for k_new_cluster in range(k_try):
                                    new_cluster_original_indices = [
                                        valid_poor_indices[i]
                                        for i, lab in enumerate(new_labels_for_poor_texts)
                                        if lab == k_new_cluster
                                    ]
                                    if len(new_cluster_original_indices) < config.add_min_group_size:
                                        continue
                                    # KL divergence cohesion check on posteriors
                                    group_pzx = poor_pzx[[
                                        valid_poor_indices.index(i) for i in new_cluster_original_indices
                                    ]]
                                    cohesion = kl_divergence_cohesion(group_pzx)
                                    if cohesion < config.add_cohesion_min:
                                        continue
                                    assigned_new_id = cand_next_new_cluster_id
                                    cand_next_new_cluster_id += 1
                                    cand_df.loc[new_cluster_original_indices, "cluster_id"] = assigned_new_id
                                    texts = cand_df.loc[new_cluster_original_indices, text_column_name].astype(str).tolist()
                                    new_label = None
                                    if model_type == "vllm":
                                        llm_label = _label_cluster_vllm(
                                            texts,
                                            model_identifier,
                                            tokenizer=getattr(cand_prob_calibrator, "vllm_tokenizer", None),
                                            model=getattr(cand_prob_calibrator, "vllm_model", None),
                                            context=f"candidate:{candidate_op_name}:add_cluster:{assigned_new_id}",
                                            max_attempts=config.llm_label_max_attempts,
                                        )
                                        if llm_label:
                                            name, _, keywords = llm_label
                                            new_label = build_label_string(name, keywords)
                                    if not new_label:
                                        name, keywords = propose_label_and_keywords_from_texts(texts, parent_label=None)
                                        new_label = build_label_string(name, keywords)
                                    cand_df.loc[new_cluster_original_indices, "agglomerative_label"] = new_label
                                    added_clusters_info.append({"id": assigned_new_id, "size": len(new_cluster_original_indices)})
                                    created += 1
                                    if created >= config.add_max_new_clusters_per_iter:
                                        break
                                if created > 0:
                                    break
                            if created == 0:
                                cand_df.loc[valid_poor_indices, "cluster_id"] = prev_cluster_ids.values
                                added_clusters_info = []
                        except Exception:
                            pass

            if added_clusters_info:
                labels_now = cand_df.groupby("cluster_id")["agglomerative_label"].first().astype(str).tolist()
                dup_pairs = detect_near_duplicate_labels(labels_now, threshold=DUP_LABEL_SIM_THRESHOLD)
                if dup_pairs:
                    if add_prev_cluster_ids is not None:
                        cand_df.loc[add_prev_cluster_ids.index, "cluster_id"] = add_prev_cluster_ids.values
                    added_clusters_info = []
                else:
                    cand_add_cooldown_until = iter_number + config.add_cooldown_iters

            cand_iteration_actions["add"] = added_clusters_info

            dup_renamed += _rename_duplicate_labels(cand_df, text_column_name)
            forced_unique = _enforce_unique_choice_labels(cand_df)
            if forced_unique > 0:
                logging.info(
                    "[Iter %s] [Candidate %s] enforced unique schema labels for %s clusters.",
                    iter_number,
                    candidate_op_name,
                    forced_unique,
                )

            cand_schema_state = build_schema_state_from_df(cand_df)
            cand_df["agglomerative_label"] = cand_df["cluster_id"].map(cand_schema_state.label_map)
            cand_choices = list(cand_schema_state.choices_list)
            if list(cand_prob_calibrator.choices) != list(cand_choices):
                try:
                    cand_prob_calibrator = rebuild_calibrator_with_existing_backend(
                        cand_prob_calibrator,
                        cand_choices,
                        verbose=False,
                    )
                except Exception:
                    cand_prob_calibrator = initialize_probability_calibrator(
                        model_identifier=model_identifier,
                        model_type=model_type,
                        choices=cand_choices,
                        num_trials=cand_prob_calibrator.num_trials,
                        scorer_type="batch" if cand_prob_calibrator.batch_prompts is False else "single",
                        verbose=False,
                    )

            cand_tokenizer = (
                getattr(cand_prob_calibrator, "tokenizer", None)
                or getattr(cand_prob_calibrator, "vllm_tokenizer", None)
                or cand_tokenizer
            )
            if cand_schema_state.p_z_prior is not None:
                cand_prob_calibrator.p_z = cand_schema_state.p_z_prior

            post_doc_scores, _, post_corpus_scores = _compute_variant_metrics(
                cand_df,
                cand_schema_state,
                choices_override=cand_choices,
                calibrator_override=cand_prob_calibrator,
                tokenizer_override=cand_tokenizer,
            )
            accept_metric_key = f"avg_logL_per_token_{config.accept_metric}"
            metric_delta = float(post_corpus_scores[accept_metric_key] - baseline_accept_metric_value)
            prepost_delta = (
                float(post_corpus_scores[accept_metric_key] - pre_corpus_scores[accept_metric_key])
                if pre_corpus_scores is not None
                else np.nan
            )
            proposed_action_count = int(
                len(cand_iteration_actions.get("split", []))
                + len(cand_iteration_actions.get("merge", []))
                + len(cand_iteration_actions.get("remove", []))
                + len(cand_iteration_actions.get("add", []))
                + len(cand_iteration_actions.get("revise", []))
            )

            return {
                "candidate_op_name": candidate_op_name,
                "iteration_actions": cand_iteration_actions,
                "merged_df": cand_df,
                "schema_state": cand_schema_state,
                "choices_list": cand_choices,
                "prob_calibrator": cand_prob_calibrator,
                "tokenizer": cand_tokenizer,
                "next_new_cluster_id": cand_next_new_cluster_id,
                "split_cooldown": cand_split_cooldown,
                "revise_cooldown": cand_revise_cooldown,
                "add_cooldown_until": cand_add_cooldown_until,
                "post_doc_scores": post_doc_scores,
                "post_corpus_scores": post_corpus_scores,
                "metric_delta": metric_delta,
                "prepost_delta": prepost_delta,
                "proposed_action_count": proposed_action_count,
                "cluster_count": int(cand_df["cluster_id"].nunique()),
            }
        except Exception as exc:
            logging.warning(
                "[Iter %s] Candidate op '%s' full execution failed: %s",
                iter_number,
                candidate_op_name,
                exc,
            )
            return None
    for iter_idx in range(max_iters):
        logging.info(f"=== EM Iteration {iter_idx + 1}/{max_iters} ===")
        iteration_actions = {}
        dup_renamed = 0
        if current_merged_df.empty:
            logging.warning(f"Iteration {iter_idx + 1}: Dataframe is empty. Stopping EM refinement.")
            break
        prev_merged_df = current_merged_df.copy()
        prev_schema_state = schema_state
        prev_choices_list = list(current_choices_list)
        prev_prob_calibrator = current_prob_calibrator
        prev_token_count_tokenizer = token_count_tokenizer
        prev_split_cooldown = dict(split_cooldown)
        prev_revise_cooldown = dict(revise_cooldown)
        prev_add_cooldown_until = add_cooldown_until
        prev_next_new_cluster_id = next_new_cluster_id
        should_compute_iter_metrics = log_iteration_metrics or config.rollback_on_metric_degrade
        # Always compute E-step (PZX, log p(x|z)) — needed for LLM-only structural operations
        pre_doc_scores_iter, _, pre_corpus_scores_iter = _compute_variant_metrics(
            current_merged_df,
            schema_state,
        )
        # Extract posterior and likelihood matrices for this iteration
        iter_PZX = pre_doc_scores_iter["PZX"]                        # (N, K)
        iter_log_px_given_z = pre_doc_scores_iter["row_log_px_given_z"]  # (N, K)

        if should_compute_iter_metrics:
            logging.info(
                "[Iter %s] PRE  avg_logL_tok: oracle=%.6f, assigned=%.6f, soft=%.6f | "
                "PPL_tok: oracle=%.3f, assigned=%.3f, soft=%.3f",
                iter_idx + 1,
                float(pre_corpus_scores_iter["avg_logL_per_token_oracle"]),
                float(pre_corpus_scores_iter["avg_logL_per_token_assigned"]),
                float(pre_corpus_scores_iter["avg_logL_per_token_soft"]),
                float(pre_corpus_scores_iter["perplexity_token_oracle"]),
                float(pre_corpus_scores_iter["perplexity_token_assigned"]),
                float(pre_corpus_scores_iter["perplexity_token_soft"]),
            )
        logging.info(f"Iteration {iter_idx + 1}: Computing cluster metrics (LLM-only) for {current_merged_df['cluster_id'].nunique()} clusters.")
        cluster_metrics = compute_cluster_metrics(
            current_merged_df,
            iter_PZX,
            iter_log_px_given_z,
            idx_to_pzx_row_map,
            current_choices_list,
            baseline_log_px,
            verbose=verbose,
            text_column=text_column_name,
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

        actions = decide_schema_updates(
            cluster_metrics,
            config,
            split_cooldown,
            revise_cooldown,
            iter_idx + 1,
            verbose=verbose,
            log_top_pairs=log_top_pairs,
        )
        selected_operation_for_iter = None
        if config.best_operation_per_iteration and config.tune_operation == "all":
            baseline_accept_metric_value = float(
                accepted_avg_logL_per_token_by_variant.get(config.accept_metric)
                if accepted_avg_logL_per_token_by_variant.get(config.accept_metric) is not None
                else pre_corpus_scores_iter[f"avg_logL_per_token_{config.accept_metric}"]
            )
            logging.info(
                "[Iter %s] Threshold-passing structural actions: split=%s merge=%s remove=%s",
                iter_idx + 1,
                len(actions.get("split", [])),
                len(actions.get("merge", [])),
                len(actions.get("remove", [])),
            )

            candidate_specs: list[tuple[str, dict]] = []
            for op_name in ("merge", "split", "remove"):
                op_payload = actions.get(op_name, [])
                if not op_payload:
                    logging.info("[Iter %s] Candidate op '%s' not created: no threshold-passing actions.", iter_idx + 1, op_name)
                    continue
                if (
                    config.max_same_operation_streak > 0
                    and last_selected_operation == op_name
                    and same_operation_streak >= int(config.max_same_operation_streak)
                ):
                    logging.info(
                        "[Iter %s] Candidate op '%s' skipped by max_same_operation_streak=%s (current streak=%s).",
                        iter_idx + 1,
                        op_name,
                        config.max_same_operation_streak,
                        same_operation_streak,
                    )
                    continue
                op_actions = {"split": [], "merge": [], "remove": [], "revise": []}
                op_actions[op_name] = list(op_payload)
                candidate_specs.append((op_name, op_actions))

            if config.operation_enabled("add"):
                candidate_specs.append(("add", {"split": [], "merge": [], "remove": [], "revise": []}))
            else:
                logging.info("[Iter %s] Candidate op 'add' not created: add disabled.", iter_idx + 1)

            candidate_results: list[dict] = []
            for op_name, op_actions in candidate_specs:
                logging.info(
                    "[Iter %s] Evaluating candidate op '%s' with actions=%s (full execution on copied state).",
                    iter_idx + 1,
                    op_name,
                    op_actions,
                )
                result = _execute_candidate_full_pipeline(
                    candidate_op_name=op_name,
                    candidate_actions=op_actions,
                    iter_number=iter_idx + 1,
                    pre_df=prev_merged_df,
                    pre_schema_state=prev_schema_state,
                    pre_choices=prev_choices_list,
                    pre_prob_calibrator=prev_prob_calibrator,
                    pre_tokenizer=prev_token_count_tokenizer,
                    pre_next_new_cluster_id=prev_next_new_cluster_id,
                    pre_split_cooldown=prev_split_cooldown,
                    pre_revise_cooldown=prev_revise_cooldown,
                    pre_add_cooldown_until=prev_add_cooldown_until,
                    baseline_accept_metric_value=baseline_accept_metric_value,
                    pre_corpus_scores=pre_corpus_scores_iter,
                )
                if result is None:
                    logging.info("[Iter %s] Candidate op '%s' failed during full execution.", iter_idx + 1, op_name)
                    continue
                candidate_results.append(result)
                logging.info(
                    "[Iter %s] Candidate op '%s' result: proposed_actions=%s clusters=%s delta_vs_best_%s=%.6f delta_post_pre_%s=%.6f",
                    iter_idx + 1,
                    op_name,
                    {
                        "split": len(result["iteration_actions"].get("split", [])),
                        "merge": len(result["iteration_actions"].get("merge", [])),
                        "remove": len(result["iteration_actions"].get("remove", [])),
                        "add": len(result["iteration_actions"].get("add", [])),
                        "revise": len(result["iteration_actions"].get("revise", [])),
                    },
                    result["cluster_count"],
                    config.accept_metric,
                    float(result["metric_delta"]),
                    config.accept_metric,
                    float(result["prepost_delta"]),
                )

            improving_candidates = [
                r for r in candidate_results
                if int(r.get("proposed_action_count", 0)) > 0
                and float(r.get("metric_delta", -np.inf)) >= float(config.accept_min_delta)
            ]
            chosen_candidate = None
            if improving_candidates:
                chosen_candidate = max(improving_candidates, key=lambda r: float(r["metric_delta"]))
                selected_operation_for_iter = str(chosen_candidate["candidate_op_name"])
                logging.info(
                    "[Iter %s] Selected candidate op '%s' with best ACTUAL delta_vs_best_%s=%.6f.",
                    iter_idx + 1,
                    selected_operation_for_iter,
                    config.accept_metric,
                    float(chosen_candidate["metric_delta"]),
                )

                current_merged_df = chosen_candidate["merged_df"]
                schema_state = chosen_candidate["schema_state"]
                current_choices_list = list(chosen_candidate["choices_list"])
                current_prob_calibrator = chosen_candidate["prob_calibrator"]
                token_count_tokenizer = chosen_candidate["tokenizer"]
                next_new_cluster_id = int(chosen_candidate["next_new_cluster_id"])
                split_cooldown = dict(chosen_candidate["split_cooldown"])
                revise_cooldown = dict(chosen_candidate["revise_cooldown"])
                add_cooldown_until = int(chosen_candidate["add_cooldown_until"])

                iteration_actions = dict(chosen_candidate["iteration_actions"])
                iteration_actions["selected_operation"] = selected_operation_for_iter
                metric_delta = float(chosen_candidate["metric_delta"])
                post_doc_scores_iter = chosen_candidate["post_doc_scores"]
                post_corpus_scores_iter = chosen_candidate["post_corpus_scores"]
                proposed_action_count = int(chosen_candidate["proposed_action_count"])

                iteration_actions["accepted"] = True
                iteration_actions["metric_delta_avg_logL_tok"] = metric_delta
                iteration_actions["metric_delta_accept_metric"] = metric_delta
                iteration_actions["accept_metric"] = config.accept_metric

                for variant in VALID_ACCEPT_METRICS:
                    accepted_avg_logL_per_token_by_variant[variant] = float(
                        post_corpus_scores_iter[f"avg_logL_per_token_{variant}"]
                    )
                no_op_iters = 0
            else:
                best_failed = None
                if candidate_results:
                    best_failed = max(candidate_results, key=lambda r: float(r["metric_delta"]))
                    selected_operation_for_iter = str(best_failed["candidate_op_name"])
                logging.info(
                    "[Iter %s] No candidate op improved %s metric by min_delta %.6f; applying no-op.",
                    iter_idx + 1,
                    config.accept_metric,
                    float(config.accept_min_delta),
                )
                current_merged_df = prev_merged_df
                schema_state = prev_schema_state
                current_choices_list = prev_choices_list
                current_prob_calibrator = prev_prob_calibrator
                token_count_tokenizer = prev_token_count_tokenizer
                split_cooldown = prev_split_cooldown
                revise_cooldown = prev_revise_cooldown
                add_cooldown_until = prev_add_cooldown_until
                next_new_cluster_id = prev_next_new_cluster_id

                metric_delta = float(best_failed["metric_delta"]) if best_failed is not None else np.nan
                post_doc_scores_iter = pre_doc_scores_iter
                post_corpus_scores_iter = pre_corpus_scores_iter
                proposed_action_count = 0

                iteration_actions = {"split": [], "merge": [], "remove": [], "revise": [], "add": []}
                iteration_actions["selected_operation"] = selected_operation_for_iter
                iteration_actions["accepted"] = False
                iteration_actions["metric_delta_avg_logL_tok"] = metric_delta
                iteration_actions["metric_delta_accept_metric"] = metric_delta
                iteration_actions["accept_metric"] = config.accept_metric
                if best_failed is not None:
                    iteration_actions["proposed_actions"] = dict(best_failed.get("iteration_actions", {}))
                no_op_iters += 1

            if should_compute_iter_metrics and post_corpus_scores_iter is not None:
                logging.info(
                    "[Iter %s] POST avg_logL_tok: oracle=%.6f, assigned=%.6f, soft=%.6f | "
                    "PPL_tok: oracle=%.3f, assigned=%.3f, soft=%.3f",
                    iter_idx + 1,
                    float(post_corpus_scores_iter["avg_logL_per_token_oracle"]),
                    float(post_corpus_scores_iter["avg_logL_per_token_assigned"]),
                    float(post_corpus_scores_iter["avg_logL_per_token_soft"]),
                    float(post_corpus_scores_iter["perplexity_token_oracle"]),
                    float(post_corpus_scores_iter["perplexity_token_assigned"]),
                    float(post_corpus_scores_iter["perplexity_token_soft"]),
                )
                if pre_corpus_scores_iter is not None:
                    logging.info(
                        "[Iter %s] DELTA (POST-PRE) %s avg_logL_tok=%.6f",
                        iter_idx + 1,
                        config.accept_metric,
                        float(
                            post_corpus_scores_iter[f"avg_logL_per_token_{config.accept_metric}"]
                            - pre_corpus_scores_iter[f"avg_logL_per_token_{config.accept_metric}"]
                        ),
                    )

            if bool(iteration_actions.get("accepted", False)) and proposed_action_count > 0 and selected_operation_for_iter:
                if selected_operation_for_iter == last_selected_operation:
                    same_operation_streak += 1
                else:
                    last_selected_operation = str(selected_operation_for_iter)
                    same_operation_streak = 1
                logging.info(
                    "[Iter %s] Applied selected op '%s'. Current same-op streak=%s.",
                    iter_idx + 1,
                    selected_operation_for_iter,
                    same_operation_streak,
                )

            unique_cluster_ids_after_iter = sorted(current_merged_df["cluster_id"].unique())
            logging.info(
                "Schema diff (iter %s): #labels=%s, adds=%s, splits=%s, merges=%s, removes=%s, revises=%s",
                iter_idx + 1,
                len(current_choices_list),
                len(iteration_actions.get("add", [])),
                len(iteration_actions.get("split", [])),
                len(iteration_actions.get("merge", [])),
                len(iteration_actions.get("remove", [])),
                len(iteration_actions.get("revise", [])),
            )

            if log_iteration_metrics and post_corpus_scores_iter is not None:
                delta_prepost_oracle = (
                    float(post_corpus_scores_iter["avg_logL_per_token_oracle"] - pre_corpus_scores_iter["avg_logL_per_token_oracle"])
                    if pre_corpus_scores_iter is not None else np.nan
                )
                delta_prepost_assigned = (
                    float(post_corpus_scores_iter["avg_logL_per_token_assigned"] - pre_corpus_scores_iter["avg_logL_per_token_assigned"])
                    if pre_corpus_scores_iter is not None else np.nan
                )
                delta_prepost_soft = (
                    float(post_corpus_scores_iter["avg_logL_per_token_soft"] - pre_corpus_scores_iter["avg_logL_per_token_soft"])
                    if pre_corpus_scores_iter is not None else np.nan
                )
                iteration_metrics_rows.append({
                    "iter": int(iter_idx + 1),
                    "clusters": int(len(unique_cluster_ids_after_iter)),
                    "splits": int(len(iteration_actions.get("split", []))),
                    "merges": int(len(iteration_actions.get("merge", []))),
                    "removes": int(len(iteration_actions.get("remove", []))),
                    "adds": int(len(iteration_actions.get("add", []))),
                    "revises": int(len(iteration_actions.get("revise", []))),
                    "selected_operation": iteration_actions.get("selected_operation"),
                    "accept_metric": config.accept_metric,
                    "accepted": bool(iteration_actions.get("accepted", True)),
                    "delta_vs_best_avg_logL_tok": metric_delta,
                    "delta_vs_best_accept_metric": metric_delta,
                    "mismatch_rate": float(diag_summary["mismatch_rate"]) if diag_summary and diag_summary.get("mismatch_rate") is not None else np.nan,
                    "pmax_median": float(diag_summary["pmax_median"]) if diag_summary and diag_summary.get("pmax_median") is not None else np.nan,
                    "entropy_median": float(diag_summary["entropy_median"]) if diag_summary and diag_summary.get("entropy_median") is not None else np.nan,
                    "L_baseline": float(post_doc_scores_iter["L_baseline"]),
                    "token_count_total": float(post_corpus_scores_iter["token_count_total"]),
                    "pre_avg_logL_per_token_oracle": float(pre_corpus_scores_iter["avg_logL_per_token_oracle"]) if pre_corpus_scores_iter is not None else np.nan,
                    "pre_avg_logL_per_token_assigned": float(pre_corpus_scores_iter["avg_logL_per_token_assigned"]) if pre_corpus_scores_iter is not None else np.nan,
                    "pre_avg_logL_per_token_soft": float(pre_corpus_scores_iter["avg_logL_per_token_soft"]) if pre_corpus_scores_iter is not None else np.nan,
                    "pre_PPL_tok_oracle": float(pre_corpus_scores_iter["perplexity_token_oracle"]) if pre_corpus_scores_iter is not None else np.nan,
                    "pre_PPL_tok_assigned": float(pre_corpus_scores_iter["perplexity_token_assigned"]) if pre_corpus_scores_iter is not None else np.nan,
                    "pre_PPL_tok_soft": float(pre_corpus_scores_iter["perplexity_token_soft"]) if pre_corpus_scores_iter is not None else np.nan,
                    "logL_total_oracle": float(post_corpus_scores_iter["logL_total_oracle"]),
                    "logL_total_assigned": float(post_corpus_scores_iter["logL_total_assigned"]),
                    "logL_total_soft": float(post_corpus_scores_iter["logL_total_soft"]),
                    "avg_logL_per_token_oracle": float(post_corpus_scores_iter["avg_logL_per_token_oracle"]),
                    "avg_logL_per_token_assigned": float(post_corpus_scores_iter["avg_logL_per_token_assigned"]),
                    "avg_logL_per_token_soft": float(post_corpus_scores_iter["avg_logL_per_token_soft"]),
                    "PPL_tok_oracle": float(post_corpus_scores_iter["perplexity_token_oracle"]),
                    "PPL_tok_assigned": float(post_corpus_scores_iter["perplexity_token_assigned"]),
                    "PPL_tok_soft": float(post_corpus_scores_iter["perplexity_token_soft"]),
                    "delta_prepost_avg_logL_per_token_oracle": delta_prepost_oracle,
                    "delta_prepost_avg_logL_per_token_assigned": delta_prepost_assigned,
                    "delta_prepost_avg_logL_per_token_soft": delta_prepost_soft,
                    "delta_prepost_PPL_tok_oracle": (
                        float(post_corpus_scores_iter["perplexity_token_oracle"] - pre_corpus_scores_iter["perplexity_token_oracle"])
                        if pre_corpus_scores_iter is not None else np.nan
                    ),
                    "delta_prepost_PPL_tok_assigned": (
                        float(post_corpus_scores_iter["perplexity_token_assigned"] - pre_corpus_scores_iter["perplexity_token_assigned"])
                        if pre_corpus_scores_iter is not None else np.nan
                    ),
                    "delta_prepost_PPL_tok_soft": (
                        float(post_corpus_scores_iter["perplexity_token_soft"] - pre_corpus_scores_iter["perplexity_token_soft"])
                        if pre_corpus_scores_iter is not None else np.nan
                    ),
                    "logL_total": float(post_corpus_scores_iter["logL_cond_total"]),
                    "avg_logL_per_token": float(post_corpus_scores_iter["avg_logL_per_token"]),
                    "AIC": float(post_corpus_scores_iter["AIC"]) if post_corpus_scores_iter["AIC"] is not None else np.nan,
                    "BIC": float(post_corpus_scores_iter["BIC"]) if post_corpus_scores_iter["BIC"] is not None else np.nan,
                    "PPL": float(post_corpus_scores_iter["perplexity"]),
                    "PPL_tok": float(post_corpus_scores_iter["perplexity_token"]),
                    "ELBO": float(post_corpus_scores_iter["ELBO"]),
                })

            schema_history.append(iteration_actions)
            logging.info(f"=== EM Iteration {iter_idx + 1} Complete. #Clusters: {len(unique_cluster_ids_after_iter)} ===")

            if verbose:
                logging.info(f"Iteration {iter_idx + 1} - Schema Label Priors (P(z)) after updates:")
                if not current_merged_df.empty and 'agglomerative_label' in current_merged_df.columns and current_merged_df['agglomerative_label'].notna().any():
                    label_proportions = current_merged_df['agglomerative_label'].value_counts(normalize=True)
                    for label, prob in label_proportions.items():
                        logging.info(f"  P('{label}') = {prob:.4f}")
                else:
                    logging.info("  No data or agglomerative_labels to compute P(z) from current_merged_df.")

            if no_op_iters >= max(1, int(config.noop_patience)):
                logging.info(
                    "Stopping early after %s consecutive no-op iterations (noop_patience=%s).",
                    no_op_iters,
                    config.noop_patience,
                )
                break
            continue
        iteration_actions.update(actions)
        iteration_actions["selected_operation"] = selected_operation_for_iter
        logging.info(f"Iteration {iter_idx + 1}: Actions decided from metrics (Split/Merge/Remove/Revise): {actions}. 'Add' actions to be determined next.")
        current_merged_df, next_new_cluster_id, split_pairs, merge_groups = apply_schema_updates(
            current_merged_df,
            iter_PZX,
            iter_log_px_given_z,
            idx_to_pzx_row_map,
            current_choices_list,
            actions,
            next_new_cluster_id,
        )
        if actions.get("remove"):
            current_merged_df = _reassign_removed_datapoints(
                post_update_df=current_merged_df,
                pre_update_df=prev_merged_df,
                removed_cluster_ids=list(actions.get("remove", [])),
                base_calibrator=current_prob_calibrator,
                iter_number=iter_idx + 1,
                log_details=True,
            )
        # Rename split children (and parents) with meaningful labels based on exemplar texts.
        # Split labels are LLM-only: if we cannot obtain distinct parent/child names, revert that split pair.
        split_child_ids = {child for _, child in split_pairs}
        if split_pairs and config.operation_enabled("split"):
            shared_vllm_tokenizer = getattr(current_prob_calibrator, "vllm_tokenizer", None)
            shared_vllm_model = getattr(current_prob_calibrator, "vllm_model", None)
            kept_split_pairs: list[tuple[int, int]] = []
            for parent_id, child_id in split_pairs:
                parent_label = schema_state.label_map.get(parent_id, f"cluster_{parent_id}")
                parent_texts = current_merged_df.loc[current_merged_df["cluster_id"] == parent_id, text_column_name].astype(str).tolist()
                child_texts = current_merged_df.loc[current_merged_df["cluster_id"] == child_id, text_column_name].astype(str).tolist()
                parent_samples = _sample_texts_for_labeling(parent_texts, max_texts=10, seed=int(parent_id))
                child_samples = _sample_texts_for_labeling(child_texts, max_texts=10, seed=int(child_id))
                existing_forbidden_names = _collect_existing_cluster_labels(
                    current_merged_df,
                    exclude_cluster_ids={int(parent_id), int(child_id)},
                )

                if config.structural_label_mode == "parent_relative":
                    new_parent_label, new_child_label = _build_parent_relative_split_labels(
                        parent_label=parent_label,
                        parent_id=int(parent_id),
                    )
                elif model_type == "vllm":
                    new_parent_label, new_child_label = _propose_split_labels_with_retries(
                        parent_samples=parent_samples,
                        child_samples=child_samples,
                        parent_label=parent_label,
                        model_identifier=model_identifier,
                        tokenizer=shared_vllm_tokenizer,
                        model=shared_vllm_model,
                        context_prefix=f"split:{parent_id}->{child_id}",
                        existing_forbidden_names=existing_forbidden_names,
                        pair_attempts=config.split_label_pair_attempts,
                        llm_label_max_attempts=config.llm_label_max_attempts,
                    )
                else:
                    new_parent_label, new_child_label = _build_parent_relative_split_labels(
                        parent_label=parent_label,
                        parent_id=int(parent_id),
                    )

                parent_key = _canonical_label_key(new_parent_label)
                child_key = _canonical_label_key(new_child_label)
                pair_near_dupe = False
                if parent_key and child_key:
                    pair_near_dupe = bool(
                        detect_near_duplicate_labels(
                            [new_parent_label, new_child_label],
                            threshold=DUP_LABEL_SIM_THRESHOLD,
                        )
                    )
                existing_label_keys = {
                    _canonical_label_key(lbl)
                    for cid, lbl in current_merged_df.groupby("cluster_id")["agglomerative_label"].first().items()
                    if int(cid) not in {int(parent_id), int(child_id)}
                }
                collides_existing_label = bool(
                    (parent_key and parent_key in existing_label_keys)
                    or (child_key and child_key in existing_label_keys)
                )

                if (
                    not new_parent_label
                    or not new_child_label
                    or not parent_key
                    or not child_key
                    or parent_key == child_key
                    or pair_near_dupe
                    or collides_existing_label
                ):
                    logging.warning(
                        "Reverting split (%s -> %s): LLM failed to produce distinct child labels "
                        "(parent='%s', child='%s', collides_existing=%s).",
                        parent_id,
                        child_id,
                        new_parent_label,
                        new_child_label,
                        collides_existing_label,
                    )
                    current_merged_df.loc[current_merged_df["cluster_id"] == child_id, "cluster_id"] = parent_id
                    current_merged_df.loc[current_merged_df["cluster_id"] == parent_id, "agglomerative_label"] = parent_label
                    continue

                current_merged_df.loc[current_merged_df["cluster_id"] == parent_id, "agglomerative_label"] = new_parent_label
                current_merged_df.loc[current_merged_df["cluster_id"] == child_id, "agglomerative_label"] = new_child_label
                kept_split_pairs.append((parent_id, child_id))
            split_pairs = kept_split_pairs
            split_child_ids = {child for _, child in split_pairs}
            iteration_actions["split"] = [p for p, _ in split_pairs]

        if merge_groups and config.operation_enabled("merge"):
            shared_vllm_tokenizer = getattr(current_prob_calibrator, "vllm_tokenizer", None)
            shared_vllm_model = getattr(current_prob_calibrator, "vllm_model", None)
            for root_id, member_ids in merge_groups:
                if root_id not in set(current_merged_df["cluster_id"].unique()):
                    continue
                source_labels = [
                    schema_state.label_map.get(mid, f"cluster_{mid}")
                    for mid in member_ids
                ]
                merged_texts = current_merged_df.loc[
                    current_merged_df["cluster_id"] == root_id, text_column_name
                ].astype(str).tolist()
                merged_samples = _sample_texts_for_labeling(merged_texts, max_texts=12, seed=int(root_id))
                new_merge_label = None
                existing_forbidden_names = _collect_existing_cluster_labels(
                    current_merged_df,
                    exclude_cluster_ids={int(root_id)},
                )
                merge_banned_names = list(existing_forbidden_names) + list(source_labels)
                if config.structural_label_mode == "parent_relative":
                    new_merge_label = _build_parent_relative_merge_label(
                        source_labels=[str(x) for x in source_labels],
                        root_id=int(root_id),
                    )
                elif model_type == "vllm":
                    llm_label = _label_cluster_vllm(
                        merged_samples,
                        model_identifier,
                        tokenizer=shared_vllm_tokenizer,
                        model=shared_vllm_model,
                        parent_label=" + ".join(base_label(lbl) for lbl in source_labels[:2]),
                        banned_names=merge_banned_names,
                        context=f"merge_cluster:{root_id}",
                        max_attempts=config.llm_label_max_attempts,
                    )
                    if llm_label:
                        name, _, keywords = llm_label
                        new_merge_label = build_label_string(name, keywords)
                if not new_merge_label:
                    new_merge_label = _fallback_distinct_label_from_texts(
                        merged_samples,
                        banned_names=merge_banned_names,
                    )
                    if not new_merge_label:
                        name, keywords = propose_label_and_keywords_from_texts(merged_samples, parent_label=None)
                        new_merge_label = build_label_string(name, keywords)
                current_merged_df.loc[current_merged_df["cluster_id"] == root_id, "agglomerative_label"] = new_merge_label
                logging.info(
                    "Merge relabel: cluster %s (members=%s) -> '%s'",
                    root_id,
                    member_ids,
                    new_merge_label,
                )

        revised_clusters_applied: list[int] = []
        if actions.get("revise") and config.operation_enabled("revise"):
            shared_vllm_tokenizer = getattr(current_prob_calibrator, "vllm_tokenizer", None)
            shared_vllm_model = getattr(current_prob_calibrator, "vllm_model", None)
            current_cluster_ids = set(current_merged_df["cluster_id"].unique())
            for cid in actions["revise"]:
                if cid not in current_cluster_ids:
                    logging.warning("Revise requested for cluster %s, but cluster no longer exists after structural updates.", cid)
                    continue
                current_label = str(current_merged_df.loc[current_merged_df["cluster_id"] == cid, "agglomerative_label"].iloc[0])
                revise_texts = current_merged_df.loc[current_merged_df["cluster_id"] == cid, text_column_name].astype(str).tolist()
                revise_samples = _sample_texts_for_labeling(revise_texts, max_texts=12, seed=int(cid))
                new_label = None
                if model_type == "vllm":
                    llm_label = _label_cluster_vllm(
                        revise_samples,
                        model_identifier,
                        tokenizer=shared_vllm_tokenizer,
                        model=shared_vllm_model,
                        parent_label=base_label(current_label),
                        banned_names=[current_label],
                        context=f"revise_cluster:{cid}",
                        max_attempts=config.llm_label_max_attempts,
                    )
                    if llm_label:
                        name, _, keywords = llm_label
                        new_label = build_label_string(name, keywords)
                if not new_label:
                    new_label = _fallback_distinct_label_from_texts(revise_samples, banned_names=[current_label])
                    if not new_label:
                        name, keywords = propose_label_and_keywords_from_texts(revise_samples, parent_label=base_label(current_label))
                        new_label = build_label_string(name, keywords)
                current_merged_df.loc[current_merged_df["cluster_id"] == cid, "agglomerative_label"] = new_label
                revised_clusters_applied.append(int(cid))
                logging.info("Revise relabel: cluster %s '%s' -> '%s'", cid, current_label, new_label)
            iteration_actions["revise"] = revised_clusters_applied
            for cid in revised_clusters_applied:
                revise_cooldown[cid] = (iter_idx + 1) + config.revise_cooldown_iters

        # Rename any exact duplicates before final schema rebuild.
        dup_renamed = _rename_duplicate_labels(current_merged_df, text_column_name)

        # Near-duplicate guardrail: revert any remaining problematic split pairs (pair-local only)
        duplicates_prevented = False
        if split_pairs:
            still_kept_pairs: list[tuple[int, int]] = []
            for parent_id, child_id in split_pairs:
                if child_id not in set(current_merged_df["cluster_id"].unique()):
                    continue
                parent_label_now = str(current_merged_df.loc[current_merged_df["cluster_id"] == parent_id, "agglomerative_label"].iloc[0])
                child_label_now = str(current_merged_df.loc[current_merged_df["cluster_id"] == child_id, "agglomerative_label"].iloc[0])
                pair_dup = bool(
                    detect_near_duplicate_labels(
                        [parent_label_now, child_label_now],
                        threshold=DUP_LABEL_SIM_THRESHOLD,
                    )
                )
                if pair_dup or _canonical_label_key(parent_label_now) == _canonical_label_key(child_label_now):
                    duplicates_prevented = True
                    logging.warning(
                        "Near-duplicate split labels for (%s -> %s): parent='%s' child='%s'. Reverting this split pair.",
                        parent_id,
                        child_id,
                        parent_label_now,
                        child_label_now,
                    )
                    current_merged_df.loc[current_merged_df["cluster_id"] == child_id, "cluster_id"] = parent_id
                else:
                    still_kept_pairs.append((parent_id, child_id))
            split_pairs = still_kept_pairs
            split_child_ids = {child for _, child in split_pairs}
            iteration_actions["split"] = [p for p, _ in split_pairs]
            current_merged_df["agglomerative_label"] = current_merged_df["cluster_id"].map(
                current_merged_df.groupby("cluster_id")["agglomerative_label"].first().to_dict()
            )
        if split_pairs:
            for parent_id, _ in split_pairs:
                split_cooldown[parent_id] = (iter_idx + 1) + config.split_cooldown_iters
        
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
        if not config.operation_enabled("add"):
            logging.info("  Skipping add-step because add operation is disabled by tune configuration.")
        elif iter_idx + 1 <= add_cooldown_until:
            logging.info("  Skipping add-step due to cooldown (last add in iter %s).", add_cooldown_until - 1)
        else:
            if schema_unstable:
                logging.info(
                    "  Proceeding with add-step even after structural changes (split/merge/remove/duplicate guardrail)."
                )
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
                    f"  pmax stats vs add-threshold {config.add_low_confidence_max}: "
                    f"min/median/max={np.min(pmax_vals):.4f}/{np.median(pmax_vals):.4f}/{np.max(pmax_vals):.4f}"
                )

            poor_pos, dynamic_pmax_threshold = _select_poor_candidates(
                pmax_vals,
                entropy_vals,
                assigned_ranks,
                config.add_min_poorly_explained,
                config.add_low_confidence_max,
                config.add_low_confidence_quantile,
                config.add_entropy_min,
            )
            poor_indices = [all_indices[i] for i in poor_pos]
            logging.info(
                "  Add-step candidates: bottom-K=%s, filtered_poor=%s (pmax<=dynamic %.3f from max(fixed %.3f, quantile q=%.3f), entropy>=%.3f, rank>1)",
                max(config.add_min_poorly_explained, int(0.05 * len(current_merged_df))),
                len(poor_indices),
                dynamic_pmax_threshold,
                config.add_low_confidence_max,
                config.add_low_confidence_quantile,
                config.add_entropy_min,
            )

            if len(current_choices_list) >= config.add_max_total_clusters:
                logging.info("  Max total clusters reached; skipping add-step.")
                poor_indices = []

            if len(poor_indices) >= config.add_min_poorly_explained:
                logging.info(f"  Found {len(poor_indices)} texts poorly explained. Attempting to form new clusters (LLM-only).")
                valid_poor_indices = list(poor_indices)
                # Build posterior vectors for poor docs (already partially computed above via pmax_vals)
                poor_pzx = np.vstack([
                    current_prob_calibrator.calibrate_p_z_given_X(
                        current_merged_df.loc[idx, text_column_name]
                    )
                    for idx in valid_poor_indices
                ]) if valid_poor_indices else np.empty((0, len(current_choices_list)))

                if len(valid_poor_indices) < config.add_min_group_size:
                    logging.warning("  Not enough poorly explained texts for add step.")
                else:
                    prev_cluster_ids = current_merged_df.loc[valid_poor_indices, "cluster_id"].copy()
                    add_prev_cluster_ids = prev_cluster_ids
                    num_new_clusters_to_add = max(1, len(valid_poor_indices) // config.add_items_per_new_cluster)
                    num_new_clusters_to_add = min(num_new_clusters_to_add, config.add_max_new_clusters_per_iter)
                    logging.info(f"  Attempting to create {num_new_clusters_to_add} new clusters from posterior vectors.")
                    from sklearn.cluster import KMeans
                    if len(poor_pzx) < num_new_clusters_to_add:
                        logging.warning(
                            f"  Number of poorly explained texts ({len(poor_pzx)}) < target new clusters ({num_new_clusters_to_add}). Adjusting."
                        )
                        num_new_clusters_to_add = len(poor_pzx)
                    try:
                        created = 0
                        for k_try in [min(2, num_new_clusters_to_add), min(3, num_new_clusters_to_add)]:
                            if k_try < 2:
                                continue
                            kmeans_poor = KMeans(n_clusters=k_try, n_init='auto', random_state=42)
                            new_labels_for_poor_texts = kmeans_poor.fit_predict(poor_pzx)
                            for k_new_cluster in range(k_try):
                                new_cluster_original_indices = [valid_poor_indices[i] for i, lab in enumerate(new_labels_for_poor_texts) if lab == k_new_cluster]
                                if len(new_cluster_original_indices) < config.add_min_group_size:
                                    continue
                                # KL divergence cohesion check on posteriors
                                group_pzx = poor_pzx[[valid_poor_indices.index(i) for i in new_cluster_original_indices]]
                                cohesion = kl_divergence_cohesion(group_pzx)
                                if cohesion < config.add_cohesion_min:
                                    continue
                                assigned_new_id = next_new_cluster_id
                                next_new_cluster_id += 1
                                current_merged_df.loc[new_cluster_original_indices, 'cluster_id'] = assigned_new_id
                                texts = current_merged_df.loc[new_cluster_original_indices, text_column_name].astype(str).tolist()
                                new_label = None
                                if model_type == "vllm":
                                    llm_label = _label_cluster_vllm(
                                        texts,
                                        model_identifier,
                                        tokenizer=getattr(current_prob_calibrator, "vllm_tokenizer", None),
                                        model=getattr(current_prob_calibrator, "vllm_model", None),
                                        context=f"add_cluster:{assigned_new_id}",
                                        max_attempts=config.llm_label_max_attempts,
                                    )
                                    if llm_label:
                                        name, _, keywords = llm_label
                                        new_label = build_label_string(name, keywords)
                                if not new_label:
                                    logging.info(
                                        "Add-step relabel fallback for new cluster %s: using heuristic label generation.",
                                        assigned_new_id,
                                    )
                                    name, keywords = propose_label_and_keywords_from_texts(texts, parent_label=None)
                                    new_label = build_label_string(name, keywords)
                                current_merged_df.loc[new_cluster_original_indices, 'agglomerative_label'] = new_label
                                added_clusters_info.append({'id': assigned_new_id, 'size': len(new_cluster_original_indices)})
                                logging.info(f"    Added new cluster {assigned_new_id} with {len(new_cluster_original_indices)} texts.")
                                created += 1
                                if created >= config.add_max_new_clusters_per_iter:
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
                logging.info(
                    "  Found %s poorly explained texts (threshold: %s). Not adding new clusters based on this criterion.",
                    len(poor_indices),
                    config.add_min_poorly_explained,
                )
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
                add_cooldown_until = (iter_idx + 1) + config.add_cooldown_iters

        iteration_actions['add'] = added_clusters_info
        if (
            config.best_operation_per_iteration
            and config.tune_operation == "all"
            and iteration_actions.get("selected_operation") is None
            and len(added_clusters_info) > 0
        ):
            iteration_actions["selected_operation"] = "add"
        logging.info(f"Iteration {iter_idx + 1}: 'Add' actions decided: {added_clusters_info}. Total actions for iteration: {iteration_actions}")
        if current_merged_df.empty:
            logging.warning(f"Iteration {iter_idx + 1}: Dataframe became empty after updates. Stopping EM refinement.")
            break
        unique_cluster_ids_after_iter = sorted(current_merged_df['cluster_id'].unique())
        if not unique_cluster_ids_after_iter:
            logging.warning(f"Iteration {iter_idx + 1}: No clusters remaining after updates. Stopping EM refinement.")
            break

        # Ensure no exact-duplicate labels remain.
        dup_renamed += _rename_duplicate_labels(current_merged_df, text_column_name)
        forced_unique = _enforce_unique_choice_labels(current_merged_df)
        if forced_unique > 0:
            logging.warning(
                "Iteration %s: enforced unique schema labels for %s clusters to avoid duplicate choices.",
                iter_idx + 1,
                forced_unique,
            )

        schema_state = build_schema_state_from_df(current_merged_df)
        current_merged_df['agglomerative_label'] = current_merged_df['cluster_id'].map(schema_state.label_map)
        current_choices_list = list(schema_state.choices_list)
        if len(set(current_choices_list)) != len(current_choices_list):
            forced_unique += _enforce_unique_choice_labels(current_merged_df)
            schema_state = build_schema_state_from_df(current_merged_df)
            current_merged_df['agglomerative_label'] = current_merged_df['cluster_id'].map(schema_state.label_map)
            current_choices_list = list(schema_state.choices_list)
            if len(set(current_choices_list)) != len(current_choices_list):
                raise ValueError(
                    "Internal invariant violation: schema choices are not unique after uniqueness enforcement."
                )

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

        token_count_tokenizer = getattr(current_prob_calibrator, "tokenizer", None) or getattr(current_prob_calibrator, "vllm_tokenizer", None) or token_count_tokenizer

        # Keep calibrator prior aligned with current schema
        if schema_state.p_z_prior is not None:
            current_prob_calibrator.p_z = schema_state.p_z_prior

        logging.info(
            "Schema diff (iter %s): #labels=%s, renamed_duplicates=%s, adds=%s, splits=%s, merges=%s, removes=%s, revises=%s",
            iter_idx + 1,
            len(current_choices_list),
            dup_renamed,
            len(iteration_actions.get("add", [])),
            len(iteration_actions.get("split", [])),
            len(iteration_actions.get("merge", [])),
            len(iteration_actions.get("remove", [])),
            len(iteration_actions.get("revise", [])),
        )

        post_doc_scores_iter = None
        post_corpus_scores_iter = None
        if should_compute_iter_metrics:
            post_doc_scores_iter, _, post_corpus_scores_iter = _compute_variant_metrics(
                current_merged_df,
                schema_state,
            )
            logging.info(
                "[Iter %s] POST avg_logL_tok: oracle=%.6f, assigned=%.6f, soft=%.6f | "
                "PPL_tok: oracle=%.3f, assigned=%.3f, soft=%.3f",
                iter_idx + 1,
                float(post_corpus_scores_iter["avg_logL_per_token_oracle"]),
                float(post_corpus_scores_iter["avg_logL_per_token_assigned"]),
                float(post_corpus_scores_iter["avg_logL_per_token_soft"]),
                float(post_corpus_scores_iter["perplexity_token_oracle"]),
                float(post_corpus_scores_iter["perplexity_token_assigned"]),
                float(post_corpus_scores_iter["perplexity_token_soft"]),
            )
            if pre_corpus_scores_iter is not None:
                logging.info(
                    "[Iter %s] DELTA (POST-PRE) %s avg_logL_tok=%.6f",
                    iter_idx + 1,
                    config.accept_metric,
                    float(
                        post_corpus_scores_iter[f"avg_logL_per_token_{config.accept_metric}"]
                        - pre_corpus_scores_iter[f"avg_logL_per_token_{config.accept_metric}"]
                    ),
                )

        proposed_action_count = int(
            len(iteration_actions.get("split", []))
            + len(iteration_actions.get("merge", []))
            + len(iteration_actions.get("remove", []))
            + len(iteration_actions.get("add", []))
            + len(iteration_actions.get("revise", []))
        )
        accepted_update = True
        metric_delta = np.nan
        accept_metric_key = f"avg_logL_per_token_{config.accept_metric}"
        if (
            config.rollback_on_metric_degrade
            and post_corpus_scores_iter is not None
            and accepted_avg_logL_per_token_by_variant.get(config.accept_metric) is not None
            and proposed_action_count > 0
        ):
            metric_delta = float(
                post_corpus_scores_iter[accept_metric_key]
                - accepted_avg_logL_per_token_by_variant[config.accept_metric]
            )
            if metric_delta < float(config.accept_min_delta):
                accepted_update = False
                logging.warning(
                    "Iteration %s rejected: avg_logL_per_token_%s delta %.6f < min_delta %.6f. Rolling back schema updates.",
                    iter_idx + 1,
                    config.accept_metric,
                    metric_delta,
                    float(config.accept_min_delta),
                )
                # Record clusters whose splits were rejected so we can apply
                # cooldown even after rollback — prevents the same cluster from
                # being proposed again immediately on the next iteration.
                rejected_split_cluster_ids = list(iteration_actions.get("split", []))
                current_merged_df = prev_merged_df
                schema_state = prev_schema_state
                current_choices_list = prev_choices_list
                current_prob_calibrator = prev_prob_calibrator
                token_count_tokenizer = prev_token_count_tokenizer
                split_cooldown = prev_split_cooldown
                # Apply cooldown to rejected splits so they are not re-proposed
                # immediately. Use the same cooldown length as accepted splits.
                for _rejected_cid in rejected_split_cluster_ids:
                    split_cooldown[_rejected_cid] = (iter_idx + 1) + config.split_cooldown_iters
                    logging.info(
                        "Rejected split: applying cooldown to cluster %s until iter %s.",
                        _rejected_cid,
                        split_cooldown[_rejected_cid],
                    )
                revise_cooldown = prev_revise_cooldown
                add_cooldown_until = prev_add_cooldown_until
                next_new_cluster_id = prev_next_new_cluster_id
                unique_cluster_ids_after_iter = sorted(current_merged_df["cluster_id"].unique())
                iteration_actions["accepted"] = False
                iteration_actions["metric_delta_avg_logL_tok"] = metric_delta
                iteration_actions["metric_delta_accept_metric"] = metric_delta
                iteration_actions["accept_metric"] = config.accept_metric
                iteration_actions["proposed_actions"] = {
                    "split": iteration_actions.get("split", []),
                    "merge": iteration_actions.get("merge", []),
                    "remove": iteration_actions.get("remove", []),
                    "add": iteration_actions.get("add", []),
                    "revise": iteration_actions.get("revise", []),
                }
                iteration_actions["split"] = []
                iteration_actions["merge"] = []
                iteration_actions["remove"] = []
                iteration_actions["add"] = []
                iteration_actions["revise"] = []
                proposed_action_count = 0
                if should_compute_iter_metrics:
                    post_doc_scores_iter, _, post_corpus_scores_iter = _compute_variant_metrics(
                        current_merged_df,
                        schema_state,
                    )
                    if pre_corpus_scores_iter is not None:
                        logging.info(
                            "[Iter %s] POST (after rollback) avg_logL_tok: oracle=%.6f, assigned=%.6f, soft=%.6f",
                            iter_idx + 1,
                            float(post_corpus_scores_iter["avg_logL_per_token_oracle"]),
                            float(post_corpus_scores_iter["avg_logL_per_token_assigned"]),
                            float(post_corpus_scores_iter["avg_logL_per_token_soft"]),
                        )
                no_op_iters += 1
            else:
                for variant in VALID_ACCEPT_METRICS:
                    accepted_avg_logL_per_token_by_variant[variant] = float(
                        post_corpus_scores_iter[f"avg_logL_per_token_{variant}"]
                    )
                iteration_actions["accepted"] = True
                iteration_actions["metric_delta_avg_logL_tok"] = metric_delta
                iteration_actions["metric_delta_accept_metric"] = metric_delta
                iteration_actions["accept_metric"] = config.accept_metric
                no_op_iters = 0
        else:
            iteration_actions["accepted"] = True
            if proposed_action_count > 0 and post_corpus_scores_iter is not None:
                for variant in VALID_ACCEPT_METRICS:
                    accepted_avg_logL_per_token_by_variant[variant] = float(
                        post_corpus_scores_iter[f"avg_logL_per_token_{variant}"]
                    )
            if proposed_action_count == 0:
                no_op_iters += 1
            else:
                no_op_iters = 0

        if config.best_operation_per_iteration and config.tune_operation == "all":
            applied_op = iteration_actions.get("selected_operation")
            if bool(iteration_actions.get("accepted", True)) and proposed_action_count > 0 and applied_op:
                if applied_op == last_selected_operation:
                    same_operation_streak += 1
                else:
                    last_selected_operation = str(applied_op)
                    same_operation_streak = 1
                logging.info(
                    "[Iter %s] Applied selected op '%s'. Current same-op streak=%s.",
                    iter_idx + 1,
                    applied_op,
                    same_operation_streak,
                )

        if log_iteration_metrics and post_corpus_scores_iter is not None:
            delta_prepost_oracle = (
                float(post_corpus_scores_iter["avg_logL_per_token_oracle"] - pre_corpus_scores_iter["avg_logL_per_token_oracle"])
                if pre_corpus_scores_iter is not None else np.nan
            )
            delta_prepost_assigned = (
                float(post_corpus_scores_iter["avg_logL_per_token_assigned"] - pre_corpus_scores_iter["avg_logL_per_token_assigned"])
                if pre_corpus_scores_iter is not None else np.nan
            )
            delta_prepost_soft = (
                float(post_corpus_scores_iter["avg_logL_per_token_soft"] - pre_corpus_scores_iter["avg_logL_per_token_soft"])
                if pre_corpus_scores_iter is not None else np.nan
            )
            iteration_metrics_rows.append({
                "iter": int(iter_idx + 1),
                "clusters": int(len(unique_cluster_ids_after_iter)),
                "splits": int(len(iteration_actions.get("split", []))),
                "merges": int(len(iteration_actions.get("merge", []))),
                "removes": int(len(iteration_actions.get("remove", []))),
                "adds": int(len(iteration_actions.get("add", []))),
                "revises": int(len(iteration_actions.get("revise", []))),
                "selected_operation": iteration_actions.get("selected_operation"),
                "accept_metric": config.accept_metric,
                "accepted": bool(iteration_actions.get("accepted", True)),
                "delta_vs_best_avg_logL_tok": metric_delta,
                "delta_vs_best_accept_metric": metric_delta,
                "mismatch_rate": float(diag_summary["mismatch_rate"]) if diag_summary and diag_summary.get("mismatch_rate") is not None else np.nan,
                "pmax_median": float(diag_summary["pmax_median"]) if diag_summary and diag_summary.get("pmax_median") is not None else np.nan,
                "entropy_median": float(diag_summary["entropy_median"]) if diag_summary and diag_summary.get("entropy_median") is not None else np.nan,
                "L_baseline": float(post_doc_scores_iter["L_baseline"]),
                "token_count_total": float(post_corpus_scores_iter["token_count_total"]),
                "pre_avg_logL_per_token_oracle": float(pre_corpus_scores_iter["avg_logL_per_token_oracle"]) if pre_corpus_scores_iter is not None else np.nan,
                "pre_avg_logL_per_token_assigned": float(pre_corpus_scores_iter["avg_logL_per_token_assigned"]) if pre_corpus_scores_iter is not None else np.nan,
                "pre_avg_logL_per_token_soft": float(pre_corpus_scores_iter["avg_logL_per_token_soft"]) if pre_corpus_scores_iter is not None else np.nan,
                "pre_PPL_tok_oracle": float(pre_corpus_scores_iter["perplexity_token_oracle"]) if pre_corpus_scores_iter is not None else np.nan,
                "pre_PPL_tok_assigned": float(pre_corpus_scores_iter["perplexity_token_assigned"]) if pre_corpus_scores_iter is not None else np.nan,
                "pre_PPL_tok_soft": float(pre_corpus_scores_iter["perplexity_token_soft"]) if pre_corpus_scores_iter is not None else np.nan,
                "logL_total_oracle": float(post_corpus_scores_iter["logL_total_oracle"]),
                "logL_total_assigned": float(post_corpus_scores_iter["logL_total_assigned"]),
                "logL_total_soft": float(post_corpus_scores_iter["logL_total_soft"]),
                "avg_logL_per_token_oracle": float(post_corpus_scores_iter["avg_logL_per_token_oracle"]),
                "avg_logL_per_token_assigned": float(post_corpus_scores_iter["avg_logL_per_token_assigned"]),
                "avg_logL_per_token_soft": float(post_corpus_scores_iter["avg_logL_per_token_soft"]),
                "PPL_tok_oracle": float(post_corpus_scores_iter["perplexity_token_oracle"]),
                "PPL_tok_assigned": float(post_corpus_scores_iter["perplexity_token_assigned"]),
                "PPL_tok_soft": float(post_corpus_scores_iter["perplexity_token_soft"]),
                "delta_prepost_avg_logL_per_token_oracle": delta_prepost_oracle,
                "delta_prepost_avg_logL_per_token_assigned": delta_prepost_assigned,
                "delta_prepost_avg_logL_per_token_soft": delta_prepost_soft,
                "delta_prepost_PPL_tok_oracle": (
                    float(post_corpus_scores_iter["perplexity_token_oracle"] - pre_corpus_scores_iter["perplexity_token_oracle"])
                    if pre_corpus_scores_iter is not None else np.nan
                ),
                "delta_prepost_PPL_tok_assigned": (
                    float(post_corpus_scores_iter["perplexity_token_assigned"] - pre_corpus_scores_iter["perplexity_token_assigned"])
                    if pre_corpus_scores_iter is not None else np.nan
                ),
                "delta_prepost_PPL_tok_soft": (
                    float(post_corpus_scores_iter["perplexity_token_soft"] - pre_corpus_scores_iter["perplexity_token_soft"])
                    if pre_corpus_scores_iter is not None else np.nan
                ),
                # Backward-compatible aliases (oracle).
                "logL_total": float(post_corpus_scores_iter["logL_cond_total"]),
                "avg_logL_per_token": float(post_corpus_scores_iter["avg_logL_per_token"]),
                "AIC": float(post_corpus_scores_iter["AIC"]) if post_corpus_scores_iter["AIC"] is not None else np.nan,
                "BIC": float(post_corpus_scores_iter["BIC"]) if post_corpus_scores_iter["BIC"] is not None else np.nan,
                "PPL": float(post_corpus_scores_iter["perplexity"]),
                "PPL_tok": float(post_corpus_scores_iter["perplexity_token"]),
                "ELBO": float(post_corpus_scores_iter["ELBO"]),
            })
        schema_history.append(iteration_actions)
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

        if no_op_iters >= max(1, int(config.noop_patience)):
            logging.info(
                "Stopping early after %s consecutive no-op iterations (noop_patience=%s).",
                no_op_iters,
                config.noop_patience,
            )
            break

    final_iter_count = iter_idx + 1 if max_iters > 0 and 'iter_idx' in locals() and iter_idx is not None else 0
    if iteration_metrics_rows:
        metrics_df = pd.DataFrame(iteration_metrics_rows).sort_values("iter").reset_index(drop=True)
        for metric_col in [
            "logL_total",
            "logL_total_oracle",
            "logL_total_assigned",
            "logL_total_soft",
            "avg_logL_per_token",
            "avg_logL_per_token_oracle",
            "avg_logL_per_token_assigned",
            "avg_logL_per_token_soft",
            "AIC",
            "BIC",
            "PPL",
            "PPL_tok",
            "PPL_tok_oracle",
            "PPL_tok_assigned",
            "PPL_tok_soft",
            "ELBO",
            "mismatch_rate",
            "pmax_median",
            "entropy_median",
        ]:
            if metric_col in metrics_df.columns:
                metrics_df[f"delta_{metric_col}"] = metrics_df[metric_col].diff()

        display_cols = [
            "iter", "clusters", "splits", "merges", "removes", "adds", "revises", "selected_operation", "accepted", "accept_metric", "delta_vs_best_accept_metric",
            "mismatch_rate", "pmax_median", "entropy_median",
            "logL_total_oracle", "logL_total_assigned", "logL_total_soft",
            "avg_logL_per_token_oracle", "avg_logL_per_token_assigned", "avg_logL_per_token_soft",
            "PPL_tok_oracle", "PPL_tok_assigned", "PPL_tok_soft",
            "AIC", "BIC", "ELBO",
            "delta_logL_total", "delta_avg_logL_per_token", "delta_AIC", "delta_BIC", "delta_PPL_tok", "delta_ELBO",
        ]
        display_cols = [c for c in display_cols if c in metrics_df.columns]
        logging.info(
            "Iteration metrics summary (better: logL_total/avg_logL_tok/ELBO higher; AIC/BIC/PPL_tok/mismatch/entropy lower; pmax higher):\n%s",
            metrics_df[display_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.6g}",
            ),
        )

        best_logl_iter = int(metrics_df.loc[metrics_df["logL_total_oracle"].idxmax(), "iter"])
        best_tokll_oracle_iter = int(metrics_df.loc[metrics_df["avg_logL_per_token_oracle"].idxmax(), "iter"])
        best_tokll_assigned_iter = int(metrics_df.loc[metrics_df["avg_logL_per_token_assigned"].idxmax(), "iter"])
        best_tokll_soft_iter = int(metrics_df.loc[metrics_df["avg_logL_per_token_soft"].idxmax(), "iter"])
        best_aic_iter = int(metrics_df.loc[metrics_df["AIC"].idxmin(), "iter"])
        best_bic_iter = int(metrics_df.loc[metrics_df["BIC"].idxmin(), "iter"])
        best_ppl_oracle_iter = int(metrics_df.loc[metrics_df["PPL_tok_oracle"].idxmin(), "iter"])
        best_ppl_assigned_iter = int(metrics_df.loc[metrics_df["PPL_tok_assigned"].idxmin(), "iter"])
        best_ppl_soft_iter = int(metrics_df.loc[metrics_df["PPL_tok_soft"].idxmin(), "iter"])
        best_elbo_iter = int(metrics_df.loc[metrics_df["ELBO"].idxmax(), "iter"])
        accept_metric_col = f"avg_logL_per_token_{config.accept_metric}"
        best_accept_iter = int(metrics_df.loc[metrics_df[accept_metric_col].idxmax(), "iter"])
        logging.info(
            "Best-iteration summary: logL_total_oracle=iter%s, avg_logL_tok_oracle=iter%s, avg_logL_tok_assigned=iter%s, avg_logL_tok_soft=iter%s, "
            "PPL_tok_oracle=iter%s, PPL_tok_assigned=iter%s, PPL_tok_soft=iter%s, "
            "AIC=iter%s, BIC=iter%s, ELBO=iter%s, accept_metric(%s)=iter%s",
            best_logl_iter,
            best_tokll_oracle_iter,
            best_tokll_assigned_iter,
            best_tokll_soft_iter,
            best_ppl_oracle_iter,
            best_ppl_assigned_iter,
            best_ppl_soft_iter,
            best_aic_iter,
            best_bic_iter,
            best_elbo_iter,
            config.accept_metric,
            best_accept_iter,
        )

        if diagnostics_dir is not None:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            summary_csv = diagnostics_dir / "em_iteration_metrics_summary.csv"
            metrics_df.to_csv(summary_csv, index=False)
            logging.info("Saved iteration metrics summary CSV to: %s", summary_csv)
    elif log_iteration_metrics:
        logging.info("Iteration metrics summary unavailable: no completed iterations with metric logging.")

    logging.info(f"EM schema refinement finished after {final_iter_count} iterations.")
    return current_merged_df, schema_history, current_choices_list, current_prob_calibrator


def main(args):
    setup_logging()

    args.tune_operation = str(args.tune_operation).lower()
    if args.tune_operation not in VALID_TUNE_OPERATIONS:
        raise ValueError(f"Invalid tune operation '{args.tune_operation}'. Expected one of {VALID_TUNE_OPERATIONS}.")

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
        sample_seed = args.subsample_seed if args.subsample_seed is not None and args.subsample_seed >= 0 else None
        logging.info(
            "Subsampling to up to %s datapoints per cluster (replace=False, seed=%s).",
            args.num_datapoints_per_cluster,
            sample_seed,
        )
        sampled_groups = []
        for cluster_id, group in merged_df.groupby("cluster_id", sort=False):
            n_take = min(int(args.num_datapoints_per_cluster), len(group))
            random_state = None if sample_seed is None else int(sample_seed + int(cluster_id))
            sampled_groups.append(group.sample(n=n_take, replace=False, random_state=random_state))
        merged_df = pd.concat(sampled_groups, axis=0).sort_index()
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
            tune_operation=args.tune_operation,
            split_enabled=args.split_enabled,
            split_max_per_iter=args.split_max_per_iter,
            split_min_cluster_size=args.split_min_cluster_size,
            split_ll_margin=args.split_ll_margin,
            split_confidence_max=args.split_confidence_max,
            split_gap_median_min=args.split_gap_median_min,
            split_min_conditions=args.split_min_conditions,
            split_cooldown_iters=args.split_cooldown_iters,
            merge_enabled=args.merge_enabled,
            merge_max_per_iter=args.merge_max_per_iter,
            merge_similarity_min=args.merge_similarity_min,
            merge_l_diff_ratio_max=args.merge_l_diff_ratio_max,
            merge_c_diff_ratio_max=args.merge_c_diff_ratio_max,
            merge_min_conditions=args.merge_min_conditions,
            remove_enabled=args.remove_enabled,
            remove_max_per_iter=args.remove_max_per_iter,
            remove_min_cluster_size=args.remove_min_cluster_size,
            remove_ll_factor=args.remove_ll_factor,
            revise_enabled=args.revise_enabled,
            revise_max_per_iter=args.revise_max_per_iter,
            revise_ll_margin=args.revise_ll_margin,
            revise_confidence_max=args.revise_confidence_max,
            revise_min_conditions=args.revise_min_conditions,
            revise_cooldown_iters=args.revise_cooldown_iters,
            add_enabled=args.add_enabled,
            add_low_confidence_max=args.add_low_confidence_max,
            add_low_confidence_quantile=args.add_low_confidence_quantile,
            add_min_poorly_explained=args.add_min_poorly_explained,
            add_items_per_new_cluster=args.add_items_per_new_cluster,
            add_max_new_clusters_per_iter=args.add_max_new_clusters_per_iter,
            add_max_total_clusters=args.add_max_total_clusters,
            add_min_group_size=args.add_min_group_size,
            add_entropy_min=args.add_entropy_min,
            add_cohesion_min=args.add_cohesion_min,
            add_cooldown_iters=args.add_cooldown_iters,
            rollback_on_metric_degrade=args.rollback_on_metric_degrade,
            accept_metric=args.accept_metric,
            accept_min_delta=args.accept_min_delta,
            noop_patience=args.noop_patience,
            best_operation_per_iteration=args.best_operation_per_iteration,
            max_same_operation_streak=args.max_same_operation_streak,
            structural_label_mode=args.structural_label_mode,
            split_label_pair_attempts=args.split_label_pair_attempts,
            llm_label_max_attempts=args.llm_label_max_attempts,
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
    parser.add_argument("--subsample_seed", type=int, default=13, help="Seed for per-cluster subsampling. Use -1 for non-deterministic sampling.")
    parser.add_argument("--baseline_sample_size", type=int, default=DEFAULT_BASELINE_SAMPLE_SIZE, help="Sample size for calculating baseline P(x) during EM refinement.")
    parser.add_argument("--max_texts_per_cluster_metrics", type=int, default=DEFAULT_MAX_TEXTS_PER_CLUSTER_METRICS, help="Max texts per cluster to use for metrics calculation (-1 for no limit).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of probabilities at each EM iteration.")
    parser.add_argument("--show_progress", action="store_true", help="Show tqdm progress bars.")
    parser.add_argument("--log_iteration_metrics", action="store_true", help="Log ELBO/AIC/BIC per iteration.")
    parser.add_argument("--log_top_pairs", type=int, default=0, help="Log top-N merge candidate similarities per iteration.")
    parser.add_argument("--diagnostics_sample_size", type=int, default=200, help="Number of rows to sample for probability diagnostics per iteration.")
    parser.add_argument("--diagnostics_bins", type=int, default=40, help="Histogram bins for diagnostics plots.")
    parser.add_argument("--diagnostics_dir", type=str, default="", help="Output directory for EM diagnostics (defaults to <experiment_dir>/em_diagnostics).")

    parser.add_argument(
        "--tune_operation",
        type=str,
        choices=list(VALID_TUNE_OPERATIONS),
        default="all",
        help="Run all operations or isolate one operation for threshold tuning.",
    )
    parser.add_argument(
        "--structural_label_mode",
        type=str,
        choices=list(VALID_STRUCTURAL_LABEL_MODES),
        default="semantic",
        help=(
            "Labeling mode for structural ops (split/merge): "
            "'semantic' uses LLM-generated labels, 'parent_relative' uses deterministic labels like "
            "'<parent> Parent Zk' / '<parent> Child Zk' to reduce rename noise."
        ),
    )
    parser.add_argument(
        "--split_label_pair_attempts",
        type=int,
        default=3,
        help="Number of parent/child pair relabel retries for each split before reverting that split pair.",
    )
    parser.add_argument(
        "--llm_label_max_attempts",
        type=int,
        default=6,
        help="Max internal retries per LLM labeling call before fallback.",
    )

    # Split config
    parser.add_argument(
        "--split_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable split operation logic.",
    )
    parser.add_argument("--split_max_per_iter", type=int, default=1)
    parser.add_argument("--split_min_cluster_size", type=int, default=20)
    parser.add_argument("--split_ll_margin", type=float, default=6.0)
    parser.add_argument("--split_confidence_max", type=float, default=0.30)
    parser.add_argument("--split_gap_median_min", type=float, default=0.10)
    parser.add_argument("--split_min_conditions", type=int, default=1, help="Min number of split conditions that must pass (L_z/C_z/gap) once base gates pass.")
    parser.add_argument("--split_cooldown_iters", type=int, default=1)

    # Merge config
    parser.add_argument(
        "--merge_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable merge operation logic.",
    )
    parser.add_argument("--merge_max_per_iter", type=int, default=1)
    parser.add_argument("--merge_similarity_min", type=float, default=0.15)
    parser.add_argument("--merge_l_diff_ratio_max", type=float, default=0.10)
    parser.add_argument("--merge_c_diff_ratio_max", type=float, default=0.10)
    parser.add_argument("--merge_min_conditions", type=int, default=1, help="Min number of merge conditions that must pass (similarity/L_ratio/C_ratio).")

    # Remove config
    parser.add_argument(
        "--remove_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable remove operation logic.",
    )
    parser.add_argument("--remove_max_per_iter", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--remove_min_cluster_size", type=int, default=5)
    parser.add_argument("--remove_ll_factor", type=float, default=1.01)

    # Revise config
    parser.add_argument(
        "--revise_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable revise operation logic (relabel low-quality clusters without reassigning datapoints).",
    )
    parser.add_argument("--revise_max_per_iter", type=int, default=1, help="0 means no cap.")
    parser.add_argument("--revise_ll_margin", type=float, default=4.0)
    parser.add_argument("--revise_confidence_max", type=float, default=0.25)
    parser.add_argument("--revise_min_conditions", type=int, default=1, help="Min number of revise conditions that must pass (L_z/C_z).")
    parser.add_argument("--revise_cooldown_iters", type=int, default=1)

    # Add config
    parser.add_argument(
        "--add_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable add operation logic.",
    )
    parser.add_argument("--add_low_confidence_max", type=float, default=0.20)
    parser.add_argument("--add_low_confidence_quantile", type=float, default=0.05, help="Dynamic low-confidence quantile for add-step; threshold uses max(fixed, quantile).")
    parser.add_argument("--add_min_poorly_explained", type=int, default=5)
    parser.add_argument("--add_items_per_new_cluster", type=int, default=50)
    parser.add_argument("--add_max_new_clusters_per_iter", type=int, default=3)
    parser.add_argument("--add_max_total_clusters", type=int, default=200)
    parser.add_argument("--add_min_group_size", type=int, default=4)
    parser.add_argument("--add_entropy_min", type=float, default=0.90)
    parser.add_argument("--add_cohesion_min", type=float, default=0.15)
    parser.add_argument("--add_cooldown_iters", type=int, default=1)
    parser.add_argument("--rollback_on_metric_degrade", action=argparse.BooleanOptionalAction, default=True, help="Rollback iteration updates when the selected accept metric degrades.")
    parser.add_argument("--accept_metric", type=str, choices=list(VALID_ACCEPT_METRICS), default="assigned", help="Metric variant used for rollback gating: assigned, soft, or oracle.")
    parser.add_argument("--accept_min_delta", type=float, default=0.0, help="Minimum delta in the selected accept metric (avg_logL_per_token_<variant>) required to accept an iteration update.")
    parser.add_argument("--noop_patience", type=int, default=2, help="Early-stop after this many consecutive no-op iterations.")
    parser.add_argument(
        "--best_operation_per_iteration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled with tune_operation=all, evaluate split/merge/remove/add separately each iteration and apply only the best-scoring operation.",
    )
    parser.add_argument(
        "--max_same_operation_streak",
        type=int,
        default=0,
        help="When best_operation_per_iteration is enabled, disallow picking the same winning operation more than this many consecutive iterations (0 disables).",
    )

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
