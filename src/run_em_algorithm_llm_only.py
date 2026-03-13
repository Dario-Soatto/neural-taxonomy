from __future__ import annotations

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
VALID_LABEL_PROMPT_STYLES = ("generic", "topic", "discourse", "biography")
VALID_LABEL_PROMPT_STYLE_ARGS = ("auto",) + VALID_LABEL_PROMPT_STYLES
VALID_SPLIT_SIZE_GATE_MODES = ("fixed", "mean_ratio", "off")
VALID_MERGE_EPSILON_MODES = ("fixed", "quantile")
VALID_MERGE_SIMILARITY_MODES = ("fixed", "quantile")
DUP_LABEL_SIM_THRESHOLD = 0.90   # Near-duplicate label similarity threshold
MERGE_RECIPROCAL_SHARE_MIN = 0.30
MERGE_FRAGMENT_MEAN_RATIO_MAX = 1.00
MERGE_LABEL_OVERLAP_MIN_ONE_WAY = 0.25
LABEL_TOKEN_STOPWORDS = {
    "cluster",
    "topic",
    "subtopic",
    "misc",
    "other",
    "variant",
    "core",
    "parent",
    "child",
    "discussion",
    "custom",
    "sale",
    "sales",
    "talk",
    "rec",
    "comp",
    "sci",
}


def infer_label_prompt_style(experiment_name: str | Path | None) -> str:
    experiment_key = sanitize_text(str(experiment_name or "")).lower()
    if any(token in experiment_key for token in ("newsgroup", "20news", "news")):
        return "topic"
    if any(token in experiment_key for token in ("wiki", "biograph", "person", "people")):
        return "biography"
    if any(token in experiment_key for token in ("editorial", "discourse", "rhetor", "argument", "debate", "opinion")):
        return "discourse"
    return "generic"


def _label_token_set(label: str | None) -> set[str]:
    canonical = _canonical_label_key(label or "")
    if not canonical:
        return set()
    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", canonical)
        if tok and tok not in LABEL_TOKEN_STOPWORDS
    }
    return tokens


def _label_token_overlap(label_a: str | None, label_b: str | None) -> float:
    toks_a = _label_token_set(label_a)
    toks_b = _label_token_set(label_b)
    if not toks_a or not toks_b:
        return 0.0
    union = toks_a | toks_b
    if not union:
        return 0.0
    return float(len(toks_a & toks_b) / len(union))


def _log_text_preview(
    *,
    header: str,
    texts: list[str],
    max_items: int = 8,
    max_chars: int = 180,
) -> None:
    cleaned = [str(t).replace("\n", " ").strip() for t in texts if sanitize_text(str(t))]
    if not cleaned:
        logging.info("%s: no texts to preview.", header)
        return
    logging.info("%s (%s texts, showing up to %s):", header, len(cleaned), max_items)
    for idx, text in enumerate(cleaned[:max_items], start=1):
        preview = text[:max_chars]
        if len(text) > max_chars:
            preview += "..."
        logging.info("    [%s] %s", idx, preview)


def build_label_prompt_components(
    *,
    label_prompt_style: str,
    parent_label: str | None,
    dynamic_banned_keys: set[str],
    has_contrast_examples: bool,
) -> tuple[str, str, list[str]]:
    guidance_lines = [
        "Produce a semantically specific, standalone cluster label.",
        "Avoid generic labels such as Subtopic, Variant, Topic, Misc, Other.",
        "Return ONLY valid JSON (no markdown, no preamble).",
    ]

    if label_prompt_style == "topic":
        system_prompt = "You are an analyst labeling topic clusters from documents or posts."
        task_intro = "Given these example texts from one cluster, generate a concise topical label."
        guidance_lines.extend(
            [
                "Name must be 2-5 words.",
                "Name should describe the main topic or subject matter shared by the texts.",
                "Prefer concrete topical labels over discourse-role labels.",
            ]
        )
    elif label_prompt_style == "biography":
        system_prompt = "You are an analyst labeling biography-related text clusters."
        task_intro = "Given these example texts from one cluster, generate a concise biographical theme label."
        guidance_lines.extend(
            [
                "Name must be 2-5 words.",
                "Name should describe the shared life event, role, era, or biographical theme.",
                "Avoid using a specific person's name unless it is clearly the cluster's shared subject.",
            ]
        )
    elif label_prompt_style == "discourse":
        system_prompt = "You are an analyst labeling discourse-function clusters."
        task_intro = "Given these example texts from one cluster, generate a concise discourse-function label."
        guidance_lines.extend(
            [
                "Name must be 2-5 words.",
                "Name should describe discourse function, communicative intent, or rhetorical role, not entities.",
            ]
        )
    else:
        system_prompt = "You are an experienced analyst."
        task_intro = "Given these example texts from one cluster, generate a concise cluster label."
        guidance_lines.extend(
            [
                "Name must be 2-5 words.",
                "Name should describe the main shared topic, theme, or function of the texts.",
            ]
        )

    if parent_label:
        guidance_lines.append(f"Parent context label: '{parent_label}'.")
    if has_contrast_examples:
        guidance_lines.append("Ensure the name is distinct from the contrast cluster examples.")
    if dynamic_banned_keys:
        banned_list = ", ".join(sorted(dynamic_banned_keys))
        guidance_lines.append(f"Do NOT use any name equal to or derived from: {banned_list}.")

    return system_prompt, task_intro, guidance_lines


@dataclass
class SchemaRefinementConfig:
    tune_operation: str = "all"

    split_enabled: bool = True
    split_max_per_iter: int = 1
    split_min_cluster_size: int = 20
    split_size_gate_mode: str = "fixed"
    split_min_cluster_size_mean_ratio: float = 0.50
    split_l_bottom_quantile: float = 0.25
    split_c_bottom_quantile: float = 0.25
    split_v_top_quantile: float = 0.75
    split_ll_margin: float = 6.0
    split_confidence_max: float = 0.30
    split_gap_median_min: float = 0.10
    split_min_conditions: int = 1
    split_cooldown_iters: int = 1
    split_gap_top2_max: float = 0.25
    split_runner_up_min_share: float = 0.30

    merge_enabled: bool = True
    merge_max_per_iter: int = 1
    merge_similarity_min: float = 0.15  # confusion-based threshold (symmetric avg p(z_j|x) between clusters)
    merge_similarity_mode: str = "fixed"
    merge_similarity_quantile: float = 0.75
    merge_epsilon_mode: str = "fixed"
    merge_l_abs_diff_quantile: float = 0.25
    merge_c_abs_diff_quantile: float = 0.25
    merge_l_abs_diff_max: float = 12.0
    merge_c_abs_diff_max: float = 0.10
    merge_l_diff_ratio_max: float = 0.10
    merge_c_diff_ratio_max: float = 0.10
    merge_min_conditions: int = 1

    remove_enabled: bool = True
    remove_max_per_iter: int = 0
    remove_min_cluster_size: int = 5
    remove_min_cluster_size_mean_ratio: float = 0.30
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
    label_prompt_style: str = "generic"
    split_label_pair_attempts: int = 3
    llm_label_max_attempts: int = 6
    proposal_proportional_thresholds: bool = False
    proposal_adaptive_thresholds: bool = False
    adaptive_threshold_max_relax_iters: int = 3
    adaptive_quantile_step: float = 0.08
    adaptive_split_size_relax_ratio: float = 0.20
    adaptive_merge_similarity_relax: float = 0.10
    adaptive_merge_epsilon_expand: float = 0.30
    adaptive_remove_size_relax_ratio: float = 0.20
    adaptive_remove_ll_factor_step: float = 0.01
    proposal_top_k_log: int = 0

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
    texts: list[str] | None = None,
    min_runner_up_dominance: float = 0.3,
) -> np.ndarray:
    """Split a cluster using LLM posteriors (no sentence-transformer embeddings).

    Candidate A: Split by runner-up disagreement.
      For each doc, find its best cluster AFTER masking the assigned cluster.
      Group docs by whether they prefer the dominant runner-up or not.

    Candidate B: KMeans(k=2) on masked posterior vectors.

    We score both valid 2-way partitions and keep the stronger one. This avoids
    over-trusting a single existing runner-up label when the latent split is not
    aligned with any one currently available cluster.

    Returns array of 0/1 labels for the 2-way split.
    """
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

    n = PZX_cluster.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    masked_pzx = np.asarray(PZX_cluster, dtype=float).copy()
    if 0 <= int(assigned_col_idx) < masked_pzx.shape[1]:
        masked_pzx[:, int(assigned_col_idx)] = 0.0
    masked_row_sums = masked_pzx.sum(axis=1, keepdims=True)
    valid_rows = masked_row_sums.squeeze(-1) > 0
    if np.any(valid_rows):
        masked_pzx[valid_rows] = masked_pzx[valid_rows] / masked_row_sums[valid_rows]
    else:
        masked_pzx = np.asarray(PZX_cluster, dtype=float)

    text_matrix = None
    if texts:
        cleaned_texts = [str(t or "").replace("\n", " ").strip() for t in texts]
        if len(cleaned_texts) == n and any(cleaned_texts):
            try:
                text_vectorizer = _TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=2000,
                    min_df=1,
                )
                text_matrix = text_vectorizer.fit_transform(cleaned_texts)
            except Exception as exc:
                logging.warning("  Split candidate text-TFIDF build failed: %s", exc)
                text_matrix = None

    def _text_partition_score(labels: np.ndarray) -> float:
        if text_matrix is None:
            return 0.0
        labels = np.asarray(labels, dtype=int)
        mask0 = labels == 0
        mask1 = labels == 1
        n0 = int(mask0.sum())
        n1 = int(mask1.sum())
        if n0 < 2 or n1 < 2:
            return float("-inf")

        group0 = text_matrix[mask0]
        group1 = text_matrix[mask1]
        centroid0 = np.asarray(group0.mean(axis=0), dtype=float)
        centroid1 = np.asarray(group1.mean(axis=0), dtype=float)
        own0 = float(np.mean(_cosine_similarity(group0, centroid0).ravel()))
        own1 = float(np.mean(_cosine_similarity(group1, centroid1).ravel()))
        cross01 = float(np.mean(_cosine_similarity(group0, centroid1).ravel()))
        cross10 = float(np.mean(_cosine_similarity(group1, centroid0).ravel()))
        centroid_sep = 1.0 - float(_cosine_similarity(centroid0, centroid1).ravel()[0])
        margin = 0.5 * ((own0 - cross01) + (own1 - cross10))
        return float(max(0.0, centroid_sep) + max(0.0, margin))

    def _score_partition(labels: np.ndarray) -> float:
        labels = np.asarray(labels, dtype=int)
        if labels.shape[0] != n:
            return float("-inf")
        mask0 = labels == 0
        mask1 = labels == 1
        n0 = int(mask0.sum())
        n1 = int(mask1.sum())
        if n0 < 2 or n1 < 2:
            return float("-inf")

        group0 = masked_pzx[mask0]
        group1 = masked_pzx[mask1]
        cohesion0 = kl_divergence_cohesion(group0)
        cohesion1 = kl_divergence_cohesion(group1)
        centroid0 = np.asarray(group0.mean(axis=0), dtype=float)
        centroid1 = np.asarray(group1.mean(axis=0), dtype=float)
        separation = float(np.linalg.norm(centroid0 - centroid1, ord=1) / 2.0)
        balance = float(min(n0, n1) / max(n0, n1))
        peak_bonus = 0.25 if int(np.argmax(centroid0)) != int(np.argmax(centroid1)) else 0.0
        posterior_score = float(separation + 0.5 * (cohesion0 + cohesion1) + 0.25 * balance + peak_bonus)
        lexical_score = _text_partition_score(labels)
        if not np.isfinite(lexical_score):
            return float("-inf")
        return float(posterior_score + 0.75 * lexical_score)

    candidate_partitions: list[tuple[str, np.ndarray, float]] = []

    # --- Candidate A: runner-up split ---
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
            runner_up_score = _score_partition(labels)
            if np.isfinite(runner_up_score):
                candidate_partitions.append(("runner_up", labels, runner_up_score))
                logging.info(
                    "  Split candidate runner-up: dominant runner-up cluster idx %s has %.1f%% of docs (score=%.4f).",
                    top_runner,
                    dominance * 100,
                    runner_up_score,
                )

    # --- Candidate B: KMeans on masked posteriors ---
    kmeans = _KMeans(n_clusters=2, n_init="auto", random_state=42)
    kmeans_labels = kmeans.fit_predict(masked_pzx)
    kmeans_score = _score_partition(kmeans_labels)
    if np.isfinite(kmeans_score):
        candidate_partitions.append(("kmeans", kmeans_labels, kmeans_score))
        logging.info(
            "  Split candidate kmeans: posterior KMeans score=%.4f.",
            kmeans_score,
        )

    # --- Candidate C: KMeans on TF-IDF text features ---
    if text_matrix is not None and n >= 4:
        try:
            tfidf_kmeans = _KMeans(n_clusters=2, n_init="auto", random_state=42)
            tfidf_labels = tfidf_kmeans.fit_predict(text_matrix)
            tfidf_score = _score_partition(tfidf_labels)
            if np.isfinite(tfidf_score):
                candidate_partitions.append(("tfidf_kmeans", tfidf_labels, tfidf_score))
                logging.info(
                    "  Split candidate tfidf_kmeans: text-TFIDF KMeans score=%.4f.",
                    tfidf_score,
                )
        except Exception as exc:
            logging.warning("  Split candidate tfidf_kmeans failed: %s", exc)

    if candidate_partitions:
        method_name, best_labels, best_score = max(candidate_partitions, key=lambda item: float(item[2]))
        logging.info("  Selected split partition method=%s (score=%.4f).", method_name, best_score)
        return np.asarray(best_labels, dtype=int)

    logging.info("  Split partition scoring fell back to KMeans labels.")
    return np.asarray(kmeans_labels, dtype=int)


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
                'runner_up_dominant_idx': -1,
                'runner_up_dominant_share': 0.0,
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
        runner_up_dominant_idx = -1
        runner_up_dominant_share = 0.0
        if label_idx is not None and ll_subset.shape[1] > 1:
            masked_ll = ll_subset.copy()
            masked_ll[:, label_idx] = -np.inf
            runner_ups = masked_ll.argmax(axis=1)
            if runner_ups.size > 0:
                unique_runners, counts = np.unique(runner_ups, return_counts=True)
                top_pos = int(np.argmax(counts))
                runner_up_dominant_idx = int(unique_runners[top_pos])
                runner_up_dominant_share = float(counts[top_pos] / max(1, runner_ups.size))
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
            'runner_up_dominant_idx': runner_up_dominant_idx,
            'runner_up_dominant_share': runner_up_dominant_share,
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
    independent_operation_proposals: bool = False,
    proposal_relax_level: int = 0,
    verbose: bool = False,
    log_top_pairs: int = 0,
):
    """Determine which clusters to split, merge, remove. Returns dictionaries describing actions."""
    logging.info("Starting schema update decisions...")
    if independent_operation_proposals:
        logging.info(
            "Independent proposal mode enabled: split/merge/remove candidates are generated without cross-operation exclusions."
        )
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

    relax_level = 0
    if config.proposal_adaptive_thresholds:
        relax_level = max(0, int(proposal_relax_level))
        relax_level = min(relax_level, max(0, int(config.adaptive_threshold_max_relax_iters)))
        if relax_level > 0:
            logging.info(
                "Adaptive proposal thresholds active: relax_level=%s (raw=%s, max=%s).",
                relax_level,
                proposal_relax_level,
                config.adaptive_threshold_max_relax_iters,
            )

    l_vals = np.array([cluster_metrics[cid]['L_z'] for cid in valid_cluster_ids], dtype=float)
    c_vals = np.array([cluster_metrics[cid]['C_z'] for cid in valid_cluster_ids], dtype=float)
    v_vals = np.array([cluster_metrics[cid]['V_z'] for cid in valid_cluster_ids], dtype=float)
    median_V_z = float(np.median(v_vals))
    cluster_size_vals = np.array([cluster_metrics[cid]['size'] for cid in valid_cluster_ids], dtype=float)
    mean_cluster_size = float(np.mean(cluster_size_vals)) if len(cluster_size_vals) else 0.0

    split_size_gate_mode_effective = config.split_size_gate_mode
    if config.proposal_proportional_thresholds and split_size_gate_mode_effective == "fixed":
        split_size_gate_mode_effective = "mean_ratio"
    if split_size_gate_mode_effective == "off":
        split_min_size_effective = 1
    elif split_size_gate_mode_effective == "mean_ratio":
        split_min_size_effective = max(
            1,
            int(np.ceil(mean_cluster_size * max(0.0, float(config.split_min_cluster_size_mean_ratio)))),
        )
    else:
        split_min_size_effective = max(1, int(config.split_min_cluster_size))

    if relax_level > 0:
        split_size_scale = max(
            0.10,
            1.0 - max(0.0, float(config.adaptive_split_size_relax_ratio)) * float(relax_level),
        )
        split_min_size_effective = max(1, int(np.ceil(split_min_size_effective * split_size_scale)))

    split_l_q_eff = float(np.clip(config.split_l_bottom_quantile, 0.0, 1.0))
    split_c_q_eff = float(np.clip(config.split_c_bottom_quantile, 0.0, 1.0))
    split_v_q_eff = float(np.clip(config.split_v_top_quantile, 0.0, 1.0))
    if relax_level > 0:
        q_step = max(0.0, float(config.adaptive_quantile_step)) * float(relax_level)
        split_l_q_eff = float(np.clip(split_l_q_eff + q_step, 0.0, 1.0))
        split_c_q_eff = float(np.clip(split_c_q_eff + q_step, 0.0, 1.0))
        split_v_q_eff = float(np.clip(split_v_q_eff - q_step, 0.0, 1.0))

    split_l_thresh = float(np.quantile(l_vals, split_l_q_eff))
    split_c_thresh = float(np.quantile(c_vals, split_c_q_eff))
    split_v_thresh = float(np.quantile(v_vals, split_v_q_eff))

    remove_min_cluster_size_effective = int(config.remove_min_cluster_size)
    if config.proposal_proportional_thresholds:
        remove_min_cluster_size_effective = max(
            2,
            int(np.ceil(mean_cluster_size * max(0.0, float(config.remove_min_cluster_size_mean_ratio)))),
        )
    if relax_level > 0:
        remove_min_cluster_size_effective = max(
            1,
            int(
                np.ceil(
                    remove_min_cluster_size_effective
                    * (
                        1.0
                        + max(0.0, float(config.adaptive_remove_size_relax_ratio))
                        * float(relax_level)
                    )
                )
            ),
        )
    remove_ll_factor_effective = float(config.remove_ll_factor)
    if relax_level > 0:
        remove_ll_factor_effective = max(
            0.50,
            remove_ll_factor_effective
            - max(0.0, float(config.adaptive_remove_ll_factor_step)) * float(relax_level),
        )
    remove_ll_threshold = float(decision_baseline * remove_ll_factor_effective)

    merge_epsilon_mode_effective = config.merge_epsilon_mode
    if config.proposal_proportional_thresholds and merge_epsilon_mode_effective == "fixed":
        merge_epsilon_mode_effective = "quantile"
    merge_l_q_eff = float(np.clip(config.merge_l_abs_diff_quantile, 0.0, 1.0))
    merge_c_q_eff = float(np.clip(config.merge_c_abs_diff_quantile, 0.0, 1.0))
    if relax_level > 0 and merge_epsilon_mode_effective == "quantile":
        q_step = max(0.0, float(config.adaptive_quantile_step)) * float(relax_level)
        merge_l_q_eff = float(np.clip(merge_l_q_eff + q_step, 0.0, 1.0))
        merge_c_q_eff = float(np.clip(merge_c_q_eff + q_step, 0.0, 1.0))

    merge_similarity_mode_effective = str(config.merge_similarity_mode).lower()
    if config.proposal_proportional_thresholds and merge_similarity_mode_effective == "fixed":
        merge_similarity_mode_effective = "quantile"
    merge_similarity_quantile_effective = float(np.clip(config.merge_similarity_quantile, 0.0, 1.0))
    merge_similarity_min_effective = float(config.merge_similarity_min)
    if relax_level > 0:
        if merge_similarity_mode_effective == "quantile":
            q_step = max(0.0, float(config.adaptive_quantile_step)) * float(relax_level)
            merge_similarity_quantile_effective = float(
                np.clip(merge_similarity_quantile_effective - q_step, 0.0, 1.0)
            )
        else:
            merge_similarity_min_effective = max(
                0.0,
                merge_similarity_min_effective
                * (
                    1.0
                    - max(0.0, float(config.adaptive_merge_similarity_relax))
                    * float(relax_level)
                ),
            )

    top_k = max(0, int(config.proposal_top_k_log))
    logging.info(
        "Using decision baseline L_baseline_assigned=%.4f (and log p(x) baseline=%.4f), Median V_z=%.4f. "
        "tune_operation=%s; enabled ops: split=%s merge=%s remove=%s revise=%s. "
        "proposal_proportional=%s adaptive=%s relax_level=%s top_k=%s. "
        "remove: size<%s (effective=%s) and L_z<%.4f (factor=%.4f). "
        "split: size_gate(mode=%s, effective_min=%s, mean_size=%.2f) AND "
        "structural=[runner_up_share in [%.2f, %.2f] OR gap<=%.4f] AND "
        "(L_z<=Q%.2f=%.4f OR C_z<=Q%.2f=%.4f OR V_z>=Q%.2f=%.4f), cooldown=%s. "
        "merge: similarity mode=%s (base min=%.4f, q=%.2f) epsilon mode=%s (base |L|<%.4f, |C|<%.4f) "
        "AND partner_support>=%.2f with fragment-aware relaxation for min_size<=%.2fx mean_size "
        "and one-way merges requiring label_overlap>=%.2f. "
        "revise: min_conditions=%s over [L_z<%.4f, C_z<%.4f], cooldown=%s.",
        decision_baseline,
        baseline_log_px,
        median_V_z,
        config.tune_operation,
        split_enabled,
        merge_enabled,
        remove_enabled,
        revise_enabled,
        config.proposal_proportional_thresholds,
        config.proposal_adaptive_thresholds,
        relax_level,
        top_k,
        config.remove_min_cluster_size,
        remove_min_cluster_size_effective,
        remove_ll_threshold,
        remove_ll_factor_effective,
        split_size_gate_mode_effective,
        split_min_size_effective,
        mean_cluster_size,
        config.split_runner_up_min_share,
        1.0 - float(config.split_runner_up_min_share),
        config.split_gap_top2_max,
        split_l_q_eff,
        split_l_thresh,
        split_c_q_eff,
        split_c_thresh,
        split_v_q_eff,
        split_v_thresh,
        config.split_cooldown_iters,
        merge_similarity_mode_effective,
        config.merge_similarity_min,
        merge_similarity_quantile_effective,
        merge_epsilon_mode_effective,
        config.merge_l_abs_diff_max,
        config.merge_c_abs_diff_max,
        MERGE_RECIPROCAL_SHARE_MIN,
        MERGE_FRAGMENT_MEAN_RATIO_MAX,
        MERGE_LABEL_OVERLAP_MIN_ONE_WAY,
        config.revise_min_conditions,
        decision_baseline - config.revise_ll_margin,
        config.revise_confidence_max,
        config.revise_cooldown_iters,
    )

    # Track margins to understand how close we are to thresholds
    split_l_margins = []
    split_c_margins = []
    split_v_margins = []
    remove_margins = []
    split_candidate_rows: list[dict] = []
    remove_candidate_rows: list[dict] = []
    merge_candidate_rows: list[dict] = []

    # --- SPLIT & REMOVE ---
    for cid in valid_cluster_ids:
        m = cluster_metrics[cid]
        remove_size_margin = float(remove_min_cluster_size_effective - int(m['size']))
        remove_ll_margin = float(remove_ll_threshold - float(m['L_z']))
        remove_pass_count = int(remove_size_margin > 0.0) + int(remove_ll_margin > 0.0)
        remove_candidate_rows.append(
            {
                "cid": int(cid),
                "label": m.get('label', 'N/A'),
                "size_margin": remove_size_margin,
                "ll_margin": remove_ll_margin,
                "pass_count": remove_pass_count,
                "passes": remove_pass_count == 2,
            }
        )

        split_size_margin = float(int(m['size']) - split_min_size_effective)
        split_l_margin = float(split_l_thresh - float(m['L_z']))
        split_c_margin = float(split_c_thresh - float(m['C_z']))
        split_v_margin = float(float(m['V_z']) - split_v_thresh)
        split_cond_count = int(split_l_margin >= 0.0) + int(split_c_margin >= 0.0) + int(split_v_margin >= 0.0)
        cooldown_until = split_cooldown.get(cid)
        in_cooldown = cooldown_until is not None and iter_idx <= cooldown_until
        gap_median = float(m.get('gap_median', 0.0))
        gap_margin = float(config.split_gap_top2_max - gap_median)
        runner_up_share = float(m.get('runner_up_dominant_share', 0.0))
        runner_up_lower = float(config.split_runner_up_min_share)
        runner_up_upper = float(1.0 - runner_up_lower)
        runner_up_viable = bool(
            int(m['size']) >= 4
            and runner_up_lower <= runner_up_share <= runner_up_upper
        )
        structural_signal_count = int(runner_up_viable) + int(gap_margin >= 0.0)
        split_pass_count = int(split_size_margin >= 0.0) + split_cond_count + structural_signal_count
        runner_up_balance_score = (
            max(0.0, 1.0 - (abs(runner_up_share - 0.5) / 0.5))
            if runner_up_viable else 0.0
        )
        split_score = float(
            (2.5 * runner_up_balance_score)
            + (1.5 if runner_up_viable else 0.0)
            + (0.75 if gap_margin >= 0.0 else 0.0)
            + max(0.0, split_l_margin)
            + max(0.0, split_c_margin)
            + max(0.0, split_v_margin)
            + max(0.0, split_size_margin) / max(mean_cluster_size, 1.0)
        )
        split_passes = (
            (not in_cooldown)
            and (split_size_margin >= 0.0)
            and (
                runner_up_viable
                or ((gap_margin >= 0.0) and (split_cond_count >= max(1, int(config.split_min_conditions))))
            )
        )
        split_candidate_rows.append(
            {
                "cid": int(cid),
                "label": m.get('label', 'N/A'),
                "size_margin": split_size_margin,
                "L_margin": split_l_margin,
                "C_margin": split_c_margin,
                "V_margin": split_v_margin,
                "gap_margin": gap_margin,
                "gap_median": gap_median,
                "runner_up_share": runner_up_share,
                "runner_up_viable": runner_up_viable,
                "runner_up_dominant_idx": int(m.get('runner_up_dominant_idx', -1)),
                "runner_up_balance_score": runner_up_balance_score,
                "structural_signal_count": structural_signal_count,
                "cond_count": split_cond_count,
                "pass_count": split_pass_count,
                "in_cooldown": in_cooldown,
                "passes": split_passes,
                "score": split_score,
            }
        )

        # Per-cluster threshold diagnostics
        logging.info(
            "Cluster %s ('%s'): L_z=%.4f, C_z=%.4f, V_z=%.4f, size=%s. "
            "remove_thresh=%.4f, split_size_min=%s, split_q_thresholds=[L<=%.4f, C<=%.4f, V>=%.4f], "
            "gap_median=%.4f, runner_up_share=%.4f",
            cid,
            m.get('label', 'N/A'),
            m['L_z'],
            m['C_z'],
            m['V_z'],
            m['size'],
            remove_ll_threshold,
            split_min_size_effective,
            split_l_thresh,
            split_c_thresh,
            split_v_thresh,
            gap_median,
            runner_up_share,
        )
        # Removal Check
        remove_triggered = (
            remove_enabled
            and m['size'] < remove_min_cluster_size_effective
            and m['L_z'] < remove_ll_threshold
        )
        if remove_triggered:
            actions['remove'].append(cid)
            logging.info(
                "  Suggest REMOVE for cluster %s ('%s'): size %s < %s AND L_z %.4f < baseline_thresh %.4f.",
                cid,
                m.get('label', 'N/A'),
                m['size'],
                remove_min_cluster_size_effective,
                m['L_z'],
                remove_ll_threshold,
            )
            if not independent_operation_proposals:
                continue # Legacy coupled mode: if removed, don't consider for split
        remove_margins.append(m['L_z'] - remove_ll_threshold)
        # Split Check (gated)
        if split_enabled:
            split_conditions = [
                ("L_bottom_quantile", m['L_z'] <= split_l_thresh),
                ("C_bottom_quantile", m['C_z'] <= split_c_thresh),
                ("V_top_quantile", m['V_z'] >= split_v_thresh),
            ]
            passed_names = [name for name, cond in split_conditions if cond]
            structural_names = []
            if runner_up_viable:
                structural_names.append("runner_up_concentration")
            if gap_margin >= 0.0:
                structural_names.append("low_gap")
            if split_passes:
                logging.info(
                    f"  Suggest SPLIT for cluster {cid} ('{m.get('label', 'N/A')}'): "
                    f"structural={structural_names} quality={passed_names} "
                    f"runner_up_share={runner_up_share:.3f} gap_median={gap_median:.3f} score={split_score:.4f}."
                )
        split_l_margins.append(m['L_z'] - split_l_thresh)
        split_c_margins.append(m['C_z'] - split_c_thresh)
        split_v_margins.append(split_v_thresh - m['V_z'])

    # --- MERGE (LLM-only: confusion matrix from posteriors) ---
    top_pairs = []
    # Legacy coupled mode filtered split/remove clusters out of merge candidacy.
    # In independent proposal mode we keep merge candidacy independent of split/remove triggers.
    if independent_operation_proposals:
        eligible_for_merge_ids = list(valid_cluster_ids)
    else:
        eligible_for_merge_ids = [
            cid
            for cid in valid_cluster_ids
            if cid not in actions['remove'] and cid not in actions['split']
        ]
    effective_merge_l_eps = float(config.merge_l_abs_diff_max)
    effective_merge_c_eps = float(config.merge_c_abs_diff_max)
    if relax_level > 0 and merge_epsilon_mode_effective == "fixed":
        expansion = 1.0 + max(0.0, float(config.adaptive_merge_epsilon_expand)) * float(relax_level)
        effective_merge_l_eps = float(effective_merge_l_eps * expansion)
        effective_merge_c_eps = float(effective_merge_c_eps * expansion)
    if merge_epsilon_mode_effective == "quantile" and len(eligible_for_merge_ids) >= 2:
        pair_diff_l: list[float] = []
        pair_diff_c: list[float] = []
        for i in range(len(eligible_for_merge_ids)):
            for j in range(i + 1, len(eligible_for_merge_ids)):
                m_i = cluster_metrics[eligible_for_merge_ids[i]]
                m_j = cluster_metrics[eligible_for_merge_ids[j]]
                pair_diff_l.append(abs(float(m_i['L_z']) - float(m_j['L_z'])))
                pair_diff_c.append(abs(float(m_i['C_z']) - float(m_j['C_z'])))
        if pair_diff_l:
            effective_merge_l_eps = float(
                np.quantile(
                    np.array(pair_diff_l, dtype=float),
                    merge_l_q_eff,
                )
            )
            effective_merge_c_eps = float(
                np.quantile(
                    np.array(pair_diff_c, dtype=float),
                    merge_c_q_eff,
                )
            )
    confusion = cluster_metrics.get('_confusion_matrix')
    cid_to_choice_idx = cluster_metrics.get('_confusion_cid_to_choice_idx', {})
    choice_idx_to_cid = {int(choice_idx): int(cid) for cid, choice_idx in cid_to_choice_idx.items()}
    pair_sim_values: list[float] = []
    if confusion is not None and len(eligible_for_merge_ids) >= 2:
        for i in range(len(eligible_for_merge_ids)):
            for j in range(i + 1, len(eligible_for_merge_ids)):
                ci = cid_to_choice_idx.get(eligible_for_merge_ids[i], -1)
                cj = cid_to_choice_idx.get(eligible_for_merge_ids[j], -1)
                if ci < 0 or cj < 0:
                    continue
                pair_sim_values.append(float((confusion[ci, cj] + confusion[cj, ci]) / 2.0))
    if merge_similarity_mode_effective == "quantile" and pair_sim_values:
        merge_similarity_min_effective = float(
            np.quantile(np.asarray(pair_sim_values, dtype=float), merge_similarity_quantile_effective)
        )
    logging.info(
        "Effective merge thresholds: sim(mode=%s)=>%.4f, eps(mode=%s)=>|L_a-L_b|<%.4f, |C_a-C_b|<%.4f",
        merge_similarity_mode_effective,
        merge_similarity_min_effective,
        merge_epsilon_mode_effective,
        effective_merge_l_eps,
        effective_merge_c_eps,
    )
    if not merge_enabled:
        logging.info("Merge operation disabled for this run.")
    elif len(eligible_for_merge_ids) < 2:
        logging.info("Not enough eligible clusters to consider merging.")
    elif confusion is None:
        logging.warning("No confusion matrix available for merge decisions. Skipping merge.")
    else:
        n = len(eligible_for_merge_ids)
        for i in range(n):
            for j in range(i + 1, n):
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
                label_overlap = _label_token_overlap(m_i.get("label", ""), m_j.get("label", ""))
                size_i = int(m_i.get('size', 0))
                size_j = int(m_j.get('size', 0))
                partner_i_choice = int(m_i.get('runner_up_dominant_idx', -1))
                partner_j_choice = int(m_j.get('runner_up_dominant_idx', -1))
                partner_i_cid = choice_idx_to_cid.get(partner_i_choice, -1)
                partner_j_cid = choice_idx_to_cid.get(partner_j_choice, -1)
                partner_match_i = int(partner_i_cid) == int(id_j)
                partner_match_j = int(partner_j_cid) == int(id_i)
                partner_share_i = float(m_i.get('runner_up_dominant_share', 0.0)) if partner_match_i else 0.0
                partner_share_j = float(m_j.get('runner_up_dominant_share', 0.0)) if partner_match_j else 0.0
                reciprocal_partner = bool(partner_match_i and partner_match_j)
                mutual_partner_share = float(min(partner_share_i, partner_share_j)) if reciprocal_partner else 0.0
                min_pair_size = int(min(size_i, size_j))
                fragment_size_cap = float(max(2.0, MERGE_FRAGMENT_MEAN_RATIO_MAX * mean_cluster_size))
                fragment_pair = bool(min_pair_size <= fragment_size_cap)
                pair_sim_threshold = float(merge_similarity_min_effective)
                pair_c_epsilon = float(effective_merge_c_eps)
                partner_support_threshold = float(MERGE_RECIPROCAL_SHARE_MIN)
                partner_support_share = float(mutual_partner_share)
                support_mode = "reciprocal"
                lexical_margin = float(label_overlap)
                if fragment_pair:
                    pair_sim_threshold = float(min(pair_sim_threshold, max(0.10, pair_sim_threshold * 0.75)))
                    pair_c_epsilon = float(max(pair_c_epsilon, 0.25))
                    if not reciprocal_partner:
                        partner_support_share = float(max(partner_share_i, partner_share_j))
                        partner_support_threshold = float(min(MERGE_RECIPROCAL_SHARE_MIN, 0.15))
                        support_mode = "one_way"
                        lexical_margin = float(label_overlap - MERGE_LABEL_OVERLAP_MIN_ONE_WAY)
                else:
                    lexical_margin = float(label_overlap - MERGE_LABEL_OVERLAP_MIN_ONE_WAY)
                sim_margin = float(sim_val - pair_sim_threshold)
                l_margin = float(effective_merge_l_eps - diff_L)
                c_margin = float(pair_c_epsilon - diff_C)
                partner_margin = float(partner_support_share - partner_support_threshold)
                size_margin = float(fragment_size_cap - float(min_pair_size))
                pass_count = (
                    int(sim_margin > 0.0)
                    + int(c_margin > 0.0)
                    + int(partner_margin >= 0.0)
                    + int(fragment_pair)
                    + int(reciprocal_partner or lexical_margin >= 0.0)
                )
                merge_passes = (
                    (sim_margin > 0.0)
                    and (c_margin > 0.0)
                    and (partner_margin >= 0.0)
                    and fragment_pair
                    and (reciprocal_partner or lexical_margin >= 0.0)
                )
                merge_score = float(
                    (1.25 * max(0.0, sim_margin))
                    + (1.75 * max(0.0, partner_margin))
                    + (0.25 if reciprocal_partner else 0.0)
                    + (0.20 * max(0.0, size_margin) / max(mean_cluster_size, 1.0))
                    + (0.35 * max(0.0, c_margin))
                    + (0.10 * max(0.0, l_margin))
                    + (1.25 * max(0.0, lexical_margin))
                )
                merge_candidate_rows.append(
                    {
                        "id_i": int(id_i),
                        "id_j": int(id_j),
                        "label_i": m_i.get('label', 'N/A'),
                        "label_j": m_j.get('label', 'N/A'),
                        "sim": float(sim_val),
                        "diff_L": float(diff_L),
                        "diff_C": float(diff_C),
                        "sim_margin": sim_margin,
                        "l_margin": l_margin,
                        "c_margin": c_margin,
                        "partner_share_i": partner_share_i,
                        "partner_share_j": partner_share_j,
                        "mutual_partner_share": mutual_partner_share,
                        "partner_support_share": partner_support_share,
                        "partner_support_threshold": partner_support_threshold,
                        "support_mode": support_mode,
                        "partner_margin": partner_margin,
                        "reciprocal_partner": reciprocal_partner,
                        "min_pair_size": min_pair_size,
                        "size_margin": size_margin,
                        "fragment_pair": fragment_pair,
                        "pair_sim_threshold": pair_sim_threshold,
                        "pair_c_epsilon": pair_c_epsilon,
                        "label_overlap": float(label_overlap),
                        "lexical_margin": float(lexical_margin),
                        "pass_count": pass_count,
                        "passes": merge_passes,
                        "score": merge_score,
                    }
                )
        passing_merge_rows = [r for r in merge_candidate_rows if r["passes"]]
        passing_merge_rows.sort(
            key=lambda r: (
                float(r["score"]),
                float(r["sim"]),
                float(r["c_margin"]),
                float(r["l_margin"]),
            ),
            reverse=True,
        )
        merged_already = set()
        for r in passing_merge_rows:
            id_i = int(r["id_i"])
            id_j = int(r["id_j"])
            if id_i in merged_already or id_j in merged_already:
                continue
            actions['merge'].append(tuple(sorted((id_i, id_j))))
            merged_already.add(id_i)
            merged_already.add(id_j)
            logging.info(
                "  Suggest MERGE for clusters %s ('%s') and %s ('%s'): "
                "confusion=%.4f, partner_support=%.4f (%s), label_overlap=%.4f, min_size=%s, C_diff=%.4f, "
                "L_diff(optional)=%.4f, score=%.4f.",
                id_i,
                r["label_i"],
                id_j,
                r["label_j"],
                r["sim"],
                r["partner_support_share"],
                r["support_mode"],
                r["label_overlap"],
                r["min_pair_size"],
                r["diff_C"],
                r["diff_L"],
                r["score"],
            )
    
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
    if top_k > 0:
        if split_candidate_rows:
            split_candidate_rows.sort(
                key=lambda r: (
                    int(r["passes"]),
                    int(r["pass_count"]),
                    int(r["cond_count"]),
                    float(max(r["L_margin"], r["C_margin"], r["V_margin"])),
                    float(r["size_margin"]),
                ),
                reverse=True,
            )
            k = min(top_k, len(split_candidate_rows))
            logging.info("Top-%s split candidates by threshold proximity:", k)
            for r in split_candidate_rows[:k]:
                logging.info(
                    "  cid=%s label='%s' pass=%s conds=%s/3 size_margin=%.2f "
                    "L_margin=%.4f C_margin=%.4f V_margin=%.4f gap=%.4f gap_margin=%.4f "
                    "runner_up=%.4f structural=%s score=%.4f cooldown=%s",
                    r["cid"],
                    r["label"],
                    r["passes"],
                    r["cond_count"],
                    r["size_margin"],
                    r["L_margin"],
                    r["C_margin"],
                    r["V_margin"],
                    r["gap_median"],
                    r["gap_margin"],
                    r["runner_up_share"],
                    r["runner_up_viable"],
                    r["score"],
                    r["in_cooldown"],
                )
        if remove_candidate_rows:
            remove_candidate_rows.sort(
                key=lambda r: (
                    int(r["passes"]),
                    int(r["pass_count"]),
                    float(min(r["size_margin"], r["ll_margin"])),
                ),
                reverse=True,
            )
            k = min(top_k, len(remove_candidate_rows))
            logging.info("Top-%s remove candidates by threshold proximity:", k)
            for r in remove_candidate_rows[:k]:
                logging.info(
                    "  cid=%s label='%s' pass=%s pass_count=%s/2 size_margin=%.2f ll_margin=%.4f",
                    r["cid"],
                    r["label"],
                    r["passes"],
                    r["pass_count"],
                    r["size_margin"],
                    r["ll_margin"],
                )
        if merge_candidate_rows:
            merge_candidate_rows.sort(
                key=lambda r: (
                    int(r["passes"]),
                    int(r["pass_count"]),
                    float(r["sim_margin"]),
                    float(r["partner_margin"]),
                    float(r["size_margin"]),
                    float(r["c_margin"]),
                ),
                reverse=True,
            )
            k = min(top_k, len(merge_candidate_rows))
            logging.info("Top-%s merge candidates by threshold proximity:", k)
            for r in merge_candidate_rows[:k]:
                logging.info(
                    "  pair=(%s,%s) labels=('%s','%s') pass=%s pass_count=%s/5 "
                    "sim=%.4f thr=%.4f (margin %.4f) reciprocal=%s support=%.4f/%0.4f (%s, margin %.4f) "
                    "label_overlap=%.4f (margin %.4f) "
                    "min_size=%s (margin %.2f) C_diff=%.4f eps=%.4f (margin %.4f) "
                    "L_diff(optional)=%.4f (margin %.4f) score=%.4f",
                    r["id_i"],
                    r["id_j"],
                    r["label_i"],
                    r["label_j"],
                    r["passes"],
                    r["pass_count"],
                    r["sim"],
                    r["pair_sim_threshold"],
                    r["sim_margin"],
                    r["reciprocal_partner"],
                    r["partner_support_share"],
                    r["partner_support_threshold"],
                    r["support_mode"],
                    r["partner_margin"],
                    r["label_overlap"],
                    r["lexical_margin"],
                    r["min_pair_size"],
                    r["size_margin"],
                    r["diff_C"],
                    r["pair_c_epsilon"],
                    r["c_margin"],
                    r["diff_L"],
                    r["l_margin"],
                    r["score"],
                )
    if verbose:
        if remove_margins:
            logging.info(f"Remove margin (L_z - remove_thresh) min/median/max: "
                         f"{np.min(remove_margins):.4f}/{np.median(remove_margins):.4f}/{np.max(remove_margins):.4f}")
        if split_l_margins:
            logging.info(
                "Split L margin (L_z - L_q_thresh) min/median/max: %.4f/%.4f/%.4f",
                float(np.min(split_l_margins)),
                float(np.median(split_l_margins)),
                float(np.max(split_l_margins)),
            )
        if split_c_margins:
            logging.info(
                "Split C margin (C_z - C_q_thresh) min/median/max: %.4f/%.4f/%.4f",
                float(np.min(split_c_margins)),
                float(np.median(split_c_margins)),
                float(np.max(split_c_margins)),
            )
        if split_v_margins:
            logging.info(
                "Split V margin (V_q_thresh - V_z) min/median/max: %.4f/%.4f/%.4f",
                float(np.min(split_v_margins)),
                float(np.median(split_v_margins)),
                float(np.max(split_v_margins)),
            )
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
    if split_enabled and split_candidate_rows:
        passing_split_rows = [r for r in split_candidate_rows if r["passes"]]
        passing_split_rows.sort(
            key=lambda r: (
                float(r["score"]),
                int(r["runner_up_viable"]),
                float(r["runner_up_balance_score"]),
                int(r["cond_count"]),
                float(r["size_margin"]),
            ),
            reverse=True,
        )
        actions['split'] = [int(r["cid"]) for r in passing_split_rows]
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
    label_prompt_style: str,
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
            label_prompt_style=label_prompt_style,
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
            label_prompt_style=label_prompt_style,
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
    label_prompt_style: str = "generic",
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

        system_prompt, task_intro, guidance_lines = build_label_prompt_components(
            label_prompt_style=label_prompt_style,
            parent_label=parent_label,
            dynamic_banned_keys=dynamic_banned_keys,
            has_contrast_examples=bool(contrast_examples),
        )

        prompt = (
            f"{task_intro}\n\n"
            "Output schema (exact keys):\n"
            '{"name":"<2-5 words>","description":"<one sentence>","keywords":["k1","k2"]}\n\n'
            "Rules:\n- " + "\n- ".join(guidance_lines) + "\n\n"
            "Target cluster examples:\n- " + "\n- ".join(examples)
        )
        if contrast_examples:
            prompt += "\n\nContrast cluster examples:\n- " + "\n- ".join(contrast_examples)

        prompt_dict = [
            {"role": "system", "content": system_prompt},
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


def _log_top_add_candidates(
    *,
    all_indices: list[int],
    pmax_vals: list[float],
    entropy_vals: list[float],
    assigned_ranks: list[int | None],
    dynamic_pmax_threshold: float,
    entropy_threshold: float,
    top_k: int,
    log_prefix: str = "",
) -> None:
    if top_k <= 0 or not all_indices:
        return

    rows = []
    for pos, row_idx in enumerate(all_indices):
        pmax = float(pmax_vals[pos])
        ent = float(entropy_vals[pos])
        rank = assigned_ranks[pos]
        cond_pmax = pmax <= dynamic_pmax_threshold
        cond_entropy = ent >= entropy_threshold
        cond_rank = (rank is None) or (rank > 1)
        pass_count = int(cond_pmax) + int(cond_entropy) + int(cond_rank)
        rows.append({
            "row_idx": int(row_idx),
            "pmax": pmax,
            "entropy": ent,
            "rank": rank,
            "pmax_margin": float(dynamic_pmax_threshold - pmax),
            "entropy_margin": float(ent - entropy_threshold),
            "pass_count": pass_count,
            "passes": pass_count == 3,
        })

    rows.sort(
        key=lambda r: (
            int(r["passes"]),
            int(r["pass_count"]),
            float(r["pmax_margin"]),
            float(r["entropy_margin"]),
        ),
        reverse=True,
    )
    k = min(int(top_k), len(rows))
    logging.info(
        "%sTop-%s add candidates (proximity to add filters pmax<=%.3f, entropy>=%.3f):",
        f"{log_prefix} " if log_prefix else "",
        k,
        dynamic_pmax_threshold,
        entropy_threshold,
    )
    for r in rows[:k]:
        logging.info(
            "  row=%s pass=%s pass_count=%s/3 pmax=%.4f (margin %.4f) entropy=%.4f (margin %.4f) rank=%s",
            r["row_idx"],
            r["passes"],
            r["pass_count"],
            r["pmax"],
            r["pmax_margin"],
            r["entropy"],
            r["entropy_margin"],
            r["rank"],
        )


def _iter_add_partition_assignments(
    poor_pzx: np.ndarray,
    num_new_clusters_to_add: int,
):
    if poor_pzx is None or len(poor_pzx) == 0 or num_new_clusters_to_add <= 0:
        return

    k_try_values: list[int] = []
    if int(num_new_clusters_to_add) <= 1:
        k_try_values.append(1)
    else:
        for raw_k in (int(num_new_clusters_to_add), 2, 3):
            k_try = min(int(raw_k), int(num_new_clusters_to_add), int(len(poor_pzx)))
            if k_try >= 2 and k_try not in k_try_values:
                k_try_values.append(k_try)

    for k_try in k_try_values:
        if k_try == 1:
            yield 1, np.zeros(len(poor_pzx), dtype=int)
            continue
        from sklearn.cluster import KMeans
        kmeans_poor = KMeans(n_clusters=k_try, n_init='auto', random_state=42)
        yield k_try, kmeans_poor.fit_predict(poor_pzx)


def _execute_add_step(
    *,
    working_df: pd.DataFrame,
    prob_calibrator: ProbabilityCalibrator,
    schema_state: SchemaState,
    choices_list: list[str],
    config: SchemaRefinementConfig,
    text_column_name: str,
    model_type: str,
    model_identifier: str,
    iter_number: int,
    next_new_cluster_id: int,
    add_cooldown_until: int,
    schema_unstable: bool,
    log_prefix: str = "",
    label_context_prefix: str = "",
) -> tuple[int, int, list[dict[str, int]]]:
    prefix = f"{log_prefix} " if log_prefix else ""
    added_clusters_info: list[dict[str, int]] = []
    add_prev_cluster_ids = None

    logging.info("%sIdentifying poorly explained texts for potential new clusters.", prefix)

    if not config.operation_enabled("add"):
        logging.info("%sSkipping add-step because add operation is disabled by tune configuration.", prefix)
        return next_new_cluster_id, add_cooldown_until, added_clusters_info

    if iter_number <= add_cooldown_until:
        logging.info(
            "%sSkipping add-step due to cooldown (last add in iter %s).",
            prefix,
            add_cooldown_until - 1,
        )
        return next_new_cluster_id, add_cooldown_until, added_clusters_info

    if schema_unstable:
        logging.info(
            "%sProceeding with add-step even after structural changes (split/merge/remove/duplicate guardrail).",
            prefix,
        )

    poor_indices: list[int] = []
    pmax_vals: list[float] = []
    entropy_vals: list[float] = []
    assigned_ranks: list[int | None] = []
    all_indices: list[int] = []

    for original_idx in working_df.index:
        sentence_text = working_df.loc[original_idx, text_column_name]
        probabilities = prob_calibrator.calibrate_p_z_given_X(sentence_text)
        pmax = float(np.max(probabilities)) if hasattr(probabilities, "__len__") and len(probabilities) > 0 else 0.0
        pmax_vals.append(pmax)
        all_indices.append(int(original_idx))

        diag = compute_posterior_diagnostics(
            pzx=np.asarray(probabilities, dtype=float),
            choices_list=choices_list,
            p_z_prior=(
                schema_state.p_z_prior
                if schema_state.p_z_prior is not None
                else np.ones(len(choices_list)) / max(1, len(choices_list))
            ),
            assigned_label=working_df.loc[original_idx, "agglomerative_label"],
        )
        entropy_vals.append(diag["entropy"])
        assigned_ranks.append(diag["assigned_rank"])

    if pmax_vals:
        logging.info(
            "%spmax stats vs add-threshold %s: min/median/max=%.4f/%.4f/%.4f",
            prefix,
            config.add_low_confidence_max,
            float(np.min(pmax_vals)),
            float(np.median(pmax_vals)),
            float(np.max(pmax_vals)),
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
    _log_top_add_candidates(
        all_indices=all_indices,
        pmax_vals=pmax_vals,
        entropy_vals=entropy_vals,
        assigned_ranks=assigned_ranks,
        dynamic_pmax_threshold=dynamic_pmax_threshold,
        entropy_threshold=float(config.add_entropy_min),
        top_k=int(config.proposal_top_k_log),
        log_prefix=log_prefix,
    )
    logging.info(
        "%sAdd-step candidates: bottom-K=%s, filtered_poor=%s (pmax<=dynamic %.3f from max(fixed %.3f, quantile q=%.3f), entropy>=%.3f, rank>1)",
        prefix,
        max(config.add_min_poorly_explained, int(0.05 * len(working_df))),
        len(poor_indices),
        dynamic_pmax_threshold,
        config.add_low_confidence_max,
        config.add_low_confidence_quantile,
        config.add_entropy_min,
    )

    if len(choices_list) >= config.add_max_total_clusters:
        logging.info("%sMax total clusters reached; skipping add-step.", prefix)
        poor_indices = []

    if len(poor_indices) < config.add_min_poorly_explained:
        logging.info(
            "%sFound %s poorly explained texts (threshold: %s). Not adding new clusters based on this criterion.",
            prefix,
            len(poor_indices),
            config.add_min_poorly_explained,
        )
        return next_new_cluster_id, add_cooldown_until, added_clusters_info

    logging.info(
        "%sFound %s texts poorly explained. Attempting to form new clusters (LLM-only).",
        prefix,
        len(poor_indices),
    )
    valid_poor_indices = list(poor_indices)
    _log_text_preview(
        header=f"{prefix}Add-step poorly explained text preview",
        texts=working_df.loc[valid_poor_indices, text_column_name].astype(str).tolist(),
        max_items=min(8, len(valid_poor_indices)),
    )

    poor_pzx = (
        np.vstack(
            [
                prob_calibrator.calibrate_p_z_given_X(working_df.loc[idx, text_column_name])
                for idx in valid_poor_indices
            ]
        )
        if valid_poor_indices
        else np.empty((0, len(choices_list)))
    )

    if len(valid_poor_indices) < config.add_min_group_size:
        logging.warning("%sNot enough poorly explained texts for add step.", prefix)
        return next_new_cluster_id, add_cooldown_until, added_clusters_info

    prev_cluster_ids = working_df.loc[valid_poor_indices, "cluster_id"].copy()
    add_prev_cluster_ids = prev_cluster_ids
    num_new_clusters_to_add = max(1, len(valid_poor_indices) // config.add_items_per_new_cluster)
    num_new_clusters_to_add = min(num_new_clusters_to_add, config.add_max_new_clusters_per_iter)
    logging.info(
        "%sAttempting to create %s new clusters from posterior vectors.",
        prefix,
        num_new_clusters_to_add,
    )
    if len(poor_pzx) < num_new_clusters_to_add:
        logging.warning(
            "%sNumber of poorly explained texts (%s) < target new clusters (%s). Adjusting.",
            prefix,
            len(poor_pzx),
            num_new_clusters_to_add,
        )
        num_new_clusters_to_add = len(poor_pzx)

    try:
        created = 0
        for k_try, new_labels_for_poor_texts in _iter_add_partition_assignments(
            poor_pzx,
            num_new_clusters_to_add,
        ):
            for k_new_cluster in range(k_try):
                group_positions = [i for i, lab in enumerate(new_labels_for_poor_texts) if lab == k_new_cluster]
                new_cluster_original_indices = [valid_poor_indices[i] for i in group_positions]
                if len(new_cluster_original_indices) < config.add_min_group_size:
                    continue
                group_pzx = poor_pzx[group_positions]
                cohesion = kl_divergence_cohesion(group_pzx)
                if cohesion < config.add_cohesion_min:
                    continue

                assigned_new_id = next_new_cluster_id
                next_new_cluster_id += 1
                working_df.loc[new_cluster_original_indices, "cluster_id"] = assigned_new_id
                texts = working_df.loc[new_cluster_original_indices, text_column_name].astype(str).tolist()

                new_label = None
                if model_type == "vllm":
                    llm_label = _label_cluster_vllm(
                        texts,
                        model_identifier,
                        tokenizer=getattr(prob_calibrator, "vllm_tokenizer", None),
                        model=getattr(prob_calibrator, "vllm_model", None),
                        label_prompt_style=config.label_prompt_style,
                        context=f"{label_context_prefix}add_cluster:{assigned_new_id}",
                        max_attempts=config.llm_label_max_attempts,
                    )
                    if llm_label:
                        name, _, keywords = llm_label
                        new_label = build_label_string(name, keywords)
                if not new_label:
                    logging.info(
                        "%sAdd-step relabel fallback for new cluster %s: using heuristic label generation.",
                        prefix,
                        assigned_new_id,
                    )
                    name, keywords = propose_label_and_keywords_from_texts(texts, parent_label=None)
                    new_label = build_label_string(name, keywords)

                working_df.loc[new_cluster_original_indices, "agglomerative_label"] = new_label
                added_clusters_info.append({"id": int(assigned_new_id), "size": int(len(new_cluster_original_indices))})
                logging.info(
                    "%sAdded new cluster %s with %s texts.",
                    prefix,
                    assigned_new_id,
                    len(new_cluster_original_indices),
                )
                _log_text_preview(
                    header=f"{prefix}Add-step cluster {assigned_new_id} text preview",
                    texts=texts,
                    max_items=min(8, len(texts)),
                )
                created += 1
                if created >= config.add_max_new_clusters_per_iter:
                    break
            if created > 0:
                break

        if created == 0:
            logging.info("%sPoorly explained set did not produce a cohesive add candidate; skipping add-step.", prefix)
            working_df.loc[valid_poor_indices, "cluster_id"] = prev_cluster_ids.values
            added_clusters_info = []
    except Exception as e:
        logging.error("%sKMeans failed for adding new clusters from poorly explained texts: %s", prefix, e)

    if added_clusters_info:
        labels_now = working_df.groupby("cluster_id")["agglomerative_label"].first().astype(str).tolist()
        dup_pairs = detect_near_duplicate_labels(labels_now, threshold=DUP_LABEL_SIM_THRESHOLD)
        if dup_pairs:
            logging.warning("%sNear-duplicate labels detected after add-step; reverting added clusters.", prefix)
            if add_prev_cluster_ids is not None:
                working_df.loc[add_prev_cluster_ids.index, "cluster_id"] = add_prev_cluster_ids.values
            added_clusters_info = []
        else:
            add_cooldown_until = iter_number + config.add_cooldown_iters

    return next_new_cluster_id, add_cooldown_until, added_clusters_info


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
    text_column_name: str,
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
            split_texts = current_df.loc[valid_subset_indices, text_column_name].astype(str).tolist()
            try:
                labels_split = split_cluster_by_posteriors(
                    pzx_subset,
                    ll_subset,
                    assigned_col_idx,
                    texts=split_texts,
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
    if split_pairs:
        expected = count_before_split + len(split_pairs)
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
    if config.label_prompt_style not in VALID_LABEL_PROMPT_STYLES:
        raise ValueError(
            f"Invalid label_prompt_style '{config.label_prompt_style}'. "
            f"Expected one of {VALID_LABEL_PROMPT_STYLES}."
        )
    if config.split_size_gate_mode not in VALID_SPLIT_SIZE_GATE_MODES:
        raise ValueError(
            f"Invalid split_size_gate_mode '{config.split_size_gate_mode}'. "
            f"Expected one of {VALID_SPLIT_SIZE_GATE_MODES}."
        )
    if config.merge_epsilon_mode not in VALID_MERGE_EPSILON_MODES:
        raise ValueError(
            f"Invalid merge_epsilon_mode '{config.merge_epsilon_mode}'. "
            f"Expected one of {VALID_MERGE_EPSILON_MODES}."
        )
    if config.merge_similarity_mode not in VALID_MERGE_SIMILARITY_MODES:
        raise ValueError(
            f"Invalid merge_similarity_mode '{config.merge_similarity_mode}'. "
            f"Expected one of {VALID_MERGE_SIMILARITY_MODES}."
        )
    logging.info(
        "Operation config: tune_operation=%s | split=%s merge=%s remove=%s add=%s revise=%s | "
        "best_operation_per_iteration=%s max_same_operation_streak=%s | structural_label_mode=%s "
        "label_prompt_style=%s "
        "split_size_gate_mode=%s merge_epsilon_mode=%s merge_similarity_mode=%s "
        "proportional_thresholds=%s adaptive_proposals=%s proposal_top_k_log=%s "
        "split_label_pair_attempts=%s llm_label_max_attempts=%s",
        config.tune_operation,
        config.operation_enabled("split"),
        config.operation_enabled("merge"),
        config.operation_enabled("remove"),
        config.operation_enabled("add"),
        config.operation_enabled("revise"),
        config.best_operation_per_iteration,
        config.max_same_operation_streak,
        config.structural_label_mode,
        config.label_prompt_style,
        config.split_size_gate_mode,
        config.merge_epsilon_mode,
        config.merge_similarity_mode,
        config.proposal_proportional_thresholds,
        config.proposal_adaptive_thresholds,
        config.proposal_top_k_log,
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
                text_column_name,
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
                text_column_name,
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
                            label_prompt_style=config.label_prompt_style,
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
                            label_prompt_style=config.label_prompt_style,
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
                            label_prompt_style=config.label_prompt_style,
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

            schema_unstable = bool(
                cand_iteration_actions.get("split")
                or cand_iteration_actions.get("merge")
                or cand_iteration_actions.get("remove")
                or duplicates_prevented
            )
            if config.operation_enabled("add") and candidate_op_name == "add":
                cand_next_new_cluster_id, cand_add_cooldown_until, added_clusters_info = _execute_add_step(
                    working_df=cand_df,
                    prob_calibrator=cand_prob_calibrator,
                    schema_state=cand_schema_state,
                    choices_list=cand_choices,
                    config=config,
                    text_column_name=text_column_name,
                    model_type=model_type,
                    model_identifier=model_identifier,
                    iter_number=iter_number,
                    next_new_cluster_id=cand_next_new_cluster_id,
                    add_cooldown_until=cand_add_cooldown_until,
                    schema_unstable=schema_unstable,
                    log_prefix=f"[Iter {iter_number}] [Candidate {candidate_op_name}]",
                    label_context_prefix=f"candidate:{candidate_op_name}:",
                )
            else:
                added_clusters_info = []
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

        proposal_relax_level = 0
        if config.proposal_adaptive_thresholds:
            proposal_relax_level = min(
                max(0, int(no_op_iters)),
                max(0, int(config.adaptive_threshold_max_relax_iters)),
            )
            if proposal_relax_level > 0:
                logging.info(
                    "[Iter %s] Applying adaptive proposal relax_level=%s after %s consecutive no-op iterations.",
                    iter_idx + 1,
                    proposal_relax_level,
                    no_op_iters,
                )

        actions = decide_schema_updates(
            cluster_metrics,
            config,
            split_cooldown,
            revise_cooldown,
            iter_idx + 1,
            independent_operation_proposals=bool(
                config.best_operation_per_iteration and config.tune_operation == "all"
            ),
            proposal_relax_level=proposal_relax_level,
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
            text_column_name,
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
                        label_prompt_style=config.label_prompt_style,
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
                        label_prompt_style=config.label_prompt_style,
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
                        label_prompt_style=config.label_prompt_style,
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

        schema_unstable = bool(actions.get("split") or actions.get("merge") or actions.get("remove") or duplicates_prevented)
        next_new_cluster_id, add_cooldown_until, added_clusters_info = _execute_add_step(
            working_df=current_merged_df,
            prob_calibrator=current_prob_calibrator,
            schema_state=schema_state,
            choices_list=current_choices_list,
            config=config,
            text_column_name=text_column_name,
            model_type=model_type,
            model_identifier=model_identifier,
            iter_number=iter_idx + 1,
            next_new_cluster_id=next_new_cluster_id,
            add_cooldown_until=add_cooldown_until,
            schema_unstable=schema_unstable,
        )
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
    resolved_label_prompt_style = (
        infer_label_prompt_style(experiment_base_dir.name)
        if args.label_prompt_style == "auto"
        else args.label_prompt_style
    )
    logging.info(
        "Resolved label_prompt_style=%s from requested=%s (experiment=%s)",
        resolved_label_prompt_style,
        args.label_prompt_style,
        experiment_base_dir.name,
    )
    models_dir = experiment_base_dir / "models"
    if args.agglomerative_output_dir:
        agglomerative_output_dir = Path(args.agglomerative_output_dir)
        if not agglomerative_output_dir.is_absolute():
            agglomerative_output_dir = experiment_base_dir / agglomerative_output_dir
        if not agglomerative_output_dir.exists():
            raise FileNotFoundError(f"Specified agglomerative output dir not found: {agglomerative_output_dir}")
        logging.info("Using user-specified agglomerative output dir: %s", agglomerative_output_dir)
    else:
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
            split_size_gate_mode=args.split_size_gate_mode,
            split_min_cluster_size_mean_ratio=args.split_min_cluster_size_mean_ratio,
            split_l_bottom_quantile=args.split_l_bottom_quantile,
            split_c_bottom_quantile=args.split_c_bottom_quantile,
            split_v_top_quantile=args.split_v_top_quantile,
            split_ll_margin=args.split_ll_margin,
            split_confidence_max=args.split_confidence_max,
            split_gap_median_min=args.split_gap_median_min,
            split_min_conditions=args.split_min_conditions,
            split_cooldown_iters=args.split_cooldown_iters,
            split_gap_top2_max=args.split_gap_top2_max,
            split_runner_up_min_share=args.split_runner_up_min_share,
            merge_enabled=args.merge_enabled,
            merge_max_per_iter=args.merge_max_per_iter,
            merge_similarity_min=args.merge_similarity_min,
            merge_similarity_mode=args.merge_similarity_mode,
            merge_similarity_quantile=args.merge_similarity_quantile,
            merge_epsilon_mode=args.merge_epsilon_mode,
            merge_l_abs_diff_quantile=args.merge_l_abs_diff_quantile,
            merge_c_abs_diff_quantile=args.merge_c_abs_diff_quantile,
            merge_l_abs_diff_max=args.merge_l_abs_diff_max,
            merge_c_abs_diff_max=args.merge_c_abs_diff_max,
            merge_l_diff_ratio_max=args.merge_l_diff_ratio_max,
            merge_c_diff_ratio_max=args.merge_c_diff_ratio_max,
            merge_min_conditions=args.merge_min_conditions,
            remove_enabled=args.remove_enabled,
            remove_max_per_iter=args.remove_max_per_iter,
            remove_min_cluster_size=args.remove_min_cluster_size,
            remove_min_cluster_size_mean_ratio=args.remove_min_cluster_size_mean_ratio,
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
            label_prompt_style=resolved_label_prompt_style,
            split_label_pair_attempts=args.split_label_pair_attempts,
            llm_label_max_attempts=args.llm_label_max_attempts,
            proposal_proportional_thresholds=args.proposal_proportional_thresholds,
            proposal_adaptive_thresholds=args.proposal_adaptive_thresholds,
            adaptive_threshold_max_relax_iters=args.adaptive_threshold_max_relax_iters,
            adaptive_quantile_step=args.adaptive_quantile_step,
            adaptive_split_size_relax_ratio=args.adaptive_split_size_relax_ratio,
            adaptive_merge_similarity_relax=args.adaptive_merge_similarity_relax,
            adaptive_merge_epsilon_expand=args.adaptive_merge_epsilon_expand,
            adaptive_remove_size_relax_ratio=args.adaptive_remove_size_relax_ratio,
            adaptive_remove_ll_factor_step=args.adaptive_remove_ll_factor_step,
            proposal_top_k_log=args.proposal_top_k_log,
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
    parser.add_argument(
        "--agglomerative_output_dir",
        type=str,
        default=None,
        help=(
            "Override hierarchy directory containing inner_node_labels.csv and clusters_level_*_clusters.csv. "
            "If relative, it is resolved under --experiment_dir. "
            "Default auto-discovers hierarchy_results, then hierarchy_results__standard, then hierarchy_results__*."
        ),
    )
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
        "--label_prompt_style",
        type=str,
        choices=list(VALID_LABEL_PROMPT_STYLE_ARGS),
        default="auto",
        help=(
            "Prompt style for LLM relabeling. 'auto' infers from the experiment name "
            "(for example newsgroups -> topic, editorial -> discourse, wiki -> biography)."
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
    parser.add_argument(
        "--split_size_gate_mode",
        type=str,
        choices=list(VALID_SPLIT_SIZE_GATE_MODES),
        default="fixed",
        help="Split size gate mode: fixed, mean_ratio, or off.",
    )
    parser.add_argument(
        "--split_min_cluster_size_mean_ratio",
        type=float,
        default=0.50,
        help="When split_size_gate_mode=mean_ratio, effective split min size is ceil(mean_cluster_size * this_ratio).",
    )
    parser.add_argument("--split_l_bottom_quantile", type=float, default=0.25, help="Bottom quantile threshold for L_z split trigger.")
    parser.add_argument("--split_c_bottom_quantile", type=float, default=0.25, help="Bottom quantile threshold for C_z split trigger.")
    parser.add_argument("--split_v_top_quantile", type=float, default=0.75, help="Top quantile threshold for V_z split trigger.")
    parser.add_argument("--split_ll_margin", type=float, default=6.0)
    parser.add_argument("--split_confidence_max", type=float, default=0.30)
    parser.add_argument("--split_gap_median_min", type=float, default=0.10, help="Deprecated legacy split knob; retained for backward compatibility.")
    parser.add_argument("--split_gap_top2_max", type=float, default=0.25, help="Structural split gate: cluster median top1-top2 posterior gap must be <= this value unless runner-up concentration is strong.")
    parser.add_argument("--split_runner_up_min_share", type=float, default=0.30, help="Structural split gate: dominant runner-up share must lie in [p, 1-p] to count as a balanced split signal.")
    parser.add_argument("--split_min_conditions", type=int, default=1, help="Minimum number of quality signals (L/C/V) required when split is justified only by low-gap ambiguity.")
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
    parser.add_argument(
        "--merge_similarity_mode",
        type=str,
        choices=list(VALID_MERGE_SIMILARITY_MODES),
        default="fixed",
        help="Merge similarity threshold mode: fixed or quantile over pairwise confusion similarities.",
    )
    parser.add_argument(
        "--merge_similarity_quantile",
        type=float,
        default=0.75,
        help="When merge_similarity_mode=quantile, similarity threshold is this quantile of pairwise confusion similarities.",
    )
    parser.add_argument(
        "--merge_epsilon_mode",
        type=str,
        choices=list(VALID_MERGE_EPSILON_MODES),
        default="fixed",
        help="Merge epsilon mode: fixed or quantile.",
    )
    parser.add_argument(
        "--merge_l_abs_diff_quantile",
        type=float,
        default=0.25,
        help="When merge_epsilon_mode=quantile, epsilon_L is this quantile of pairwise |L_a-L_b| among eligible merge pairs.",
    )
    parser.add_argument(
        "--merge_c_abs_diff_quantile",
        type=float,
        default=0.25,
        help="When merge_epsilon_mode=quantile, epsilon_C is this quantile of pairwise |C_a-C_b| among eligible merge pairs.",
    )
    parser.add_argument("--merge_l_abs_diff_max", type=float, default=12.0, help="Absolute epsilon_L threshold for merge: |L_a-L_b| <= epsilon_L.")
    parser.add_argument("--merge_c_abs_diff_max", type=float, default=0.10, help="Absolute epsilon_C threshold for merge: |C_a-C_b| <= epsilon_C.")
    parser.add_argument("--merge_l_diff_ratio_max", type=float, default=0.10)
    parser.add_argument("--merge_c_diff_ratio_max", type=float, default=0.10)
    parser.add_argument("--merge_min_conditions", type=int, default=1, help="Deprecated for mentor merge logic; retained for backward compatibility.")

    # Remove config
    parser.add_argument(
        "--remove_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable remove operation logic.",
    )
    parser.add_argument("--remove_max_per_iter", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--remove_min_cluster_size", type=int, default=5)
    parser.add_argument(
        "--remove_min_cluster_size_mean_ratio",
        type=float,
        default=0.30,
        help="When proportional proposal thresholds are enabled, remove min size becomes ceil(mean_cluster_size * this_ratio).",
    )
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
    parser.add_argument(
        "--proposal_proportional_thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable proportional proposal thresholds (dataset-scale-aware split/merge/remove proposals).",
    )
    parser.add_argument(
        "--proposal_adaptive_thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If consecutive no-op iterations occur, progressively relax proposal thresholds (acceptance gate unchanged).",
    )
    parser.add_argument(
        "--adaptive_threshold_max_relax_iters",
        type=int,
        default=3,
        help="Maximum adaptive proposal relaxation level from consecutive no-op iterations.",
    )
    parser.add_argument(
        "--adaptive_quantile_step",
        type=float,
        default=0.08,
        help="Per-relax-level quantile shift used by adaptive proposal thresholds.",
    )
    parser.add_argument(
        "--adaptive_split_size_relax_ratio",
        type=float,
        default=0.20,
        help="Per-relax-level fractional reduction for split min-size gate.",
    )
    parser.add_argument(
        "--adaptive_merge_similarity_relax",
        type=float,
        default=0.10,
        help="Per-relax-level fractional reduction for fixed merge similarity threshold.",
    )
    parser.add_argument(
        "--adaptive_merge_epsilon_expand",
        type=float,
        default=0.30,
        help="Per-relax-level fractional expansion for fixed merge epsilon thresholds.",
    )
    parser.add_argument(
        "--adaptive_remove_size_relax_ratio",
        type=float,
        default=0.20,
        help="Per-relax-level fractional increase for remove min-size threshold.",
    )
    parser.add_argument(
        "--adaptive_remove_ll_factor_step",
        type=float,
        default=0.01,
        help="Per-relax-level decrement for remove LL factor (moves remove threshold toward easier triggering).",
    )
    parser.add_argument(
        "--proposal_top_k_log",
        type=int,
        default=0,
        help="If >0, log top-K candidates per operation with threshold margins/proximity diagnostics.",
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
