"""
Score baseline outputs with LLM-based perplexity.

Takes one or more baseline output directories (each containing
inner_node_labels.csv and clusters_level_1_K_clusters.csv) and scores
them using the same LLM perplexity metric used by the EM pipeline.

Loads the LLM model once and reuses it across all baselines.

Usage:
  python src/score_baseline_with_llm.py \
    --baseline_dirs experiments/newsgroups_sanity_tiny/baseline_results/lda_gibbs \
                     experiments/newsgroups_sanity_tiny/baseline_results/bertopic \
    --sentence_data_path experiments/newsgroups_sanity_tiny/models/all_extracted_discourse_with_clusters_and_text.csv \
    --sentence_column description \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_type vllm \
    --num_trials 3
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def find_clusters_file(baseline_dir: str) -> str:
    """Find the clusters_level_1_*_clusters.csv file in a baseline dir."""
    pattern = os.path.join(baseline_dir, "clusters_level_1_*_clusters.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No clusters_level_1_*_clusters.csv found in {baseline_dir}"
        )
    return matches[0]


def load_baseline_data(
    baseline_dir: str,
    sentence_data_path: str,
    sentence_column: str,
) -> tuple:
    """Load and merge baseline outputs with sentence data.

    Returns: (merged_df, choices_list, label_map)
    """
    from schema_utils import build_label_string, sanitize_text

    # Load sentence data
    logger.info("Loading sentence data from: %s", sentence_data_path)
    sentences_df = pd.read_csv(sentence_data_path)
    if "cluster" in sentences_df.columns:
        sentences_df.rename(columns={"cluster": "centroid_id"}, inplace=True)
    if "centroid_id" not in sentences_df.columns:
        raise KeyError(
            f"Expected 'cluster' or 'centroid_id' column in {sentence_data_path}"
        )

    # Load cluster assignments
    clusters_path = find_clusters_file(baseline_dir)
    logger.info("Loading cluster assignments from: %s", clusters_path)
    level_assignments_df = pd.read_csv(clusters_path)

    # Load cluster labels
    labels_path = os.path.join(baseline_dir, "inner_node_labels.csv")
    logger.info("Loading cluster labels from: %s", labels_path)
    agg_labels_df = pd.read_csv(labels_path)

    # Merge sentences with cluster assignments
    merged_df = pd.merge(
        sentences_df, level_assignments_df, on="centroid_id", how="inner"
    )
    if merged_df.empty:
        raise ValueError(
            "No data after merging sentences with cluster assignments. "
            "Check centroid_id alignment."
        )

    merged_df.rename(columns={"graph_node_id": "cluster_id"}, inplace=True)
    merged_df["cluster_id"] = merged_df["cluster_id"].astype(int)

    # Build label map and choices
    unique_cluster_ids = sorted(merged_df["cluster_id"].unique())
    id_to_label = dict(
        zip(agg_labels_df["node_id"], agg_labels_df["label"].astype(str))
    )

    # Build unique label map (same logic as EM pipeline)
    label_map = {}
    used_labels = set()
    choices_list = []

    for cid in unique_cluster_ids:
        raw = id_to_label.get(cid, str(cid))
        label = build_label_string(sanitize_text(raw))
        if not label:
            label = f"cluster_{cid}"

        unique_label = label
        if unique_label in used_labels:
            variant = 2
            while f"{label} Variant {variant}" in used_labels:
                variant += 1
            unique_label = f"{label} Variant {variant}"

        used_labels.add(unique_label)
        label_map[cid] = unique_label
        choices_list.append(unique_label)

    merged_df["agglomerative_label"] = merged_df["cluster_id"].map(label_map)

    logger.info(
        "Loaded %d rows, %d clusters, choices: %s",
        len(merged_df),
        len(choices_list),
        choices_list,
    )
    return merged_df, choices_list, label_map


def score_baseline(
    baseline_dir: str,
    sentence_data_path: str,
    sentence_column: str,
    calibrator,
    rebuild_fn,
) -> dict:
    """Score a single baseline directory with LLM perplexity.

    Returns dict with perplexity, logL, AIC, BIC, ELBO values.
    """
    from em_scores import (
        compute_document_level_scores,
        compute_corpus_level_scores_variants,
    )

    # Load data
    merged_df, choices_list, label_map = load_baseline_data(
        baseline_dir, sentence_data_path, sentence_column
    )

    # Rebuild calibrator with this baseline's labels
    logger.info("Rebuilding calibrator with %d choices...", len(choices_list))
    cal = rebuild_fn(calibrator, choices_list)

    # Estimate token counts
    texts = merged_df[sentence_column].astype(str).tolist()
    tokenizer = getattr(cal, "tokenizer", None)
    token_counts = _estimate_token_counts(texts, tokenizer)

    # Compute document-level scores
    logger.info("Computing document-level scores for %d texts...", len(texts))
    doc_scores = compute_document_level_scores(
        df=merged_df,
        text_col=sentence_column,
        cluster_col="agglomerative_label",
        choices=choices_list,
        prob_calibrator=cal,
        embeddings=None,
        p_z_prior=None,
    )

    # Compute corpus-level scores
    k = len(choices_list)
    corpus_scores = compute_corpus_level_scores_variants(
        doc_scores=doc_scores,
        token_counts=token_counts,
        k_complexity=k,
        is_test_mask=None,
    )

    # Extract key metrics
    result = {
        "n_texts": len(texts),
        "n_clusters": k,
        "choices": choices_list,
        "L_baseline": doc_scores["L_baseline"],
    }
    # Add all corpus-level metrics
    for key, val in corpus_scores.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            result[key] = float(val)
        elif val is None:
            result[key] = None

    return result


def _estimate_token_counts(texts, tokenizer=None):
    """Estimate token counts per text."""
    counts = []
    use_tok = tokenizer is not None and hasattr(tokenizer, "encode")
    for txt in texts:
        t = str(txt)
        if use_tok:
            try:
                n = len(tokenizer.encode(t, add_special_tokens=False))
            except Exception:
                n = len(t.split())
        else:
            n = len(t.split())
        counts.append(float(max(1, n)))
    return np.array(counts, dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Score baseline outputs with LLM perplexity."
    )
    parser.add_argument(
        "--baseline_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing baseline outputs (inner_node_labels.csv, etc.).",
    )
    parser.add_argument(
        "--sentence_data_path",
        type=str,
        required=True,
        help="Path to sentence data CSV.",
    )
    parser.add_argument(
        "--sentence_column",
        type=str,
        default="description",
        help="Column name for text data.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name for scoring.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vllm",
        choices=["vllm", "hf", "together"],
        help="Model backend type.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=3,
        help="Number of permutation trials for calibration.",
    )
    parser.add_argument(
        "--scorer_type",
        type=str,
        default="batch",
        choices=["single", "batch"],
        help="Scorer type.",
    )

    args = parser.parse_args()

    # Import calibrator utilities
    from utils_probability_calibrator import (
        ProbabilityCalibrator,
        initialize_probability_calibrator,
    )

    # Load model once with dummy choices (will be rebuilt per baseline)
    logger.info("Initializing LLM model (loaded once, reused for all baselines)...")
    dummy_choices = ["placeholder_A", "placeholder_B"]
    calibrator = initialize_probability_calibrator(
        model_identifier=args.model_name,
        model_type=args.model_type,
        choices=dummy_choices,
        num_trials=args.num_trials,
        scorer_type=args.scorer_type,
    )

    # Build the rebuild function
    def rebuild_fn(existing_cal, new_choices):
        return ProbabilityCalibrator(
            choices=new_choices,
            logprob_scorer=existing_cal.logprob_scorer,
            full_logprob_fn=existing_cal.full_logprob_fn,
            num_trials=existing_cal.num_trials,
            content_free_input=existing_cal.content_free_input,
            alpha=existing_cal.alpha,
            verbose=False,
            batch_prompts=existing_cal.batch_prompts,
            batch_permutations=existing_cal.batch_permutations,
            vllm_model=getattr(existing_cal, "vllm_model", None),
            vllm_tokenizer=getattr(existing_cal, "vllm_tokenizer", None),
        )

    # Score each baseline
    all_results = {}
    for baseline_dir in args.baseline_dirs:
        method_name = os.path.basename(baseline_dir.rstrip("/"))
        logger.info("\n" + "=" * 60)
        logger.info("Scoring baseline: %s", method_name)
        logger.info("=" * 60)

        try:
            result = score_baseline(
                baseline_dir=baseline_dir,
                sentence_data_path=args.sentence_data_path,
                sentence_column=args.sentence_column,
                calibrator=calibrator,
                rebuild_fn=rebuild_fn,
            )

            # Save per-baseline
            output_path = os.path.join(baseline_dir, "llm_scores.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=float)
            logger.info("Saved LLM scores to: %s", output_path)

            # Print summary
            ppl_assigned = result.get("perplexity_token_assigned", "N/A")
            ppl_oracle = result.get("perplexity_token_oracle", "N/A")
            avg_ll = result.get("avg_logL_per_token_assigned", "N/A")
            logger.info(
                "  PPL(assigned)=%.4f  PPL(oracle)=%.4f  avg_logL/tok=%.6f",
                ppl_assigned if isinstance(ppl_assigned, float) else 0,
                ppl_oracle if isinstance(ppl_oracle, float) else 0,
                avg_ll if isinstance(avg_ll, float) else 0,
            )

            all_results[method_name] = result

        except Exception as e:
            logger.error("Failed to score %s: %s", method_name, e, exc_info=True)
            all_results[method_name] = {"error": str(e)}

    # Print comparison table
    print("\n" + "=" * 80)
    print("  LLM PERPLEXITY COMPARISON")
    print("=" * 80)
    print(
        f"{'Method':<25} {'PPL(asgn)':>12} {'PPL(orac)':>12} {'avgLL/tok':>12} {'K':>4}"
    )
    print("-" * 80)
    for method, result in all_results.items():
        if "error" in result:
            print(f"{method:<25}  ERROR: {result['error']}")
            continue
        print(
            f"{method:<25} "
            f"{result.get('perplexity_token_assigned', 0):>12.4f} "
            f"{result.get('perplexity_token_oracle', 0):>12.4f} "
            f"{result.get('avg_logL_per_token_assigned', 0):>12.6f} "
            f"{result.get('n_clusters', '?'):>4}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
