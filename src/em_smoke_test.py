import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from em_scores import compute_document_level_scores, compute_corpus_level_scores
from run_em_algorithm import determine_target_clustering_info, resolve_agglomerative_output_dir
from utils_probability_calibrator import initialize_probability_calibrator


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def reservoir_sample_csv(
    path: Path,
    sample_size: int,
    seed: int,
    usecols: Optional[list[str]] = None,
    chunksize: int = 10000,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reservoir = []
    seen = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        for _, row in chunk.iterrows():
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = rng.integers(0, seen)
                if j < sample_size:
                    reservoir[j] = row
    if not reservoir:
        return pd.DataFrame(columns=usecols or [])
    return pd.DataFrame(reservoir)


def load_and_prepare_sample(
    sentence_data_path: Path,
    selected_level_assignments_path: Path,
    agglomerative_labels_path: Path,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    sample_df = reservoir_sample_csv(sentence_data_path, sample_size, seed)
    if sample_df.empty:
        raise ValueError(f"No data loaded from {sentence_data_path}")

    if "cluster" in sample_df.columns:
        sample_df = sample_df.rename(columns={"cluster": "centroid_id"})
    elif "centroid_id" not in sample_df.columns:
        raise KeyError(f"Expected 'cluster' or 'centroid_id' column in {sentence_data_path}")

    agg_labels_df = pd.read_csv(agglomerative_labels_path)  # columns: node_id, label, description
    level_assignments_df = pd.read_csv(selected_level_assignments_path)  # centroid_id, graph_node_id

    if "centroid_id" not in level_assignments_df.columns or "graph_node_id" not in level_assignments_df.columns:
        raise KeyError(f"File {selected_level_assignments_path} must contain 'centroid_id' and 'graph_node_id'.")

    merged_df = pd.merge(sample_df, level_assignments_df, on="centroid_id", how="inner")
    if merged_df.empty:
        raise ValueError("No data after merging sample with cluster assignments.")

    merged_df = merged_df.rename(columns={"graph_node_id": "cluster_id"})
    merged_df["cluster_id"] = merged_df["cluster_id"].astype(int)

    if "node_id" not in agg_labels_df.columns or "label" not in agg_labels_df.columns:
        raise KeyError(f"Agglomerative labels file {agglomerative_labels_path} must contain 'node_id' and 'label'.")

    unique_ids = sorted(merged_df["cluster_id"].unique())
    level_textual_labels_df = agg_labels_df.loc[agg_labels_df["node_id"].isin(unique_ids)].copy()
    level_textual_labels_df.sort_values(by="node_id", inplace=True)
    id_to_label = pd.Series(level_textual_labels_df.label.values, index=level_textual_labels_df.node_id).to_dict()
    merged_df["agglomerative_label"] = merged_df["cluster_id"].map(id_to_label)

    choices_list = []
    for cid in unique_ids:
        label = id_to_label.get(cid)
        if label is None:
            choices_list.append(str(cid))
            merged_df.loc[merged_df["cluster_id"] == cid, "agglomerative_label"] = str(cid)
        else:
            choices_list.append(str(label))

    return merged_df, choices_list


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="EM scoring smoke test on a sampled subset.")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sentence_column_name", type=str, default="sentences")
    parser.add_argument("--model_type", type=str, default="hf", choices=["hf", "together", "vllm"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--scorer_type", type=str, default="batch", choices=["single", "batch"])
    parser.add_argument("--agglomerative_level", type=int, default=None)
    parser.add_argument("--num_agglomerative_clusters", type=int, default=None)
    args = parser.parse_args()

    if args.agglomerative_level is None and args.num_agglomerative_clusters is None:
        raise ValueError("Provide --agglomerative_level or --num_agglomerative_clusters")

    experiment_dir = Path(args.experiment_dir)
    models_dir = experiment_dir / "models"
    agglomerative_output_dir = resolve_agglomerative_output_dir(experiment_dir)
    sentence_data_path = models_dir / "all_extracted_discourse_with_clusters.csv"
    agglomerative_labels_path = agglomerative_output_dir / "inner_node_labels.csv"

    _, _, selected_level_assignments_path = determine_target_clustering_info(
        agglomerative_output_dir,
        args.agglomerative_level,
        args.num_agglomerative_clusters,
    )

    merged_df, choices_list = load_and_prepare_sample(
        sentence_data_path,
        selected_level_assignments_path,
        agglomerative_labels_path,
        args.sample_size,
        args.seed,
    )

    if args.sentence_column_name not in merged_df.columns:
        raise KeyError(f"Column '{args.sentence_column_name}' not found in sample data.")

    calibrator = initialize_probability_calibrator(
        model_identifier=args.model_name,
        model_type=args.model_type,
        choices=choices_list,
        num_trials=args.num_trials,
        scorer_type=args.scorer_type,
        verbose=False,
    )

    doc_scores = compute_document_level_scores(
        df=merged_df,
        text_col=args.sentence_column_name,
        cluster_col="agglomerative_label",
        choices=choices_list,
        prob_calibrator=calibrator,
        embeddings=None,
        p_z_prior=getattr(calibrator, "p_z", None),
    )

    corpus_scores = compute_corpus_level_scores(
        log_px_given_z_hat=doc_scores["row_log_px_given_z_norm"][np.arange(len(merged_df)), doc_scores["row_z_hat"]],
        token_counts=None,
        k_complexity=len(choices_list),
        is_test_mask=None,
        q_ij=doc_scores["PZX"],
        log_px_given_z=doc_scores["row_log_px_given_z_norm"],
        log_pz=np.log(np.clip(doc_scores["pz_prior"], 1e-30, None)),
    )

    lz = doc_scores["L_z"]
    top_idx = int(np.argmax(lz))
    bottom_idx = int(np.argmin(lz))

    print("EM smoke test summary")
    print(f"- sample_size: {len(merged_df)}")
    print(f"- L_baseline: {doc_scores['L_baseline']:.4f}")
    print(f"- mean pmax: {doc_scores['row_pmax'].mean():.4f}")
    print(f"- top L(z): {choices_list[top_idx]} = {lz[top_idx]:.4f}")
    print(f"- bottom L(z): {choices_list[bottom_idx]} = {lz[bottom_idx]:.4f}")
    print(f"- AIC: {corpus_scores['AIC']}")
    print(f"- BIC: {corpus_scores['BIC']}")
    print(f"- ELBO: {corpus_scores['ELBO']}")


if __name__ == "__main__":
    main()
