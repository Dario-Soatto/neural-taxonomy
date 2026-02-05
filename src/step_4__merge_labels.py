"""
This script performs K-means clustering on input data and assigns clusters to embeddings.
It utilizes the FAISS library for efficient clustering and supports GPU acceleration.

Command line arguments:
- input_file (str): Path to the input file containing embeddings.
- output_file (str): Path to the output file where cluster assignments will be saved.
- model_name (str): Name of the SentenceTransformer model to use for computing embeddings.
- ncentroids (int): Number of centroids for K-means clustering.
- niter (int): Number of iterations for K-means clustering.
- downsample_to (int, optional): Number of samples to downsample to before clustering.
- save_path (str, optional): Path to save the trained K-means centroids.

Output:
- A file containing the cluster assignments for the input embeddings.
"""



import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import numpy as np
import faiss
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
import re
import os
import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class ProgressWriter:
    def write(self, text):
        match = re.search(r"(\d+)/(\d+)", text)
        if match:
            n, total = map(int, match.groups())
            print("custom progress", n, total)
            # custom reporting logic here

    def flush(self):
        pass

##
## kmeans clustering
##

def train_kmeans_clustering(x, ncentroids=1024, niter=50, verbose=True, downsample_to=None, save_path=None):
    """
    Perform K-means clustering on the input data.

    Args:
        x (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
        ncentroids (int): Number of centroids.
        niter (int): Number of iterations for training.
        verbose (bool): Whether to print verbose information.

    Returns:
        numpy.ndarray: Cluster labels for the input data.
    """
    d = x.shape[1]
    if downsample_to is not None:
        x = x[:downsample_to]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=torch.cuda.is_available())
    kmeans.train(x)
    if save_path is not None:
        with open(save_path, 'wb') as f:
            np.save(f, kmeans.centroids)
    return kmeans


def assign_kmeans_clusters(embs, save_path=None, kmeans=None):
    """
    Assigns K-means clusters to the given embeddings.

    Args:
        embs (numpy.ndarray): Embeddings to be clustered.
        save_path (str): Path to the saved centroids.

    Returns:
        numpy.ndarray: Cluster indices for the embeddings.
    """
    if kmeans is None:
        centroids = np.load(save_path)
        n, d = centroids.shape
        kmeans = faiss.Kmeans(d, n, verbose=True, gpu=True, niter=0, nredo=0)
        kmeans.train(embs, init_centroids=centroids) # this ensures that kmeans.index is created
        assert np.sum(kmeans.centroids - centroids) == 0, "centroids are not the same" # sanity check
    
    cluster_distances, cluster_indices = kmeans.assign(embs)
    return cluster_indices


def read_dataframe(input_data_file, experiment=None):
    if not os.path.exists(input_data_file):
        raise FileNotFoundError(f"Input data file not found: {input_data_file}")

    if any(ext in input_data_file for ext in ['.jsonl', '.json', '.json.gz']):
        df = pd.read_json(input_data_file, lines=True)
    elif '.csv' in input_data_file:
        df = pd.read_csv(input_data_file)
    else:
        raise ValueError(f"Unsupported file type: {input_data_file}")   

    if 'response' in df.columns:
        if 'label' not in df.columns:
            df['label'] = df['response'].apply(_extract_response_field, field_name='label')
        if 'description' not in df.columns:
            df['description'] = df['response'].apply(_extract_response_field, field_name='description')
    
    if experiment is not None and 'reasoning' in experiment:
        if '__' in experiment:
            subsection = experiment.split('__')[1]
            df = df.loc[lambda df: df['index'].str.split('__').str.get(0) == subsection] # this is a hack to get the correct subsection of the data
    
    return df


def _extract_response_field(response_value, field_name):
    if isinstance(response_value, dict):
        return response_value.get(field_name)
    if isinstance(response_value, str):
        try:
            parsed = json.loads(response_value)
            if isinstance(parsed, dict):
                return parsed.get(field_name)
        except Exception:
            return None
    return None


if __name__ == "__main__":
    import  argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_file', type=str, required=True)
    parser.add_argument('--input_col_name', type=str, required=True)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--trained_sbert_model_name', type=str, required=True)
    parser.add_argument('--sbert_batch_size', type=int, default=100)
    parser.add_argument('--datapoint_embeddings_path', type=str, default=None)
    parser.add_argument('--output_cluster_file', type=str, required=True)
    parser.add_argument('--output_data_file', type=str, required=True)
    parser.add_argument('--umap_output_file', type=str, default=None)
    parser.add_argument('--n_rows_to_process', type=int, default=None)
    # parameters for kmeans clustering
    parser.add_argument('--ncentroids', type=int, default=1024)
    parser.add_argument('--niter', type=int, default=200)
    parser.add_argument('--kmeans_downsample_to', type=int, default=None, help='Number of samples to downsample to before K-means training')
    # umap parameters
    parser.add_argument('--umap_n_components', type=int, default=25)
    parser.add_argument('--umap_n_neighbors', type=int, default=100)
    parser.add_argument('--umap_min_dist', type=float, default=0.3)
    parser.add_argument('--skip_umap', action='store_true', help='Skip UMAP dimensionality reduction')
    parser.add_argument('--skip_kmeans', action='store_true', help='Skip K-means clustering')
    args = parser.parse_args()

    df = read_dataframe(args.input_data_file, args.experiment)
    df = df.dropna(subset=[args.input_col_name])
    if args.n_rows_to_process is not None:
        df = df.head(args.n_rows_to_process)

    sbert_model = SentenceTransformer(args.trained_sbert_model_name)
    embeddings = sbert_model.encode(df[args.input_col_name].tolist(), show_progress_bar=True, batch_size=args.sbert_batch_size)

    if args.datapoint_embeddings_path is not None:
        os.makedirs(os.path.dirname(args.datapoint_embeddings_path), exist_ok=True)
        logging.info(f"Saving datapoint embeddings to {args.datapoint_embeddings_path}")
        np.savez_compressed(args.datapoint_embeddings_path, embeddings=embeddings)

    if not args.skip_umap and args.umap_output_file is not None:
        import umap 
        os.makedirs(os.path.dirname(args.umap_output_file), exist_ok=True)
        logging.info("performing UMAP embedding...")
        tqdm_kwds = {"file": ProgressWriter(), "disable": False }
        umap_embeddings = umap.UMAP(
            n_components=args.umap_n_components,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric='cosine',
            tqdm_kwds=tqdm_kwds,
            verbose=True,
        ).fit_transform(embeddings)
        np.savez_compressed(args.umap_output_file, umap_embeddings=umap_embeddings)
    else:
        logging.info("Skipping UMAP embedding...")

    if not args.skip_kmeans and args.output_cluster_file is not None:
        os.makedirs(os.path.dirname(args.output_cluster_file), exist_ok=True)
        logging.info("performing K-means clustering...")
        kmeans = train_kmeans_clustering(embeddings, ncentroids=args.ncentroids, niter=args.niter, save_path=args.output_cluster_file, downsample_to=args.kmeans_downsample_to)
        clusters = assign_kmeans_clusters(embeddings, kmeans=kmeans)
        df["cluster"] = clusters
        df.to_csv(args.output_data_file, index=False)
    else:
        logging.info("Skipping K-means clustering...")

"""

python merge_labels.py \
    --input_data_file ../../data/v3_discourse_summaries/news-discourse/all_extracted_discourse.csv.gz \
    --input_col_name discourse_label \
    --trained_sbert_model_name models/mpnet-base-all-nli-triplet/trained-model \
    --output_cluster_file ../../data/v3_discourse_summaries/news-discourse/cluster_centroids.npy \
    --output_data_file ../../data/v3_discourse_summaries/news-discourse/all_extracted_discourse_with_clusters.csv


python src/merge_labels.py \
    --input_data_file experiments/editorial/editorial-discourse-initial-labeling-labeling__experiment-editorials__model_gpt-4o-mini__0_1635.json \
    --input_col_name label \
    --trained_sbert_model_name experiments/editorial/models/editorial-sentence-similarity-model/trained-model \
    --output_cluster_file experiments/editorial/models/cluster_centroids.npy \
    --output_data_file experiments/editorial/models/all_extracted_discourse_with_clusters.csv \
    --skip_umap \
    --ncentroids 512

python src/step_4__merge_labels.py \
    --input_data_file experiments/reasoning/qwq-32b/clusters/nodes_with_preliminary_clusters.csv.gz \
    --input_col_name output \
    --datapoint_embeddings_path experiments/reasoning/qwq-32b/clusters/datapoint_embeddings.npz \
    --trained_sbert_model_name experiments/reasoning/qwq-32b/models/sentence-similarity-model/trained-model \
    --output_cluster_file experiments/reasoning/qwq-32b/clusters/preliminary_clusters.npy \
    --output_data_file experiments/reasoning/qwq-32b/clusters/nodes_with_preliminary_clusters_with_clusters.csv \
    --skip_umap \
    --ncentroids 512
"""
