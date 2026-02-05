#!/usr/bin/env python
# coding: utf-8

# ----------------------------------------------------------------
""" 
Step 2: Create Supervised Similarity Data
SYNCHRONOUS API VERSION - uses direct API calls instead of batch API.
Faster and more reliable than batch API when batches are stuck.
Uses concurrent requests for speed.

This script processes narrative function data to generate prompts for OpenAI API.

Command line arguments:
- input_file: Path to the input CSV file with the following columns:
    - 'Narrative Function': Contains the text data for which comparisons will be made. (or other column name specified with --col_name)
    - 'Is_Error' (optional): Indicates whether a row should be excluded from processing (rows with 'No' will be included).
- output_file: Path to the output file where the processed data will be saved.
- text_col_name: Name of the column containing the text data (default: 'Narrative Function').
- text_keyword_name: Keyword to use in the prompt (default: 'Description').
- embedding_model_name: Name of the SentenceTransformer model to use for computing embeddings.
- model_name: Name of the LLM to use for prompting.
- concurrency: Number of concurrent API requests (default: 50).
- checkpoint_every: Save checkpoint every N responses (default: 100).

Output:
- A CSV file containing paired labeled data.
- A JSONL file containing triplets for training.
"""

import os
import re
import glob
import json
import argparse
import pandas as pd
import numpy as np
import jsonlines
from tqdm.auto import tqdm
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
from sentence_transformers import SentenceTransformer
import sys
import jsonlines
sys.path.append('..')
import torch
import logging
import itertools
from prompts import (
    MultiSimilarityResponse, 
    EMOTIONS_SIMILARITY_PROMPT, 
    MATHEMATICAL_REASONING_SIMILARITY_PROMPT,
    NEWS_DISCOURSE_SIMILARITY_PROMPT,
    EDITORIAL_SIMILARITY_PROMPT,
    HATE_SPEECH_SIMILARITY_PROMPT,
    GENERIC_SIMILARITY_PROMPT,
)

# Set environment variable
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(f'{here}/../config.json'):
    config_data = json.load(open(f'{here}/../config.json'))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


# ============================================================================
# CHECKPOINTING FUNCTIONS (for sync mode)
# ============================================================================

def get_checkpoint_paths(output_file, temp_dir):
    """Get paths for checkpoint files."""
    base = os.path.splitext(output_file)[0]
    checkpoint_dir = temp_dir or os.path.dirname(output_file) or '.'
    return {
        'results': f"{base}_partial_results.jsonl",
        'state': os.path.join(checkpoint_dir, f"{os.path.basename(base)}_sync_state.json"),
    }


def save_all_results(checkpoint_paths, all_responses):
    """Save ALL results to disk (overwrites previous checkpoint)."""
    with open(checkpoint_paths['results'], 'w') as f:
        for resp in all_responses:
            f.write(json.dumps(resp) + '\n')


def save_checkpoint_state(checkpoint_paths, completed_ids):
    """Save the set of completed prompt IDs."""
    with open(checkpoint_paths['state'], 'w') as f:
        json.dump({'completed_ids': list(completed_ids)}, f)


def save_checkpoint(checkpoint_paths, all_responses, completed_ids):
    """Save full checkpoint: all responses + all completed IDs."""
    save_all_results(checkpoint_paths, all_responses)
    save_checkpoint_state(checkpoint_paths, completed_ids)
    logging.info(f"[Checkpoint] 💾 Saved {len(all_responses)} responses")


def load_checkpoint_state(checkpoint_paths):
    """Load existing checkpoint state or create new."""
    if os.path.exists(checkpoint_paths['state']):
        with open(checkpoint_paths['state'], 'r') as f:
            data = json.load(f)
            return set(data.get('completed_ids', []))
    return set()


def load_partial_results(checkpoint_paths):
    """Load any existing partial results."""
    results = []
    if os.path.exists(checkpoint_paths['results']):
        with open(checkpoint_paths['results'], 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        logging.info(f"[Checkpoint] 📂 Loaded {len(results)} existing responses from checkpoint")
    return results


def cleanup_checkpoint_files(checkpoint_paths):
    """Remove checkpoint files after successful completion."""
    for name, path in checkpoint_paths.items():
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"[Cleanup] Removed checkpoint file: {path}")


# ============================================================================
# SYNCHRONOUS OPENAI API (replaces batch API)
# ============================================================================

def make_single_request(client, prompt_id, prompt_text, model_name, response_format):
    """Make a single synchronous API request with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                response_format=response_format,
                temperature=0.1,
                max_tokens=1024,
            )
            
            # Extract the parsed response
            parsed = response.choices[0].message.parsed
            if parsed:
                return {
                    'custom_id': prompt_id,
                    'response': parsed.model_dump(),
                }
            else:
                # Fallback to raw content if parsing failed
                content = response.choices[0].message.content
                return {
                    'custom_id': prompt_id,
                    'response': json.loads(content) if content else None,
                }
                
        except Exception as e:
            if "rate" in str(e).lower() and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                logging.warning(f"[Sync] Failed after {max_retries} attempts for {prompt_id}: {e}")
                return {
                    'custom_id': prompt_id,
                    'response': None,
                    'error': str(e),
                }
            else:
                time.sleep(1)
    
    return {'custom_id': prompt_id, 'response': None, 'error': 'Unknown error'}


def generate_responses_sync(
    prompt_ids,
    prompts,
    model_name,
    response_format=None,
    temp_dir=None,
    output_file=None,
    concurrency=50,
    checkpoint_every=100,
):
    """
    Generate responses using synchronous OpenAI API with concurrent requests.
    Much faster than batch API when batches are stuck.
    """
    from openai import OpenAI
    
    logging.info("[Sync] Initializing OpenAI client...")
    client = OpenAI()
    
    if temp_dir is None:
        temp_dir = os.path.join(os.getcwd(), 'temp_batches')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Setup checkpointing
    checkpoint_paths = get_checkpoint_paths(output_file, temp_dir) if output_file else None
    completed_ids = set()
    all_responses = []
    
    if checkpoint_paths:
        completed_ids = load_checkpoint_state(checkpoint_paths)
        all_responses = load_partial_results(checkpoint_paths)
        if completed_ids:
            logging.info(f"[Checkpoint] ⏭️  Resuming: {len(completed_ids)} already completed")
    
    # Filter out already-completed prompts
    pending_prompts = [
        (pid, ptext) for pid, ptext in zip(prompt_ids, prompts)
        if pid not in completed_ids
    ]
    
    if not pending_prompts:
        logging.info("[Checkpoint] ✅ All prompts already completed from previous run!")
        return all_responses
    
    logging.info(f"[Sync] Processing {len(pending_prompts)} prompts with {concurrency} concurrent requests")
    
    # Process with thread pool
    new_responses = []
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(make_single_request, client, pid, ptext, model_name, response_format): pid
            for pid, ptext in pending_prompts
        }
        
        # Process as they complete with progress bar
        with tqdm(total=len(pending_prompts), desc="Processing", unit="req") as pbar:
            for future in as_completed(future_to_id):
                prompt_id = future_to_id[future]
                try:
                    result = future.result()
                    
                    if result.get('response') is not None:
                        new_responses.append(result)
                        all_responses.append(result)
                        completed_ids.add(prompt_id)
                        
                        # Checkpoint periodically
                        if checkpoint_paths and len(new_responses) % checkpoint_every == 0:
                            save_checkpoint(checkpoint_paths, all_responses, completed_ids)
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logging.warning(f"[Sync] Error processing {prompt_id}: {e}")
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({'done': len(new_responses), 'failed': failed_count})
    
    # Final checkpoint save
    if checkpoint_paths and new_responses:
        save_checkpoint(checkpoint_paths, all_responses, completed_ids)
    
    logging.info(f"[Sync] ✅ Complete! {len(new_responses)} new responses, {failed_count} failed")
    
    return all_responses


def batchify_dataframe(df, batch_size):
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def compute_embeddings(
        input_df,
        embedding_model_name, 
        text_col_name='Narrative Function', 
        err_col_name='Is_Error',
        max_returned=None
):
    """
    Compute embeddings for the specified column in the DataFrame using a SentenceTransformer model.
    
    Parameters:
    - source_df: DataFrame containing the source data.
    - embedding_model_name: Name of the SentenceTransformer model to use for computing embeddings.
    - text_col_name: Name of the column to compute embeddings for (default is 'Narrative Function').
    - err_col_name: Name of the column indicating errors (default is 'Is_Error').
    - max_returned: Maximum number of samples to return (optional).
    
    Returns:
    - embeddings: Computed embeddings for the specified column.
    - idx_of_df: Index of the DataFrame rows used for computing embeddings.
    """
    logging.info(f"Loading SentenceTransformer model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    
    if err_col_name in input_df.columns:
        logging.info(f"Filtering out rows with {err_col_name} != 'No'")
        input_df = input_df.loc[input_df[err_col_name] == 'No', text_col_name]
    if max_returned:
        logging.info(f"Sampling {max_returned} rows from input data")
        input_df = input_df.sample(n=max_returned)

    narrative_functions = input_df[text_col_name].dropna()
    if text_col_name == 'Narrative Function':
        logging.info("Processing Narrative Function column - extracting first part before ':'")
        texts = narrative_functions.str.split(':').str.get(0).tolist()
    else:
        texts = narrative_functions.tolist()
    
    logging.info(f"Computing embeddings for {len(texts)} texts")
    embeddings = model.encode(texts, show_progress_bar=True)
    idx_of_df = narrative_functions.index
    logging.info(f"Successfully computed embeddings of shape {embeddings.shape}")
    return embeddings, idx_of_df


def get_embeddings_cache_path(input_file, embedding_model_name, temp_dir=None):
    """Generate a cache path for embeddings based on input file and model."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    model_name_safe = embedding_model_name.replace('/', '_')
    cache_dir = temp_dir or os.path.dirname(input_file) or '.'
    return os.path.join(cache_dir, f"{base_name}_embeddings_{model_name_safe}.npz")


def save_embeddings(embeddings, idx_of_df, cache_path):
    """Save embeddings and indices to disk."""
    logging.info(f"[Embeddings] 💾 Saving embeddings to: {cache_path}")
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        idx_of_df=np.array(idx_of_df)
    )
    logging.info(f"[Embeddings] ✅ Saved {len(embeddings)} embeddings ({os.path.getsize(cache_path) / 1024 / 1024:.1f} MB)")


def load_embeddings(cache_path):
    """Load embeddings and indices from disk."""
    logging.info(f"[Embeddings] 📂 Loading cached embeddings from: {cache_path}")
    data = np.load(cache_path)
    embeddings = data['embeddings']
    idx_of_df = pd.Index(data['idx_of_df'])
    logging.info(f"[Embeddings] ✅ Loaded {len(embeddings)} embeddings of shape {embeddings.shape}")
    return embeddings, idx_of_df


def get_or_compute_embeddings(input_df, embedding_model_name, text_col_name, input_file, temp_dir=None):
    """
    Get embeddings from cache if available, otherwise compute and save them.
    """
    cache_path = get_embeddings_cache_path(input_file, embedding_model_name, temp_dir)
    
    if os.path.exists(cache_path):
        try:
            embeddings, idx_of_df = load_embeddings(cache_path)
            # Verify the cached embeddings match the current input
            if len(embeddings) == len(input_df):
                return embeddings, idx_of_df
            else:
                logging.warning(f"[Embeddings] ⚠️ Cache size mismatch ({len(embeddings)} vs {len(input_df)}), recomputing...")
        except Exception as e:
            logging.warning(f"[Embeddings] ⚠️ Failed to load cache: {e}, recomputing...")
    
    # Compute embeddings
    embeddings, idx_of_df = compute_embeddings(input_df, embedding_model_name, text_col_name=text_col_name)
    
    # Save to cache
    save_embeddings(embeddings, idx_of_df, cache_path)
    
    return embeddings, idx_of_df


def sample_from_batch(batch_high_sim_pairs_gpu, sample_size_per_batch):
    # Get the number of rows in the batch
    num_rows = batch_high_sim_pairs_gpu.size(0)
    sample_size_per_batch = min(sample_size_per_batch, num_rows)
    sampled_indices = torch.randperm(num_rows, device=batch_high_sim_pairs_gpu.device)[:sample_size_per_batch]
    sampled_batch = batch_high_sim_pairs_gpu[sampled_indices]
    return sampled_indices, sampled_batch


def compute_high_similarity_pairs(
        embeddings, 
        idx_of_df,
        min_sim_threshold=0.3, 
        max_sim_threshold=0.99999, 
        sample_size=2_000_000, 
        batch_size=50_000
):
    """
    Identify pairs of data points with high similarity based on cosine similarity of embeddings.
    This version processes embeddings in batches to handle large datasets efficiently using GPU.
    
    Parameters:
    - embeddings: Embeddings of the data points.
    - idx_of_df: Index of the DataFrame rows corresponding to the embeddings.
    - min_sim_threshold: Minimum threshold for considering a pair as highly similar (default is 0.3).
    - max_sim_threshold: Maximum threshold for considering a pair as highly similar (default is 0.99999).
    - sample_size: Number of high similarity pairs to sample (default is 2,000,000).
    - batch_size: Number of embeddings to process in each batch (default is 1000).
    
    Returns:
    - high_sim_sample: DataFrame containing sampled high similarity pairs.
    """
    logging.info(f"Computing high similarity pairs with thresholds [{min_sim_threshold}, {max_sim_threshold}]")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    embeddings_tensor = torch.tensor(embeddings).to(device)
    idx_of_df_tensor = torch.tensor(idx_of_df).to(device)
    norm_embeddings = embeddings_tensor / embeddings_tensor.norm(dim=1, keepdim=True)
    num_embeddings = len(embeddings)
    high_sim_pairs = []
    num_batches = int(np.ceil(num_embeddings / batch_size))
    sample_size_per_batch = int(np.ceil(sample_size / num_batches))
    
    logging.info(f"Processing {num_batches} batches of size {batch_size}")
    logging.info(f"Target sample size: {sample_size} (per batch: {sample_size_per_batch})")

    # Process embeddings in batches
    for start in tqdm(range(0, num_embeddings, batch_size), total=num_batches, desc='Processing similarity batches'):
        end = min(start + batch_size, num_embeddings)
        batch_idx = idx_of_df_tensor[start:end]
        batch_norm_embeddings = norm_embeddings[start:end]

        # Compute cosine similarity for the current batch using GPU
        similarity_matrix = torch.mm(batch_norm_embeddings, norm_embeddings.t())
        
        # Zero out lower triangle and diagonal for the current batch
        similarities = torch.triu(similarity_matrix, diagonal=1)
        high_sim_indices = torch.nonzero(
            (similarity_matrix > min_sim_threshold) & (similarity_matrix < max_sim_threshold), as_tuple=True
        )
        similarities = similarity_matrix[high_sim_indices]
        
        # No need to move to CPU until sampling
        batch_high_sim_pairs_gpu = torch.stack([
            batch_idx[high_sim_indices[0]],  # row indices from batch
            idx_of_df_tensor[high_sim_indices[1]],  # column indices from original embeddings
            similarities
        ], dim=1)

        # Sample from the batch
        num_high_sim_pairs = batch_high_sim_pairs_gpu.size(0)
        sample_size_this_batch = min(sample_size_per_batch, num_high_sim_pairs)
        if sample_size_this_batch > 0:
            sampled_indices = torch.randperm(num_high_sim_pairs, device=device)[:sample_size_this_batch]
            sampled_batch = batch_high_sim_pairs_gpu[sampled_indices]
            high_sim_pairs.append(sampled_batch.cpu())
            del batch_high_sim_pairs_gpu
            torch.cuda.empty_cache()

    all_high_sim_pairs = torch.cat(high_sim_pairs, dim=0)
    all_high_sim_pairs = all_high_sim_pairs.numpy()
    high_sim_df = pd.DataFrame(all_high_sim_pairs, columns=['level_0', 'level_1', 'similarity'])

    logging.info(f"Found {len(high_sim_df)} high similarity pairs")
    logging.info(f"Sampling {min(sample_size, len(high_sim_df))} pairs")
    high_sim_sample = high_sim_df.sample(n=min(sample_size, len(high_sim_df)))
    return high_sim_sample


def create_high_similarity_samples(text_df, high_sim_sample, text_col_name='Narrative Function'):
    """
    Create a DataFrame with pairs of texts that have high similarity.
    
    Parameters:
    - text_df: DataFrame containing the text data.
    - high_sim_sample: DataFrame containing high similarity pairs.
    - text_col_name: Name of the column containing the text data (default is 'Narrative Function').
    
    Returns:
    - high_sim_pairwise_samples_to_evaluate: DataFrame with pairs of high similarity texts.
    """
    narrative_functions = text_df[text_col_name]
    high_sim_pairwise_samples_to_evaluate = (
        pd.concat([
            narrative_functions.loc[high_sim_sample['level_0']].reset_index(drop=True).rename('description_1'),
            narrative_functions.loc[high_sim_sample['level_1']].reset_index(drop=True).rename('description_2'),
        ], axis=1)
        .dropna()
        .drop_duplicates()
    )
    return high_sim_pairwise_samples_to_evaluate


def generate_prompts(high_sim_samples, prompt_template, text_keyword='Label', k=3):
    """
    Generate prompts for the LLM based on high similarity samples.
    
    Parameters:
    - high_sim_samples: DataFrame containing high similarity text pairs.
    - prompt_template: Template for the prompt. Must contain {k_i} and {samples_str}.
    - k: Number of pairs to include in each prompt (default is 5).
    
    Returns:
    - all_prompts: List of generated prompts for the OpenAI API.
    """
    logging.info(f"Generating prompts for {len(high_sim_samples)} samples with k={k}")
    all_prompts = []
    all_batched_inputs = []
    total_batches = int(len(high_sim_samples) / k)
    
    for t in tqdm(batchify_dataframe(high_sim_samples, k), total=total_batches, desc='Generating prompts'):
        samples = []
        for i, (_, row) in enumerate(t.iterrows(), 1):
            line = f'{i}. {text_keyword} 1: {row["description_1"]}, {text_keyword} 2: {row["description_2"]}'
            samples.append(line)
        
        k_i = min(k, len(t))
        samples_str = '\n'.join(samples)
        prompt = prompt_template.format(k_i=k_i, samples_str=samples_str)
        all_prompts.append(prompt)
        all_batched_inputs.append(t)
    
    logging.info(f"Generated {len(all_prompts)} prompts")
    return all_prompts, all_batched_inputs


def create_triplets(full_data_exp_df, max_negatives_per_positive=10, max_positives=10):
    """
    Creates triplets of the form {'anchor': description_1, 'positive': description_2_positive, 'negative': description_2_negative}
    where each description_1 is paired with a description_2 with "Yes" (positive) and "No" (negative).
    
    Parameters:
        full_data_exp_df (pd.DataFrame): DataFrame containing 'description_1', 'description_2', and 'output' columns.
    
    Returns:
        triplets (list): List of dictionaries containing 'anchor', 'positive', and 'negative' keys.
    """
    # Initialize an empty list to collect triplets
    triplets = []

    # Ensure the 'output' column is in a consistent format
    full_data_exp_df['label'] = full_data_exp_df['label'].str.strip().str.lower()

    # Group the DataFrame by 'description_1'
    grouped = full_data_exp_df.groupby('description_1')

    # For each group (each 'description_1')
    for description_1_value, group in grouped:
        # Get the positive and negative examples
        positive_examples = group.loc[group['label'] == 'yes', 'description_2'].tolist()
        negative_examples = group.loc[group['label'] == 'no', 'description_2'].tolist()

        # If we have at least one positive and one negative example
        if positive_examples and negative_examples:
            # For each positive example
            # Optionally limit the number of negatives per positive
            if max_positives:
                positives_to_use = random.sample(positive_examples, min(max_positives, len(positive_examples)))
            else:
                positives_to_use = positive_examples

            if max_negatives_per_positive:
                negatives_to_use = random.sample(negative_examples, min(max_negatives_per_positive, len(negative_examples)))
            else:
                negatives_to_use = negative_examples
                
            for pos in positives_to_use:
                for neg in negatives_to_use:
                    # Create triplet
                    triplet = {
                        'anchor': description_1_value,
                        'positive': pos,
                        'negative': neg
                    }
                    # Append to list
                    triplets.append(triplet)
                    
    return triplets


def match_batched_vllm_results_to_prompts(all_results, all_batched_inputs):
    """
    Match prompts to VLLM results when each prompt is a batch of multiple queries.
    """
    all_merged = []
    logging.info(f"Number of results: {len(all_results)}")
    logging.info(f"Number of batched inputs: {len(all_batched_inputs)}")
    
    for i, (r, b) in enumerate(zip(all_results, all_batched_inputs)):
        try:
            # Parse the JSON response
            parsed_result = json.loads(r)
            
            # Extract the labels from the pairs
            if isinstance(parsed_result, dict) and 'pairs' in parsed_result:
                sorted_pairs = sorted(parsed_result['pairs'], key=lambda x: x['pair_idx'])
                labels = [pair['label'] for pair in sorted_pairs]
                if len(labels) == len(b):
                    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
                    b_copy = b.copy()
                    b_copy.loc[:, 'label'] = labels
                    all_merged.append(b_copy)
                else:
                    logging.warning(f"Batch {i}: JSON pairs length mismatch - result: {len(labels)}, input: {len(b)}")
                    logging.warning(f"First few result items: {labels[:3]}")
                    logging.warning(f"First few input items: {b.head(3) if hasattr(b, 'head') else b[:3]}")
            else:
                logging.warning(f"Batch {i}: Expected JSON with 'pairs' key but got {type(parsed_result)}")
                logging.warning(f"Result content: {r[:200]}...")
        except json.JSONDecodeError as e:
            logging.error(f"Batch {i}: Failed to parse JSON response: {str(e)}")
            logging.error(f"Raw response: {r[:200]}...")

    if not all_merged:
        logging.error("No batches were successfully matched!")
        return pd.DataFrame()  # Return empty DataFrame instead of raising error
        
    full_data_df = pd.concat(all_merged)
    return full_data_df


def match_batched_openai_results_to_prompts(all_results, all_batched_inputs):
    """
    Match prompts to OpenAI results when each prompt is a batch of multiple queries.
    """
    all_merged = []
    for result, batch in zip(all_results, all_batched_inputs):
        if result and 'response' in result and result['response'] is not None:
            response = result['response']['pairs']
            if isinstance(response, str):
                response = response.split('\n')
            if len(response) == len(batch):
                if isinstance(response[0], dict) and isinstance(response[0].get('pair_idx'), int) or response[0].get('pair_idx').isdigit():
                    response = sorted(response, key=lambda x: int(x['pair_idx']))
                    batch['label'] = list(map(lambda x: x['label'], response))
                    all_merged.append(batch)

    full_data_df = pd.concat(all_merged)
    return full_data_df


def load_and_preprocess(input_file, text_col_name, text_col_name_2, debug=False):
    if '.csv' in input_file:
        input_df = pd.read_csv(input_file)
    elif '.json' in input_file:
        input_df = pd.read_json(input_file, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {input_file}")
    
    if debug:
        logging.info("Running in debug mode - limiting to 10,000 rows")
        input_df = input_df.iloc[:10_000]

    if text_col_name == 'Narrative Function':
        logging.info("Processing Narrative Function column")
        input_df[text_col_name] = input_df[text_col_name].str.split('\n').str.get(0).str.strip()
    
    # Always create 'text_col' for downstream processing
    if text_col_name_2:
        logging.info(f"Combining columns {text_col_name} and {text_col_name_2}")
        input_df['text_col'] = '"' + input_df[text_col_name] + '": ' + input_df[text_col_name_2]
    else:
        logging.info(f"Using column {text_col_name} as text_col")
        input_df['text_col'] = input_df[text_col_name]

    return input_df


def get_prompt_template(prompt_template):
    # 
    if prompt_template == 'generic':
        return GENERIC_SIMILARITY_PROMPT

    # prompt template is a file
    if prompt_template.endswith('.txt'):
        with open(prompt_template, 'r') as f:
            return f.read()
    elif prompt_template.endswith('.json'):
        with open(prompt_template, 'r') as f:
            return json.load(f)
    
    # prompt template is an experiment name
    elif prompt_template == 'emotions':
        return EMOTIONS_SIMILARITY_PROMPT
    elif prompt_template == 'news_discourse':
        return NEWS_DISCOURSE_SIMILARITY_PROMPT
    elif prompt_template == 'editorial':
        return EDITORIAL_SIMILARITY_PROMPT
    elif prompt_template == 'hate_speech':
        return HATE_SPEECH_SIMILARITY_PROMPT
    
    ## prompt template is a string 
    else:
        return prompt_template

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process narrative function similarity using synchronous OpenAI API.')
    parser.add_argument('--input_file', type=str, default=None, help='Input data file.')
    parser.add_argument('--output_file', type=str, default=None, help='Output file filename.')
    parser.add_argument('--prompt_template', type=str, default='generic', help='Either a file, an experiment name, or a string literal')
    parser.add_argument('--embedding_model_name', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name for embeddings.')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help='OpenAI model name for prompting.')
    parser.add_argument('--text_col_name', type=str, default=None, help='Column name to compute embeddings for.')
    parser.add_argument('--text_col_name_2', type=str, default=None, help='If there is a second text column (e.g. one column for the "Label", another for the "Description").')
    parser.add_argument('--text_keyword_name', type=str, default='Label', help='Keyword to use in the prompt.')
    parser.add_argument('--min_sim_threshold', type=float, default=0.3, help='Similarity threshold for selecting pairs.')
    parser.add_argument('--max_sim_threshold', type=float, default=0.9, help='Similarity threshold for selecting pairs.')
    parser.add_argument('--sample_size', type=int, default=2_000_000, help='Number of sample pairs to process.')
    parser.add_argument('--k', type=int, default=5, help='Number of pairs per prompt.')
    parser.add_argument('--batch_size', type=int, default=40_000, help='Batch size for similarity computation.')
    parser.add_argument('--concurrency', type=int, default=50, help='Number of concurrent API requests (default: 50)')
    parser.add_argument('--checkpoint_every', type=int, default=100, help='Save checkpoint every N responses (default: 100)')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory to store temporary files.')
    args = parser.parse_args()

    logging.info(f"Starting processing with arguments: {args}")

    # Process source data
    logging.info(f"Loading input file: {args.input_file}")
    input_df = load_and_preprocess(args.input_file, args.text_col_name, args.text_col_name_2, args.debug)
    
    logging.info(f"Loaded {len(input_df)} rows from input file")

    # Step 0: create data for training a similarity model
    # ----------------------------------------------------------------
    logging.info("Getting embeddings (will load from cache if available)")
    embeddings, idx_of_df = get_or_compute_embeddings(
        input_df, 
        args.embedding_model_name, 
        text_col_name='text_col',
        input_file=args.input_file,
        temp_dir=args.temp_dir
    )
    if args.debug:
        embeddings = embeddings[:10_000]
        idx_of_df = idx_of_df[:10_000]
    
    logging.info("Computing high similarity pairs")
    high_sim_sample = compute_high_similarity_pairs(
        embeddings, 
        idx_of_df, 
        args.min_sim_threshold, 
        args.max_sim_threshold, 
        args.sample_size,
        args.batch_size
    )
    high_sim_samples = create_high_similarity_samples(input_df, high_sim_sample, text_col_name='text_col')
    logging.info(f"Created {len(high_sim_samples)} high similarity samples")

    #
    # Step 2: Generate prompts for prompting an LLM to label the data
    # ----------------------------------------------------------------
    logging.info("Generating prompts for language model supervision")
    all_prompts, all_batched_inputs = generate_prompts(
        high_sim_samples, 
        prompt_template=get_prompt_template(args.prompt_template), 
        text_keyword=args.text_keyword_name, 
        k=args.k
    )
    
    if args.debug:
        logging.info("Writing debug prompts to file")
        with open('debug_prompts.txt', 'w') as f:
            f.write('\n'.join(all_prompts))

    # Generate responses using synchronous OpenAI API
    logging.info(f"Using OpenAI model: {args.model_name} (synchronous mode, concurrency={args.concurrency})")
    results = generate_responses_sync(
        prompt_ids=[str(i) for i in range(len(all_prompts))],
        prompts=all_prompts,
        model_name=args.model_name,
        response_format=MultiSimilarityResponse,
        temp_dir=args.temp_dir,
        output_file=args.output_file,
        concurrency=args.concurrency,
        checkpoint_every=args.checkpoint_every,
    )
    
    # Match results to prompts
    logging.info("Matching sync results to prompts")
    full_data_exp_df = match_batched_openai_results_to_prompts(results, all_batched_inputs)

    if args.debug:
        logging.info("Writing debug results to file")
        with open('debug_results.txt', 'w') as f:
            f.write('\n'.join(str(r) for r in results))

    output_path, output_filename = os.path.split(args.output_file)
    output_fname, output_ext = os.path.splitext(output_filename)
    if not os.path.exists(output_path):
        logging.info(f"Creating output directory: {output_path}")
        os.makedirs(output_path)
    
    output_filename = f'paired_labeled_data_{output_fname}.csv'
    paired_labeled_data_path = os.path.join(output_path, output_filename)
    logging.info(f"Saving paired labeled data to: {paired_labeled_data_path}")
    full_data_exp_df.to_csv(paired_labeled_data_path, mode='a')

    # Step 3: Create triplets and save to file
    # ----------------------------------------------------------------
    logging.info("Creating triplets from labeled data")
    triplets = create_triplets(full_data_exp_df)
    output_filename = f'triplets_{output_fname}.jsonl'
    triplets_path = os.path.join(output_path, output_filename)
    logging.info(f"Saving {len(triplets)} triplets to: {triplets_path}")
    with jsonlines.open(triplets_path, 'a') as f:
        f.write_all(triplets)
    
    # Clean up checkpoint files on successful completion
    checkpoint_paths = get_checkpoint_paths(args.output_file, args.temp_dir)
    cleanup_checkpoint_files(checkpoint_paths)
        
    logging.info("Processing complete")

if __name__ == "__main__":
    main()


"""
Example commands using synchronous OpenAI API:

# Basic usage with default concurrency (50)
python src/step_2_abridged_sync.py \
    --input_file experiments/editorial/step1_labels.json \
    --output_file experiments/editorial/similarity-data.jsonl \
    --model_name gpt-4o-mini \
    --text_col_name label \
    --text_col_name_2 description \
    --sample_size 100_000 \
    --concurrency 50 \
    --checkpoint_every 100 \
    --temp_dir experiments/editorial/temp_batches

# Higher concurrency for faster processing
python src/step_2_abridged_sync.py \
    --input_file experiments/emotions/step1_labels.json \
    --output_file experiments/emotions/similarity-data.jsonl \
    --model_name gpt-4o-mini \
    --prompt_template emotions \
    --min_sim_threshold 0.5 \
    --max_sim_threshold 0.9 \
    --text_col_name label \
    --text_col_name_2 description \
    --sample_size 200_000 \
    --concurrency 100 \
    --checkpoint_every 200 \
    --temp_dir experiments/emotions/temp_batches

# Debug mode (smaller sample)
python src/step_2_abridged_sync.py \
    --input_file experiments/hate-speech/step1_labels.json \
    --output_file experiments/hate-speech/similarity-data.jsonl \
    --model_name gpt-4o-mini \
    --prompt_template hate_speech \
    --min_sim_threshold 0.3 \
    --max_sim_threshold 0.9 \
    --text_col_name label \
    --text_col_name_2 description \
    --sample_size 10_000 \
    --concurrency 20 \
    --debug \
    --temp_dir experiments/hate-speech/temp_batches
"""








"""

import logging
import json
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
from prompts import MultiSimilarityResponse

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Test parameters
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Use a smaller model for testing
test_prompt = \"""
I will show you 2 pairs of labels.
Are the two labels in each pair describing similar concepts?
Answer with "Yes" or "No". Answer each in a JSON in the format:
{
  "pairs": [
    {
      "pair_idx": 1,
      "label": "Yes" or "No"
    },
    {
      "pair_idx": 2,
      "label": "Yes" or "No"
    }
  ]
}

1. Label 1: "Happy" Label 2: "Joyful"
2. Label 1: "Angry" Label 2: "Sad"
\"""

# Step 1: Set up sampling parameters
logging.info("Step 1: Setting up sampling parameters")
json_schema = MultiSimilarityResponse.model_json_schema()
guided_decoding_params = GuidedDecodingParams(json=json_schema)
sampling_params = SamplingParams(temperature=0.1, max_tokens=1024, guided_decoding=guided_decoding_params)
logging.info(f"Sampling parameters: {sampling_params}")

# Step 2: Load model and tokenizer
logging.info("\nStep 2: Loading model and tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LLM(
    model_name,
    dtype=torch.float16,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=10_000,
    gpu_memory_utilization=0.95,
)
logging.info("Model and tokenizer loaded")

# Step 3: Format prompt with system message
logging.info("\nStep 3: Formatting prompt with system message")
prompt_dicts = [
    {
        "role": "system",
        "content": "You are an experienced analyst.",
    },
    {
        "role": "user",
        "content": test_prompt,
    },
]
logging.info(f"Prompt dicts: {json.dumps(prompt_dicts, indent=2)}")

# Step 4: Apply chat template
logging.info("\nStep 4: Applying chat template")
formatted_prompt = tokenizer.apply_chat_template(prompt_dicts, tokenize=False, add_generation_prompt=True)
logging.info(f"Formatted prompt:\n{formatted_prompt}")

# Step 5: Generate response
logging.info("\nStep 5: Generating response")
results = model.generate([formatted_prompt], sampling_params=sampling_params)
logging.info(f"Raw results type: {type(results)}")
logging.info(f"Number of results: {len(results)}")

# Step 6: Sort results by request ID
logging.info("\nStep 6: Sorting results by request ID")
sorted_results = sorted(results, key=lambda x: int(x.request_id))
logging.info(f"Number of sorted results: {len(sorted_results)}")

# Step 7: Extract text from results
logging.info("\nStep 7: Extracting text from results")
text_results = [x.outputs[0].text for x in sorted_results]
logging.info(f"Text results:\n{json.dumps(text_results, indent=2)}")

# Step 8: Try parsing the JSON response
logging.info("\nStep 8: Parsing JSON response")
try:
    parsed_result = json.loads(text_results[0])
    logging.info(f"Parsed result:\n{json.dumps(parsed_result, indent=2)}")
except json.JSONDecodeError as e:
    logging.error(f"Failed to parse JSON: {str(e)}")
    logging.error(f"Raw response: {text_results[0]}")

# Cleanup
logging.info("\nCleaning up")
del model
torch.cuda.empty_cache()
"""