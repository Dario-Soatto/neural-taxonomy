"""
OpenAI-only version of Step 1: Initial Labeling
SYNCHRONOUS API VERSION - uses direct API calls instead of batch API.
Faster and more reliable than batch API when batches are stuck.
Uses concurrent requests for speed.
"""

import time
_start_time = time.time()
print("🚀 Starting Step 1: Initial Labeling (Synchronous API mode)")
print("⏳ Loading libraries...\n")

import pandas as pd
import os, json
import logging
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import (
    EDITORIAL_INITIAL_LABELING_PROMPT, 
    MULTI_SENTENCE_EDITORIAL_LABELING_PROMPT,
    EditorialLabelingResponse,
    MultiSentenceLabelingResponse,
    SINGLE_COMMENT_EMOTION_LABELING_PROMPT,
    MULTI_SENTENCE_EMOTION_LABELING_PROMPT,
    SingleCommentLabelingResponse,
    MultiCommentLabelingResponse,
    HATE_SPEECH_LABELING_PROMPT,
    HateSpeechLabelingResponse,
    MultiHateSpeechLabelingResponse,
    NEWS_DISCOURSE_LABELING_PROMPT,
    NewsDiscourseLabelingResponse,
    MultiNewsDiscourseLabelingResponse,
)

print(f"✅ Libraries loaded in {time.time() - _start_time:.1f}s\n")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(f'{here}/../config.json'):
    config_data = json.load(open(f'{here}/../config.json'))
    if 'OPENAI_API_KEY' in config_data:
        os.environ['OPENAI_API_KEY'] = config_data["OPENAI_API_KEY"]


def save_outputs(output_fname, model_outputs):
    """Save the outputs to a file."""
    if model_outputs is None:
        logging.info("Debug mode: Stopped after creating batch files")
        return False
    logging.info(f'output: {model_outputs.sample(min(10, len(model_outputs)))}')
    model_outputs.to_json(output_fname, orient='records', lines=True)
    return True


def check_existing_outputs(output_fname, start_idx, end_idx):
    """Check if we already have results for this range."""
    if not os.path.exists(output_fname):
        return False
    try:
        existing_df = pd.read_json(output_fname, orient='records', lines=True)
        if len(existing_df) >= (end_idx - start_idx):
            logging.info(f"Found existing results in {output_fname} with {len(existing_df)} rows")
            return True
    except Exception as e:
        logging.warning(f"Error reading existing file {output_fname}: {e}")
    return False


# ============================================================================
# CHECKPOINTING FUNCTIONS (for sync mode)
# ============================================================================

def get_checkpoint_paths(output_fname, temp_dir):
    """Get paths for checkpoint files."""
    base = os.path.splitext(output_fname)[0]
    checkpoint_dir = temp_dir or os.path.dirname(output_fname) or '.'
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
# MEMORY-EFFICIENT PROMPT GENERATION
# ============================================================================

def build_document_index(input_df):
    """
    Build a lightweight index mapping sentence positions to document info.
    Returns: (doc_texts, sentence_list) where sentence_list is [(doc_idx, sent_idx, sent_text), ...]
    """
    logging.info("Building document index (memory-efficient)...")
    
    doc_texts = {}  # doc_index -> concatenated text
    sentence_list = []  # List of (doc_index, sent_index, sent_text)
    
    g = input_df.groupby('doc_index')
    for doc_index, df in tqdm(g, desc="Indexing documents", unit="doc"):
        df = df[['sent_index', 'sentence_text']].sort_values('sent_index')
        
        # Store document text
        doc_texts[doc_index] = df['sentence_text'].str.cat(sep=' ')
        
        # Store sentence references (not the full prompt!)
        for _, (sent_index, sentence_text) in df.iterrows():
            sentence_list.append((doc_index, sent_index, sentence_text))
    
    logging.info(f"✅ Indexed {len(doc_texts)} documents, {len(sentence_list)} sentences")
    return doc_texts, sentence_list


def generate_prompts_for_range(doc_texts, sentence_list, start_idx, end_idx, 
                                multi_sentence=False, num_sents_per_prompt=8):
    """
    Generate prompts ONLY for the specified range. Memory-efficient!
    """
    if not multi_sentence:
        # Single sentence mode
        prompts_data = []
        for i in tqdm(range(start_idx, end_idx), desc="Generating prompts", unit="prompt"):
            if i >= len(sentence_list):
                break
            doc_index, sent_index, sentence_text = sentence_list[i]
            doc_text = doc_texts[doc_index]
            
            prompt = EDITORIAL_INITIAL_LABELING_PROMPT.format(
                article=doc_text, 
                sentence=sentence_text
            )
            prompts_data.append({
                'index': f'{doc_index.replace("/", "_")}__sent_index-{sent_index}',
                'prompt': prompt,
                'doc_text': doc_text,
                'sent_text': sentence_text,
            })
        
        prompt_df = pd.DataFrame(prompts_data)
        prompt_df['response_format'] = EditorialLabelingResponse
        return prompt_df
    else:
        # Multi-sentence mode - group sentences into chunks
        # First, rebuild the chunking for the range we need
        prompts_data = []
        
        # Group sentences by document
        doc_sentences = {}
        for i, (doc_index, sent_index, sent_text) in enumerate(sentence_list):
            if doc_index not in doc_sentences:
                doc_sentences[doc_index] = []
            doc_sentences[doc_index].append((i, sent_index, sent_text))
        
        # Create chunks
        all_chunks = []
        for doc_index, sents in doc_sentences.items():
            doc_text = doc_texts[doc_index]
            sent_texts = [s[2] for s in sents]
            
            for i in range(0, len(sent_texts), num_sents_per_prompt):
                chunk = sent_texts[i:i + num_sents_per_prompt]
                chunk_formatted = list(map(
                    lambda x: f"(idx {i + x[0]}) {x[1].replace(chr(10), ' ').strip()}", 
                    enumerate(chunk)
                ))
                all_chunks.append({
                    'doc_index': doc_index,
                    'doc_text': doc_text,
                    'sentences': sent_texts,
                    'sents_chunk': '\n'.join(chunk_formatted),
                    'num_sents': len(chunk),
                    'chunk_start': i,
                })
        
        # Only generate prompts for requested range
        for i in tqdm(range(start_idx, min(end_idx, len(all_chunks))), 
                      desc="Generating prompts", unit="prompt"):
            chunk = all_chunks[i]
            prompt = MULTI_SENTENCE_EDITORIAL_LABELING_PROMPT.format(
                k=chunk['num_sents'],
                article=chunk['doc_text'],
                sentences=chunk['sents_chunk']
            )
            prompts_data.append({
                'index': f'{chunk["doc_index"].replace("/", "_")}__chunk-{chunk["chunk_start"]}',
                'prompt': prompt,
                'doc_text': chunk['doc_text'],
                'sentences': chunk['sentences'],
                'sents_chunk': chunk['sents_chunk'],
                'num_sents': chunk['num_sents'],
            })
        
        prompt_df = pd.DataFrame(prompts_data)
        prompt_df['response_format'] = MultiSentenceLabelingResponse
        return prompt_df


def count_total_prompts(input_df, multi_sentence=False, num_sents_per_prompt=8):
    """Count total prompts without creating them (for progress reporting)."""
    if not multi_sentence:
        return len(input_df)
    else:
        total = 0
        for doc_index, df in input_df.groupby('doc_index'):
            num_sents = len(df)
            num_chunks = (num_sents + num_sents_per_prompt - 1) // num_sents_per_prompt
            total += num_chunks
        return total


# ============================================================================
# PROCESSING FUNCTIONS (Synchronous API)
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


def process_all_openai_sync(prompts, model_name, temp_dir, num_sents_per_prompt, 
                            response_format, experiment, output_fname=None,
                            concurrency=50, checkpoint_every=100):
    """
    Process all prompts using synchronous OpenAI API with concurrent requests.
    Much faster than batch API when batches are stuck.
    """
    from openai import OpenAI
    
    logging.info("[Sync] Initializing OpenAI client...")
    client = OpenAI()
    
    if temp_dir is None:
        temp_dir = os.path.join(os.getcwd(), 'temp_batches')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Setup checkpointing
    checkpoint_paths = get_checkpoint_paths(output_fname, temp_dir) if output_fname else None
    completed_ids = set()
    all_responses = []
    
    if checkpoint_paths:
        completed_ids = load_checkpoint_state(checkpoint_paths)
        all_responses = load_partial_results(checkpoint_paths)
        if completed_ids:
            logging.info(f"[Checkpoint] ⏭️  Resuming: {len(completed_ids)} already completed")
    
    # Prepare prompts to process
    prompt_ids = prompts['index'].tolist()
    prompt_texts = prompts['prompt'].tolist()
    
    # Filter out already-completed prompts
    pending_prompts = [
        (pid, ptext) for pid, ptext in zip(prompt_ids, prompt_texts)
        if pid not in completed_ids
    ]
    
    if not pending_prompts:
        logging.info("[Checkpoint] ✅ All prompts already completed from previous run!")
        if all_responses:
            output_df = pd.DataFrame(all_responses)
            output_df = output_df.merge(prompts, left_on='custom_id', right_on='index')
            output_df = postprocess_outputs(output_df, experiment, num_sents_per_prompt)
            return output_df
        return None
    
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
                        
                        # Checkpoint periodically - save ALL responses
                        if checkpoint_paths and len(new_responses) % checkpoint_every == 0:
                            save_checkpoint(checkpoint_paths, all_responses, completed_ids)
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logging.warning(f"[Sync] Error processing {prompt_id}: {e}")
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({'done': len(new_responses), 'failed': failed_count})
    
    # Final checkpoint save (in case last batch wasn't a multiple of checkpoint_every)
    if checkpoint_paths and new_responses:
        save_checkpoint(checkpoint_paths, all_responses, completed_ids)
    
    logging.info(f"[Sync] ✅ Complete! {len(new_responses)} new responses, {failed_count} failed")
    
    if not all_responses:
        logging.error("[Sync] ❌ No responses received!")
        return None
    
    # Create output DataFrame
    output_df = pd.DataFrame(all_responses)
    output_df = output_df.merge(prompts, left_on='custom_id', right_on='index')
    output_df = postprocess_outputs(output_df, experiment, num_sents_per_prompt)
    
    return output_df


def postprocess_outputs(output_df, experiment, num_sents_per_prompt):
    """Postprocess the outputs from the model."""
    content_column = 'sentences' if experiment in ['editorials', 'news-discourse'] else 'comments'
    content_id_column = 'sentence_idx' if experiment in ['editorials', 'news-discourse'] else 'comment_idx'

    if output_df['response'].isna().sum() > 0:
        logging.warning(f'{output_df["response"].isna().sum()} responses are missing')
        output_df = output_df[output_df['response'].notna()]
    
    if num_sents_per_prompt > 1:
        logging.info("[Postprocess] Sorting sentences...")
        output_df['sentences'] = output_df['response'].apply(
            lambda x: sorted(x[content_column], key=lambda y: y[content_id_column])
        )
        output_df = output_df[['custom_id', 'sentences', 'response']]
        
        logging.info("[Postprocess] Exploding sentences...")
        full_exp_df = output_df.explode(['sentences'])
        output_df = pd.concat([
            full_exp_df[['custom_id']].reset_index(drop=True), 
            pd.DataFrame(full_exp_df['sentences'].tolist())
        ], axis=1)

    if experiment in ['hate-speech', 'emotions']:
        label_col = 'labels' if experiment == 'hate-speech' else 'comment_labels'
        output_df = output_df.rename(columns={content_id_column: 'sentence_idx'})
        output_df = output_df.loc[lambda df: df[label_col].str.len() > 0]
        output_df = output_df.explode(label_col).loc[lambda df: df[label_col].notna()]
        output_df = output_df.reset_index(drop=True)
        output_df = pd.concat([
            output_df[['custom_id', 'sentence_idx']], 
            pd.DataFrame(output_df[label_col].tolist())
        ], axis=1)
        output_df['label_idx'] = output_df.reset_index().groupby(['custom_id', 'sentence_idx'])['index'].rank(method='dense').astype(int)
    
    return output_df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 1: Initial Labeling using synchronous OpenAI API")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument('--input_data_file', type=str, required=True)
    parser.add_argument('--experiment', type=str, default='editorials')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--concurrency', type=int, default=50, help='Number of concurrent API requests (default: 50)')
    parser.add_argument('--checkpoint_every', type=int, default=100, help='Save checkpoint every N responses (default: 100)')
    parser.add_argument('--temp_dir', type=str, default=None)
    parser.add_argument('--num_sents_per_prompt', type=int, default=8)
    
    args = parser.parse_args()
    
    # ========== STEP 1: Load input data ==========
    step_start = time.time()
    logging.info(f"[Step 1/5] Loading input file: {args.input_data_file}")
    if '.json' in args.input_data_file:
        input_df = pd.read_json(args.input_data_file, orient='records', lines=True)
    elif '.csv' in args.input_data_file:
        input_df = pd.read_csv(args.input_data_file)
    else:
        raise ValueError(f'Input data file must be json or csv')
    logging.info(f"[Step 1/5] ✅ Loaded {len(input_df)} rows in {time.time() - step_start:.1f}s")
    
    # ========== STEP 2: Build document index (lightweight) ==========
    step_start = time.time()
    logging.info(f"[Step 2/5] Building document index...")
    
    multi_sentence = args.num_sents_per_prompt > 1
    doc_texts, sentence_list = build_document_index(input_df)
    
    # Free memory - we don't need the full input_df anymore
    del input_df
    import gc
    gc.collect()
    
    # Count total prompts
    if not multi_sentence:
        total_prompts = len(sentence_list)
    else:
        # For multi-sentence, we need to count chunks per document
        doc_sent_counts = {}
        for doc_idx, _, _ in sentence_list:
            doc_sent_counts[doc_idx] = doc_sent_counts.get(doc_idx, 0) + 1
        total_prompts = sum(
            (count + args.num_sents_per_prompt - 1) // args.num_sents_per_prompt 
            for count in doc_sent_counts.values()
        )
    
    logging.info(f"[Step 2/5] ✅ Total prompts available: {total_prompts} (built in {time.time() - step_start:.1f}s)")
    
    # ========== STEP 3: Set indices ==========
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = total_prompts
    
    args.end_idx = min(args.end_idx, total_prompts)
    num_to_process = args.end_idx - args.start_idx
    
    logging.info(f"[Step 3/5] ✅ Will process prompts [{args.start_idx}:{args.end_idx}] ({num_to_process} total)")
    
    # ========== STEP 4: Setup output ==========
    dirname = os.path.dirname(args.output_file)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    
    out_dirname, out_fname = os.path.split(args.output_file)
    fname, fext = os.path.splitext(out_fname)
    output_fname = f'{out_dirname}/{fname}-labeling__experiment-{args.experiment}__model_{args.model.replace("/", "-")}__{args.start_idx}_{args.end_idx}{fext}'
    
    if check_existing_outputs(output_fname, args.start_idx, args.end_idx):
        logging.info(f"[Step 4/5] ⏭️  Skipping - results already exist: {output_fname}")
        exit(0)
    
    logging.info(f"[Step 4/5] ✅ Output will be saved to: {output_fname}")
    
    # ========== STEP 5: Generate prompts ON-DEMAND and process ==========
    step_start = time.time()
    logging.info(f"[Step 5/5] Generating {num_to_process} prompts on-demand...")
    
    prompts = generate_prompts_for_range(
        doc_texts=doc_texts,
        sentence_list=sentence_list,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        multi_sentence=multi_sentence,
        num_sents_per_prompt=args.num_sents_per_prompt
    )
    
    logging.info(f"[Step 5/5] ✅ Generated {len(prompts)} prompts in {time.time() - step_start:.1f}s")
    
    # Create empty file to indicate processing
    with open(output_fname, 'w') as f:
        f.write('')
    
    logging.info(f"[Step 5/5] Starting synchronous OpenAI API calls (concurrency={args.concurrency})...")
    model_outputs = process_all_openai_sync(
        prompts=prompts, 
        model_name=args.model, 
        temp_dir=args.temp_dir, 
        num_sents_per_prompt=args.num_sents_per_prompt,
        response_format=prompts['response_format'].iloc[0],
        experiment=args.experiment,
        output_fname=output_fname,
        concurrency=args.concurrency,
        checkpoint_every=args.checkpoint_every,
    )
    
    if model_outputs is not None and len(model_outputs) > 0:
        if save_outputs(output_fname, model_outputs):
            logging.info(f'[Step 5/5] ✅ Complete! Output saved to: {output_fname}')
            logging.info(f'Sample output: {model_outputs.iloc[0].to_dict()}')
            
            # Clean up checkpoint files on successful completion
            checkpoint_paths = get_checkpoint_paths(output_fname, args.temp_dir)
            cleanup_checkpoint_files(checkpoint_paths)
    else:
        logging.error(f'[Step 5/5] ❌ No results to save. Check OpenAI dashboard for batch status.')
        logging.info(f'[Step 5/5] 💡 Partial results may be saved in checkpoint files. Re-run to resume.')
    
    total_time = time.time() - _start_time
    logging.info(f"\n🎉 Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")