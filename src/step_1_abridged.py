"""
OpenAI-only version of Step 1: Initial Labeling
MEMORY-EFFICIENT VERSION - generates prompts on-demand instead of all at once.
Works on machines with limited RAM (e.g., 8GB MacBook Air).
"""

import time
_start_time = time.time()
print("🚀 Starting Step 1: Initial Labeling (OpenAI mode)")
print("⏳ Loading libraries...\n")

import pandas as pd
import os, json
import logging
from tqdm.auto import tqdm

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
# PROCESSING FUNCTIONS
# ============================================================================

def process_all_openai(prompts, model_name, batch_size, debug_mode, temp_dir, 
                       num_sents_per_prompt, response_format, run_id, experiment,
                       max_concurrent_batches=2):
    """
    Process all prompts using OpenAI's batch API.
    Rate-limit aware - only launches a few batches at a time.
    """
    from openai import OpenAI
    import jsonlines
    from openai.lib._parsing._completions import type_to_response_format_param
    
    logging.info("[OpenAI] Initializing client...")
    client = OpenAI()
    
    if temp_dir is None:
        temp_dir = os.path.join(os.getcwd(), 'temp_batches')
    os.makedirs(temp_dir, exist_ok=True)
    
    prompt_ids = prompts['index'].tolist()
    prompt_texts = prompts['prompt'].tolist()
    
    # Split into batches
    batches_info = []
    for i in range(0, len(prompt_texts), batch_size):
        end = min(i + batch_size, len(prompt_texts))
        batches_info.append({
            'start': i,
            'end': end,
            'prompt_ids': prompt_ids[i:end],
            'prompts': prompt_texts[i:end],
            'status': 'pending',
            'openai_batch_id': None,
        })
    
    logging.info(f"[OpenAI] Split into {len(batches_info)} batches of ~{batch_size} prompts each")
    logging.info(f"[OpenAI] Will run max {max_concurrent_batches} batches at a time (run_id: {run_id})")
    
    all_responses = []
    
    while any(b['status'] in ['pending', 'running'] for b in batches_info):
        # Count running batches
        running = [b for b in batches_info if b['status'] == 'running']
        pending = [b for b in batches_info if b['status'] == 'pending']
        
        # Launch new batches if we have room
        while len(running) < max_concurrent_batches and pending:
            batch = pending[0]
            logging.info(f"[OpenAI] Launching batch {batch['start']}-{batch['end']}...")
            
            try:
                # Create batch file
                batch_requests = []
                format_json = type_to_response_format_param(response_format)
                
                for pid, prompt in zip(batch['prompt_ids'], batch['prompts']):
                    batch_requests.append({
                        "custom_id": pid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "max_tokens": 1024,
                            "response_format": format_json,
                        }
                    })
                
                # Write to file
                batch_file_path = os.path.join(temp_dir, f'batch_{batch["start"]}_{batch["end"]}.jsonl')
                with jsonlines.open(batch_file_path, 'w') as f:
                    f.write_all(batch_requests)
                
                # Upload and create batch (use context manager for file)
                with open(batch_file_path, "rb") as batch_file_handle:
                    batch_file = client.files.create(file=batch_file_handle, purpose="batch")
                
                openai_batch = client.batches.create(
                    input_file_id=batch_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h"
                )
                
                batch['openai_batch_id'] = openai_batch.id
                batch['status'] = 'running'
                running.append(batch)
                pending.remove(batch)
                
                logging.info(f"[OpenAI] ✅ Batch {batch['start']}-{batch['end']} launched (ID: {openai_batch.id})")
                
                if not debug_mode:
                    os.remove(batch_file_path)
                    
            except Exception as e:
                if "Enqueued token limit" in str(e) or "rate" in str(e).lower():
                    logging.warning(f"[OpenAI] Rate limit hit, waiting 30s: {e}")
                    time.sleep(30)
                else:
                    logging.error(f"[OpenAI] Error launching batch: {e}")
                    batch['status'] = 'failed'
                    pending.remove(batch)
                break
        
        # Check status of running batches
        for batch in running[:]:  # Copy list to allow modification
            try:
                status = client.batches.retrieve(batch['openai_batch_id'])
                
                if status.status == 'completed':
                    logging.info(f"[OpenAI] ✅ Batch {batch['start']}-{batch['end']} completed!")
                    
                    # Get results
                    file_response = client.files.content(status.output_file_id)
                    for line in file_response.iter_lines():
                        if line:
                            result = json.loads(line)
                            try:
                                response_text = result['response']['body']['choices'][0]['message']['content']
                                all_responses.append({
                                    'custom_id': result['custom_id'],
                                    'response': json.loads(response_text),
                                })
                            except (KeyError, json.JSONDecodeError) as e:
                                logging.warning(f"[OpenAI] Error parsing response: {e}")
                    
                    batch['status'] = 'completed'
                    running.remove(batch)
                    
                elif status.status == 'failed':
                    error_info = getattr(status, 'errors', None)
                    if error_info:
                        # Try to get error message from errors object
                        error_msg = str(error_info)
                    else:
                        error_msg = 'Unknown error'
                    logging.error(f"[OpenAI] ❌ Batch {batch['start']}-{batch['end']} failed: {error_msg}")
                    batch['status'] = 'failed'
                    running.remove(batch)
                    
                # Log other statuses occasionally
                elif status.status in ['validating', 'in_progress', 'finalizing']:
                    pass  # Normal, just waiting
                    
            except Exception as e:
                logging.warning(f"[OpenAI] Error checking batch {batch['start']}-{batch['end']}: {e}")
                # Don't remove - just continue checking next time
        
        # Progress update
        completed = len([b for b in batches_info if b['status'] == 'completed'])
        failed = len([b for b in batches_info if b['status'] == 'failed'])
        running_count = len([b for b in batches_info if b['status'] == 'running'])
        pending_count = len([b for b in batches_info if b['status'] == 'pending'])
        
        if running_count > 0 or pending_count > 0:
            logging.info(f"[OpenAI] Progress: {completed} ✅ | {running_count} 🔄 | {pending_count} ⏳ | {failed} ❌")
            time.sleep(10)
    
    # Check for debug mode
    if debug_mode:
        logging.info("[OpenAI] Debug mode - stopping before processing results")
        return None
    
    # Check if we got any responses
    if not all_responses:
        logging.error("[OpenAI] ❌ No responses received! Check batch status on OpenAI dashboard.")
        return None
    
    logging.info(f"[OpenAI] ✅ All batches complete! Got {len(all_responses)} responses")
    
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument('--input_data_file', type=str, required=True)
    parser.add_argument('--experiment', type=str, default='editorials')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--debug_mode', action='store_true')
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
    
    logging.info(f"[Step 5/5] Starting OpenAI API calls...")
    model_outputs = process_all_openai(
        prompts=prompts, 
        model_name=args.model, 
        batch_size=args.batch_size, 
        debug_mode=args.debug_mode, 
        temp_dir=args.temp_dir, 
        num_sents_per_prompt=args.num_sents_per_prompt,
        response_format=prompts['response_format'].iloc[0],
        experiment=args.experiment,
        run_id=f"{args.experiment}__{args.model.replace('/', '-')}__{args.start_idx}_{args.end_idx}"
    )
    
    if model_outputs is not None and len(model_outputs) > 0:
        if save_outputs(output_fname, model_outputs):
            logging.info(f'[Step 5/5] ✅ Complete! Output saved to: {output_fname}')
            logging.info(f'Sample output: {model_outputs.iloc[0].to_dict()}')
    else:
        logging.error(f'[Step 5/5] ❌ No results to save. Check OpenAI dashboard for batch status.')
    
    total_time = time.time() - _start_time
    logging.info(f"\n🎉 Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")