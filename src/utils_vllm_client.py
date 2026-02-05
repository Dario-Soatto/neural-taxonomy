import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata

import os
import json
import torch
import ast
import logging
import re
import sys

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Disable structured outputs to prevent import errors/leaks/crashes
# We will rely on robust regex parsing instead.
GuidedDecodingParams = None
StructuredOutputsParams = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

_model = None
_tokenizer = None


def load_model(model_name: str):
    global _model, _tokenizer
    if _model is None:
        try:
            # Clear any existing CUDA cache
            torch.cuda.empty_cache()

            # Get number of available GPUs
            num_gpus = torch.cuda.device_count()
            logging.info(f"Found {num_gpus} GPUs")

            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = LLM(
                model_name,
                dtype=torch.float16,
                tensor_parallel_size=num_gpus,  # Use all available GPUs
                max_model_len=10_000,
                gpu_memory_utilization=0.90,  # 0.90 is safer than 0.95
                trust_remote_code=True,
                enforce_eager=True,  # Enable eager mode for better memory management
            )
            logging.info(f"Model loaded with tensor_parallel_size={num_gpus}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            # Clean up on error
            if _model is not None:
                del _model
            if _tokenizer is not None:
                del _tokenizer
            torch.cuda.empty_cache()
            raise e
    return _tokenizer, _model


def run_vllm_batch(
    prompts,
    model_name,
    sampling_params=None,
    include_system_prompt=True,
    batch_size=100,  
    debug_mode=False,
    temp_dir=None,
    batch_type="similarity",
    response_format=None,
    verbose=False,
):
    # REMOVED outer try/except to prevent silent failures
    
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.1, max_tokens=2048)
    
    # We purposefully IGNORE response_format/GuidedDecoding here
    # to avoid the nanobind crashes. We will parse the output manually later.

    tokenizer, model = load_model(model_name)
    
    if include_system_prompt:
        prompt_dicts = list(map(lambda x: [
            { "role": "system", "content": "You are an experienced analyst.", },
            { "role": "user", "content": x, },
        ], prompts))
    else:
        prompt_dicts = list(map(lambda x: [
            { "role": "user", "content": x, },
        ], prompts))

    formatted_prompts = list(map(lambda x:
        tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True), prompt_dicts
    ))

    # [DEBUG] Print first prompt to verify we aren't sending empty garbage
    if len(formatted_prompts) > 0:
        print(f"\n[DEBUG_UTILS] First Prompt Preview:\n{'-'*20}\n{formatted_prompts[0][:200]}...\n{'-'*20}", file=sys.stderr)

    # Process in smaller batches to manage memory
    all_results = []
    for i in range(0, len(formatted_prompts), batch_size):
        batch_prompts = formatted_prompts[i:i + batch_size]
        batch_results = model.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=verbose)
        all_results.extend(batch_results)
        
        # [DEBUG] Check the output of the first item in this batch
        if len(batch_results) > 0 and batch_results[0].outputs:
            safe_out = batch_results[0].outputs[0].text[:100].replace(chr(10), ' ')
            print(f"[DEBUG_UTILS] Batch {i} First Result: {safe_out}", file=sys.stderr)
        
        # Clear CUDA cache after each batch
        torch.cuda.empty_cache()

    sorted_results = sorted(all_results, key=lambda x: int(x.request_id))
    text_results = []
    for item in sorted_results:
        # NO try/except here. If text is None, we want to know.
        text = ""
        if item.outputs and len(item.outputs) > 0:
            text = item.outputs[0].text
        
        if response_format is not None:
            # Pass raw text wrapped in _raw so Step 2 regex can handle it
            text_results.append({"_raw": text})
        else:
            text_results.append(text)
            
    return text_results


def write_to_file(fname, indices=None, outputs=None, index_output_chunk=None):
    with open(fname, 'ab') as file:
        if index_output_chunk is None:
            index_output_chunk = zip(indices, outputs)

        for index, output in index_output_chunk:
            if isinstance(output, str):
                response = output
            else:
                response = output.outputs[0].text
            response = unicodedata.normalize('NFKC', response)
            if response and index:
                output = {}
                output['index'] = str(index)
                parsed = robust_parse_outputs(response)
                output['response'] = parsed
                if parsed is None:
                    output['response_raw'] = response
                file.write(json.dumps(output).encode('utf-8'))
                file.write(b'\n')


def extract_list_brackets_from_text(text):
    pattern = r'\[.*?\]'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def extract_braces_from_text(text):
    pattern = r'\{.*?\}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _normalize_json_text(text):
    if text is None:
        return None
    text = unicodedata.normalize('NFKC', str(text)).strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # Normalize curly quotes to straight quotes.
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    # Remove trailing commas before closing braces/brackets.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def robust_parse_outputs(output):
    output = _normalize_json_text(output)
    if output is None:
        return None
    extracted = extract_list_brackets_from_text(output) or extract_braces_from_text(output) or output
    try:
        return _robust_parse_outputs(extracted)
    except Exception:
        try:
            return _robust_parse_outputs(extracted.replace('\n', ' '))
        except Exception:
            # Last-ditch regex extraction for label/description.
            try:
                label_match = re.search(r'"label"\s*:\s*"([^"]+)"', extracted)
                desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', extracted)
                if label_match and desc_match:
                    return {"label": label_match.group(1), "description": desc_match.group(1)}
            except Exception:
                pass
            return None


def _robust_parse_outputs(output):
    try:
        return ast.literal_eval(output)
    except Exception:
        try:
            return json.loads(output)
        except Exception:
            return None


def check_output_validity(output):
    """
    Check if the outputs are valid.
    """
    if robust_parse_outputs(output) is None:
        return False
    return True
