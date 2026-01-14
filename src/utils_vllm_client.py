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

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None

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
                gpu_memory_utilization=0.85,  # Reduced from 0.95 to leave more headroom
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
    batch_size=100,  # Reduced from 5000
    debug_mode=False, 
    temp_dir=None, 
    batch_type="similarity",
    response_format=None,
    verbose=False,
):
    try:
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)
        if response_format is not None and GuidedDecodingParams is not None:
            guided_decoding_params = GuidedDecodingParams(json=response_format.model_json_schema())
            sampling_params = SamplingParams(temperature=0.1, max_tokens=1024, guided_decoding=guided_decoding_params)

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

        # Process in smaller batches to manage memory
        all_results = []
        for i in range(0, len(formatted_prompts), batch_size):
            batch_prompts = formatted_prompts[i:i + batch_size]
            batch_results = model.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=verbose)
            all_results.extend(batch_results)
            # Clear CUDA cache after each batch
            torch.cuda.empty_cache()

        sorted_results = sorted(all_results, key=lambda x: int(x.request_id))
        if response_format is not None:
            text_results = list(map(lambda x: json.loads(x.outputs[0].text), sorted_results))
        else:
            text_results = list(map(lambda x: x.outputs[0].text, sorted_results))
        return text_results
    except Exception as e:
        logging.error(f"Error in run_vllm_batch: {str(e)}")
        raise e
    finally:
        # Clean up
        torch.cuda.empty_cache()


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
                output['response'] = robust_parse_outputs(response)
                file.write(json.dumps(output).encode('utf-8'))
                file.write(b'\n')


def extract_list_brackets_from_text(text):
    pattern = r'\[.*\]'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


import json, ast 
def robust_parse_outputs(output):
    output = extract_list_brackets_from_text(output)
    if output is None:
        return None
    try:
        return _robust_parse_outputs(output)
    except:
        try:
            return _robust_parse_outputs(output.replace('\n', ' '))
        except:
            return None


def _robust_parse_outputs(output):
    try:
        return ast.literal_eval(output)
    except:
        try:
            return json.loads(output)
        except:
            return None


def check_output_validity(output):
    """
    Check if the outputs are valid.
    """
    if robust_parse_outputs(output) is None:
        return False
    return True
