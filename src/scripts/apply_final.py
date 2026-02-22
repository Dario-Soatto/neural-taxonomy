import os

UTILS_FILE = "src/utils_vllm_client.py"

# This code is a direct replacement for utils_vllm_client.py
# It removes the try/except blocks and adds aggressive logging.
new_utils_code = r'''
import pandas as pd
import numpy as np
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

# Disable structured outputs to prevent import errors/leaks
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
        num_gpus = torch.cuda.device_count()
        logging.info(f"Found {num_gpus} GPUs")
        
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = LLM(
            model_name,
            dtype=torch.float16,
            tensor_parallel_size=num_gpus,
            max_model_len=10_000,
            gpu_memory_utilization=0.90, 
            trust_remote_code=True,
            enforce_eager=True,
        )
        logging.info(f"Model loaded.")
    return _tokenizer, _model

def run_vllm_batch(
    prompts, 
    model_name, 
    sampling_params=None, 
    include_system_prompt=True, 
    batch_size=20, 
    debug_mode=False, 
    temp_dir=None, 
    batch_type="similarity",
    response_format=None,
    verbose=False,
):
    # 1. SETUP SAMPLING
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.1, max_tokens=2048)

    tokenizer, model = load_model(model_name)

    # 2. PREPARE PROMPTS
    prompt_dicts = []
    for x in prompts:
        msgs = [{"role": "user", "content": x}]
        if include_system_prompt:
            msgs.insert(0, {"role": "system", "content": "You are an experienced analyst."})
        prompt_dicts.append(msgs)
        
    formatted_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) 
        for p in prompt_dicts
    ]

    # DEBUG: Print the first prompt to ensure it's not empty
    print(f"\n[DEBUG_UTILS] First Prompt (Preview):\n{'-'*20}\n{formatted_prompts[0][:500]}...\n{'-'*20}", file=sys.stderr)

    # 3. GENERATE
    all_results = []
    # Process in batches
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        
        # Call Generate
        batch_results = model.generate(batch, sampling_params=sampling_params, use_tqdm=verbose)
        all_results.extend(batch_results)
        
        # DEBUG: Inspect the first result of the batch IMMEDIATELY
        if len(batch_results) > 0:
            first_out = batch_results[0]
            if first_out.outputs:
                raw_text = first_out.outputs[0].text
                reason = first_out.outputs[0].finish_reason
                print(f"[DEBUG_UTILS] Batch {i} First Result | Reason: {reason} | Len: {len(raw_text)}", file=sys.stderr)
                print(f"[DEBUG_UTILS] Content: {raw_text[:200].replace(chr(10), ' ')}", file=sys.stderr)
            else:
                print(f"[DEBUG_UTILS] Batch {i} First Result has NO OUTPUTS!", file=sys.stderr)
        
        torch.cuda.empty_cache()

    # 4. EXTRACT TEXT (No Try/Except swallowing!)
    sorted_results = sorted(all_results, key=lambda x: int(x.request_id))
    text_results = []
    
    for item in sorted_results:
        # Direct access - if this fails, we WANT it to crash so we know why
        text = item.outputs[0].text
        
        if response_format is not None:
            # Simple wrapper to pass data downstream
            # We skip robust_parse here to keep this file clean, 
            # Step 2 script handles the heavy parsing.
            text_results.append({"_raw": text})
        else:
            text_results.append(text)
            
    return text_results

# --- HELPER FUNCTIONS (Stubs to keep imports working) ---
def robust_parse_outputs(output):
    return None 
def check_output_validity(output):
    return False
'''

with open(UTILS_FILE, 'w') as f:
    f.write(new_utils_code)
print(f"✅ Overwrote {UTILS_FILE} with No-Mercy debug version.")