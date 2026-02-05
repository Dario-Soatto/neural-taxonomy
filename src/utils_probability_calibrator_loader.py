"""
model_wrappers.py
-----------------
Utility helpers for building the callable log-probability functions expected by the
refactored `ProbabilityCalibrator` class (see `assess_clusters.py`).

A *backend* (HF Transformers, TogetherAI, vLLM, etc.) only needs to expose three
Python callables:

1. `choice_log_prob_sums_single(question: str, choices: list[str]) -> list[float]`
   Returns the *sum* of log-probabilities for each choice text appended to the
   prompt produced by `ProbabilityCalibrator.format_prompt(...)`.

2. `choice_log_prob_sums_batch(question: str, choices: list[str]) -> list[float]`
   Same as (1) but may use batching for efficiency. If the backend has no batch
   API, this function can simply call (1) in a loop.

3. `full_log_prob(text: str) -> float`
   Returns the sum of log-probabilities for *all* tokens in `text`.

The helpers below generate these three functions for HuggingFace models, the
TogetherAI completion endpoint, and vLLM (local/text generation).  OpenAI is
NOT supported because the ChatCompletion endpoint does not expose prompt
log-probs for the entire prompt in a reliable way.
"""
from __future__ import annotations

from typing import Callable, Tuple, List, Optional
import logging
import numpy as np
import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# HF TRANSFORMERS
# ------------------------------------------------------------

def build_hf_logprob_fns(
    model_name_or_path,
    cache_dir: Optional[str] = None,
    device_map: str = "auto"
) -> Tuple[Callable, Callable, Callable]:
    """Return (single_fn, batch_fn, full_fn) using local HF model.
    
    Functions isolate and score only the choice tokens when scoring prompt+choice.
    
    Function signatures:
    - choice_log_prob_sums_single(prompt: str, choices: List[str]) -> List[float]
    - choice_log_prob_sums_batch(prompt: str, choices: List[str]) -> List[float]
    - full_log_prob(text: str) -> float
    """
    # Lazily import HF libraries to avoid dependency issues
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model and tokenizer
    try:
        if isinstance(model_name_or_path, str):
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                device_map=device_map,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        else:
            model = model_name_or_path
            tokenizer = cache_dir
            if tokenizer is None:
                raise ValueError("Tokenizer instance must be provided when passing a model instance.")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logging.error(f"Error loading model {model_name_or_path}: {e}")
        raise

    # Single function for scoring prompt+choice combinations
    def choice_log_prob_sums_single(prompt: str, choices: List[str]) -> List[float]:
        """Score prompt+choices pairs with HF model.
        
        For each choice, we isolate and score only the choice tokens.
        """
        if not choices:
            return []
            
        # Identify the prompt token IDs first
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        
        results = []
        for choice in choices:
            # Create full text with prompt+choice
            full_text = prompt + choice
            
            # Tokenize
            inputs = tokenizer(full_text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate with logprobs
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
            # Calculate token-by-token logprobs
            log_probs = torch.log_softmax(logits, dim=-1)
            token_ids = inputs["input_ids"][0]
            token_log_probs = [log_probs[0, i, token_ids[i+1]].item() for i in range(len(token_ids)-1)]
            
            # Only take logprobs for the choice tokens
            choice_logprobs = token_log_probs[prompt_len:] if prompt_len < len(token_log_probs) else []
            log_prob_sum = sum(choice_logprobs)
            results.append(log_prob_sum)
            
        return results
        
    # HuggingFace batch function
    def choice_log_prob_sums_batch(prompts, choices, batch_size: int = 8) -> List[float]:
        """Score prompt+choice pairs in batches with HF model.
        
        For each choice, we isolate and score only the choice tokens.
        
        Args:
            prompts: Either a single prompt string (used for all choices) or a list of prompts 
                    (one per choice) of the same length as choices.
            choices: List of choice strings to score.
            batch_size: Number of items to process in each batch.
            
        Returns:
            List of log probability sums (one per choice).
        """
        if not choices:
            return []
        
        # Convert single prompt to list if needed
        if isinstance(prompts, str):
            prompts = [prompts] * len(choices)
        elif len(prompts) != len(choices):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of choices ({len(choices)})")
        
        results = []
        
        # Process in batches
        for i in range(0, len(choices), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_choices = choices[i:i+batch_size]
            
            # Create full texts
            batch_texts = [p + c for p, c in zip(batch_prompts, batch_choices)]
            
            # Tokenize all texts in batch
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate with logprobs for the batch
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
            # Calculate token-by-token logprobs for each sequence in batch
            log_probs = torch.log_softmax(logits, dim=-1)
            
            for b, (prompt, choice) in enumerate(zip(batch_prompts, batch_choices)):
                token_ids = inputs["input_ids"][b]
                # Only calculate logprobs where attention mask is 1 (not padding)
                valid_positions = inputs["attention_mask"][b].bool()
                valid_positions_list = valid_positions.tolist()
                
                # Determine prompt length for this specific prompt
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_len = len(prompt_tokens)
                
                # Calculate logprobs for tokens where next position is valid
                token_log_probs = []
                for i in range(len(token_ids) - 1):
                    if valid_positions_list[i] and valid_positions_list[i+1]:
                        token_log_probs.append(log_probs[b, i, token_ids[i+1]].item())
                
                # Only take logprobs for the choice tokens (skip the prompt)
                choice_logprobs = token_log_probs[prompt_len:] if prompt_len < len(token_log_probs) else []
                log_prob_sum = sum(choice_logprobs)
                results.append(log_prob_sum)
        
        return results
    
    # Full text scoring
    def full_log_prob(text: str) -> float:
        """Get full log probability of a text using HF model."""
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with logprobs
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Calculate token-by-token logprobs
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids = inputs["input_ids"][0]
        token_log_probs = [log_probs[0, i, token_ids[i+1]].item() for i in range(len(token_ids)-1)]
        
        # Sum all logprobs
        return sum(token_log_probs)
    
    return choice_log_prob_sums_single, choice_log_prob_sums_batch, full_log_prob

# ------------------------------------------------------------
# TOGETHER AI (remote completions API)
# ------------------------------------------------------------

def build_together_logprob_fns(model_id: str, api_key: Optional[str] = None) -> Tuple[Callable, Callable, Callable]:
    """Return (single_fn, batch_fn, full_fn) using the TogetherAI API.
    
    API is accessed through togethera.ai client library. You need an API key
    set in the environment or passed directly.
    
    Functions isolate and score only the choice tokens when scoring prompt+choice.
    
    Function signatures:
    - choice_log_prob_sums_single(prompt: str, choices: List[str]) -> List[float]
    - choice_log_prob_sums_batch(prompt: str, choices: List[str]) -> List[float]
    - full_log_prob(text: str) -> float
    """
    try:
        from together import Together
    except ImportError:
        raise ImportError("To use Together models, install the 'together' package: pip install together")

    # Set up API client
    if api_key is None:
        api_key = os.environ.get("TOGETHER_API_KEY")
    
    if not api_key:
        raise ValueError("Together API key must be provided or set in TOGETHER_API_KEY env var.")
    
    client = Together(api_key=api_key)
    
    # Get the model info to load the tokenizer
    try:
        model_info = client.models.info(model_id)
        tokenizer_name = model_info.config.architecture.tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logging.error(f"Failed to load tokenizer for model_id {model_id}. {e}")
        # Fallback to default tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    # Single function for scoring prompt+choice combinations
    def choice_log_prob_sums_single(prompt: str, choices: List[str]) -> List[float]:
        """Score prompt+choices pairs in Together API.
        
        For each choice, we isolate and score only the choice tokens.
        """
        if not choices:
            return []
        
        # Tokenize prompt to determine token boundaries
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        
        # Make multiple requests (could be batched in a different function)
        results = []
        for choice in choices:
            # Create full text for this choice
            full_text = prompt + choice
            
            # Get logprobs for full text
            try:
                response = client.completions.create(
                    model=model_id,
                    prompt=full_text,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=5,
                )
            except Exception as e:
                logging.error(f"TogetherAI API error: {e}")
                results.append(-float("inf"))
                continue
            
            # Check if logprobs were returned
            if not response.prompt_logprobs:
                logging.warning("TogetherAI did not return prompt_logprobs.")
                results.append(-float("inf"))
                continue
            
            # Extract only the choice tokens' logprobs and sum them
            choice_logprobs = []
            for i, token_info in enumerate(response.prompt_logprobs):
                if i >= prompt_len:  # Only consider tokens after the prompt
                    token_id = token_info.token_id
                    logprob = token_info.logprob
                    choice_logprobs.append(logprob)
            
            log_prob_sum = sum(choice_logprobs)
            results.append(log_prob_sum)
            
        return results
        
    # TogetherAI batch function
    def choice_log_prob_sums_batch(prompts, choices, batch_size: int = 5) -> List[float]:
        """Score prompt+choices pairs in Together API using batching.
        
        For each choice, we isolate and score only the choice tokens.
        Batch size is limited due to API constraints. Default is 5.
        
        Args:
            prompts: Either a single prompt string (used for all choices) or a list of prompts 
                    (one per choice) of the same length as choices.
            choices: List of choice strings to score.
            batch_size: Number of items to process in each batch.
            
        Returns:
            List of log probability sums (one per choice).
        """
        if not choices:
            return []
        
        # Convert single prompt to list if needed
        if isinstance(prompts, str):
            prompts = [prompts] * len(choices)
        elif len(prompts) != len(choices):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of choices ({len(choices)})")
        
        # Process in batches
        results = []
        for i in range(0, len(choices), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_choices = choices[i:i+batch_size]
            
            # Call the individual function for each pair in the batch
            # This is a simple implementation until Together API supports true batching
            batch_results = [choice_log_prob_sums_single(p, [c])[0] for p, c in zip(batch_prompts, batch_choices)]
            results.extend(batch_results)
        
        return results
    
    # Full text scoring
    def full_log_prob(text: str) -> float:
        """Get full log probability of a text using TogetherAI API."""
        try:
            response = client.completions.create(
                model=model_id,
                prompt=text,
                max_tokens=1,
                temperature=0.0,
                logprobs=5,
            )
        except Exception as e:
            logging.error(f"TogetherAI API error: {e}")
            return -float("inf")
            
        # Check if logprobs were returned
        if not response.prompt_logprobs:
            logging.warning("TogetherAI did not return prompt_logprobs.")
            return -float("inf")
            
        # Sum all logprobs
        log_prob_sum = sum(token_info.logprob for token_info in response.prompt_logprobs)
        return log_prob_sum
    
    return choice_log_prob_sums_single, choice_log_prob_sums_batch, full_log_prob

# ------------------------------------------------------------
# vLLM (local inference)
# ------------------------------------------------------------

def build_vllm_logprob_fns(model_name_or_path, tokenizer=None, device_map='auto'):
    """Create functions for getting log probabilities using vLLM."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    try:
        # Initialize the LLM model and parameters
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
        gpu_mem_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
        llm = LLM(
            model_name_or_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            enforce_eager=True,
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:
        print(f"ERROR: Failed to initialize vLLM. {e}")
        raise
        
    # Default sampling parameters for logprobs
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    
    def choice_log_prob_sums_single(prompt, choices):
        """Get log probability sums for each choice when appended to the prompt."""
        if not choices:
            return []
        
        # Calculate logprobs for choices
        results = []
        for choice in choices:
            full_text = prompt + choice
            
            # Generate with vLLM
            output = llm.generate([full_text], sampling_params=sampling_params, use_tqdm=False)[0]
            
            if not output.prompt_logprobs:
                raise ValueError("No prompt_logprobs retrieved from vLLM.")
                
            # Get prompt token count to identify boundaries
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
            
            # Extract only the choice token logprobs by skipping prompt tokens
            all_logprobs = output.prompt_logprobs
            choice_logprobs = all_logprobs[prompt_len:] if prompt_len < len(all_logprobs) else []
            
            # Handle the complex logprob structure from vLLM
            # choice_logprobs is a list of dictionaries mapping token IDs to Logprob objects
            log_prob_sum = 0.0
            for token_dict in choice_logprobs:
                # Each token_dict is {token_id: Logprob(logprob=X, rank=Y, decoded_token=Z)}
                for logprob_obj in token_dict.values():
                    log_prob_sum += logprob_obj.logprob
                    break  # Only need the first (top) logprob for each token
            
            results.append(log_prob_sum)
            
        return results
    
    # vLLM batch function
    def choice_log_prob_sums_batch(prompts, choices, batch_size=16):
        """Get log probability sums for each choice with batch processing.
        
        Args:
            prompts: Either a single prompt string (used for all choices) or a list of prompts 
                    (one per choice) of the same length as choices.
            choices: List of choice strings to score.
            batch_size: Number of items to process in each batch.
            
        Returns:
            List of log probability sums (one per choice).
        """
        if not choices:
            return []
        
        # Convert single prompt to list if needed
        if isinstance(prompts, str):
            prompts = [prompts] * len(choices)
        elif len(prompts) != len(choices):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of choices ({len(choices)})")
        
        results = []
        
        # Process in batches
        for i in range(0, len(choices), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_choices = choices[i:i+batch_size]
            
            # Create full texts for each prompt+choice pair
            batch_texts = [p + c for p, c in zip(batch_prompts, batch_choices)]
            
            # Generate with vLLM batch
            batch_outputs = llm.generate(batch_texts, sampling_params=sampling_params, use_tqdm=False)
            
            # Process each output
            for j, (prompt, output) in enumerate(zip(batch_prompts, batch_outputs)):
                if not output.prompt_logprobs:
                    raise ValueError(f"No prompt_logprobs retrieved from vLLM for batch item {j}.")
                    
                # Determine prompt length for this specific prompt
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_len = len(prompt_tokens)
                
                # Extract only the choice token logprobs by skipping prompt tokens
                all_logprobs = output.prompt_logprobs
                choice_logprobs = all_logprobs[prompt_len:] if prompt_len < len(all_logprobs) else []
                
                # Handle the complex logprob structure
                log_prob_sum = 0.0
                for token_dict in choice_logprobs:
                    if token_dict is None:
                        continue
                    for logprob_obj in token_dict.values():
                        log_prob_sum += logprob_obj.logprob
                        break  # Only need the first (top) logprob for each token
                
                results.append(log_prob_sum)
                
        return results
    
    def full_text_log_prob_sum(text):
        """Get log probability sum for the entire text."""
        # Generate with vLLM
        output = llm.generate([text], sampling_params=sampling_params, use_tqdm=False)[0]
        
        if not output.prompt_logprobs:
            raise ValueError("No prompt_logprobs retrieved from vLLM.")
            
        # Sum the log probabilities for all tokens, handling the complex structure
        log_prob_sum = 0.0
        for token_dict in output.prompt_logprobs:
            if token_dict is None:
                continue
            for logprob_obj in token_dict.values():
                log_prob_sum += logprob_obj.logprob
                break  # Only need the first (top) logprob for each token
        
        return log_prob_sum
    
    return choice_log_prob_sums_single, choice_log_prob_sums_batch, full_text_log_prob_sum

if __name__ == "__main__":
    # This is a simple test script for the vLLM logprob functions.
    # Make sure you have a vLLM-compatible model and the necessary environment setup.
    # Example usage: python src/model_wrappers.py
    import os
    import time
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    model_for_test = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Or your specific model
    
    test_prompt = """
Please answer the following question:

<question>
What animal says moo?
</question>

by choosing the best of the following options:

<choices>
Cow
Dog
Cat
</choices>

Your response: """
    test_choices = ["Cow", "Dog", "Cat"]
    print(f"Testing vLLM logprob functions with model: {model_for_test}")
    print(f"Test prompt: '{test_prompt}'")
    print(f"Test choices: {test_choices}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Initialize LLM and Tokenizer (mimicking build_vllm_logprob_fns setup)
    print("Initializing vLLM engine...")
    llm_instance = LLM(model_for_test)
    print("Initializing Tokenizer...")
    tokenizer_instance = AutoTokenizer.from_pretrained(model_for_test)
    
    sampling_params_instance = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    print("SamplingParams initialized.")

    # Process full text with prompt + first choice
    prompt_tokens = tokenizer_instance.encode(test_prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)
    full_text = test_prompt + test_choices[0]
    print(f"Generating with vLLM for prompt+choice...")
    output = llm_instance.generate([full_text], sampling_params=sampling_params_instance, use_tqdm=False)[0]
    
    if output.prompt_logprobs:
        # Extract only logprobs for the choice tokens
        all_logprobs = output.prompt_logprobs
        choice_logprobs = all_logprobs[prompt_len:] if prompt_len < len(all_logprobs) else []
        
        # Handle the complex logprob structure
        choice_sum = 0.0
        for token_dict in choice_logprobs:
            for logprob_obj in token_dict.values():
                choice_sum += logprob_obj.logprob
                break  # Only need the first (top) logprob for each token
                
        choice_avg = choice_sum / len(choice_logprobs) if choice_logprobs else 0
        
        print(f"Successfully retrieved prompt+choice logprobs.")
        print(f"  Total tokens: {len(output.prompt_token_ids)}")
        print(f"  Prompt tokens: {prompt_len}")
        print(f"  Choice tokens: {len(choice_logprobs)}")
        print(f"  Sum of log probabilities for CHOICE ONLY: {choice_sum:.4f}")
        print(f"  Average log probability for CHOICE ONLY: {choice_avg:.4f}")
        print(f"  Example token dict format: {choice_logprobs[0] if choice_logprobs else 'No choice tokens'}")
    else:
        print("Error: prompt_logprobs were not returned by vLLM.")
        
    # Test 3: Using the wrapper functions
    print("\nTest 3: Using the model wrapper functions")
    
    # Create all the wrapper functions using our helper
    single_fn, batch_fn, full_fn = build_vllm_logprob_fns(model_for_test, tokenizer_instance)
    
    # Scoring with single function
    print("Calling single scoring function")
    start_time = time.time()
    single_results = single_fn(test_prompt, test_choices)
    single_time = time.time() - start_time
    
    # Scoring with batch function
    print("Calling batch scoring function")
    start_time = time.time()
    batch_results = batch_fn(test_prompt, test_choices)
    batch_time = time.time() - start_time
    
    print("\nResults comparison:")
    print("  Single function results:")
    for choice, score in zip(test_choices, single_results):
        print(f"    Choice '{choice}': {score:.4f}")
    print(f"    Time taken: {single_time:.2f} seconds")
    
    print("\n  Batch function results:")
    for choice, score in zip(test_choices, batch_results):
        print(f"    Choice '{choice}': {score:.4f}")
    print(f"    Time taken: {batch_time:.2f} seconds")
    
    print("\nAPI benefits:")
    print("  1. Cleaner interface - pass prompt and choices separately")
    print("  2. More efficient batch processing with explicit prompt+choices format")
    print("  3. Explicit scoring of only the choice tokens")
