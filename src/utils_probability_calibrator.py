import logging
import itertools
import numpy as np
import torch
import math
from typing import Callable, List, Optional
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

from utils_probability_calibrator_loader import (
    build_hf_logprob_fns,
    build_together_logprob_fns,
    build_vllm_logprob_fns,
)


PREFIX_TEST_PROMPT = """
Please answer the following question:

<question>
{question}
</question>

by choosing the best of the following options:
"""

CHOICES_PROMPT = """
<choices>
{choices}
</choices>

Your response:
"""

def safe_exponentiate_torch(log_probs_tensor):
    """
    Exponentiate a log-probs GPU tensor in a numerically stable way, then normalize.
    Returns a GPU tensor of shape => sum of 1.
    """
    max_lp = torch.max(log_probs_tensor)
    shifted = torch.exp(log_probs_tensor - max_lp)
    return shifted / torch.sum(shifted)


def gather_log_probs(logits, input_ids):
    """
    Gather log-probs for the shift_labels.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs_seq = torch.log_softmax(shift_logits, dim=-1)
    return log_probs_seq.gather(2, shift_labels.unsqueeze(2)).squeeze(2)


class ProbabilityCalibrator:
    """Calibrate probabilities of label choices given an input text.

    The calibrator requires two external callables:

      1. `logprob_scorer(texts: list[str] | str) -> list[float]` –
         Returns the *sum* of log-probabilities for every text in `texts`.
         If a single string is given, the function should still work.

      2. `full_logprob(text: str) -> float` – Returns the sum of
         log-probabilities for *all* tokens in `text`.  This is used for the
         baseline P(X) calculations.

    The caller can choose to provide either a *single-text* scorer (which will
    loop internally) or a *batch* scorer for efficiency.  The class itself is
    agnostic to how the scores are produced.
    """

    def __init__(
        self,
        choices: List[str],
        logprob_scorer: Callable[[List[str] | str], List[float]],
        full_logprob_fn: Callable[[str], float],
        num_trials: int = 3,
        content_free_input: str = " ",
        sample_dataset: Optional[List[str]] = None,
        alpha: float = 0.7,
        verbose: bool = False,
        batch_prompts: bool = False,
        batch_permutations: bool = True
    ):
        self.choices = choices
        self.num_trials = num_trials
        self.content_free_input = content_free_input
        self.alpha = alpha
        self.verbose = verbose
        self.logprob_scorer = logprob_scorer
        self.full_logprob_fn = full_logprob_fn
        self.batch_prompts = batch_prompts
        self.batch_permutations = batch_permutations

        # Build a manageable permutation pool without materializing N! for large N.
        self.sampled_permutations = self._build_sampled_permutations()
        if not self.sampled_permutations and self.num_trials > 0:
            logging.warning("No permutations available for calibration; num_trials will be effectively 0.")

        # Precompute log p(z|CF)
        self.log_probs_cf = self.compute_log_p_z_given_X(self.content_free_input) if self.choices else None

        # If we have a sample dataset, compute p_z across it
        if sample_dataset is not None:
            self.p_z = self.compute_p_z(sample_dataset)
        else:
            self.p_z = None

    def _build_sampled_permutations(self):
        """Build permutation list used in calibration without exploding memory."""
        if not self.choices or self.num_trials <= 0:
            return []

        n = len(self.choices)
        total_perms = math.factorial(n)
        # Safe threshold to enumerate exactly.
        max_exact_perms = 50000

        if total_perms <= max_exact_perms:
            all_perms = list(itertools.permutations(self.choices))
            if self.num_trials > len(all_perms):
                logging.warning(
                    f"num_trials ({self.num_trials}) > unique permutations ({len(all_perms)}); "
                    f"reducing num_trials to {len(all_perms)}."
                )
                self.num_trials = len(all_perms)
            if self.num_trials == len(all_perms):
                return all_perms
            chosen_idx = np.random.choice(len(all_perms), size=self.num_trials, replace=False)
            return [all_perms[i] for i in chosen_idx]

        logging.info(
            f"Choice count {n} implies {total_perms} permutations; sampling {self.num_trials} random unique permutations."
        )
        rng = np.random.default_rng(42)
        base = np.array(self.choices, dtype=object)
        seen = set()
        perms = []
        attempts = 0
        max_attempts = max(self.num_trials * 20, 100)
        while len(perms) < self.num_trials and attempts < max_attempts:
            perm = tuple(rng.permutation(base).tolist())
            if perm not in seen:
                seen.add(perm)
                perms.append(perm)
            attempts += 1
        if len(perms) < self.num_trials:
            logging.warning(
                f"Requested {self.num_trials} unique sampled permutations; got {len(perms)} after {attempts} attempts."
            )
            self.num_trials = len(perms)
        return perms

    def _get_log_prob_sums(self, question: str, choices: List[str], batch_prompts: bool = None) -> List[float]:
        """Universal helper that uses the external scorer for the given choices.
        
        This builds a formatted prompt with question and choices, then passes
        the prompt and choices separately to the scorer function to obtain the
        log probabilities for just the choice tokens.
        
        Args:
            question: The question or text to score
            choices: List of choice strings
            batch_prompts: If provided, overrides self.batch_prompts. If True, 
                          create a separate prompt for each choice; if False, 
                          create a single prompt containing all choices
                      
        Returns:
            List of log probability sums (one per choice)
        """
        # Use instance default if not explicitly provided
        if batch_prompts is None:
            batch_prompts = self.batch_prompts
            
        if batch_prompts:
            # Create a separate prompt for each choice
            prompts = [self.format_prompt(question, [choice]) for choice in choices]
        else:
            # Create a single prompt with all choices
            prompts = self.format_prompt(question, choices)
            
        return self.logprob_scorer(prompts, choices)

    # ------------------------------------------------------------------
    # MAIN p(z|X) CALCULATION
    # ------------------------------------------------------------------

    def compute_log_p_z_given_X(self, question):
        """
        Samples 'self.num_trials' permutations from self.permutations,
        for each permutation gets raw log-prob sums, does log-softmax,
        re-maps into the original order of 'self.choices', accumulates.
        Then we average in probability space, convert back to log, and return.

        Args:
            question: The question or text to evaluate

        Returns: a GPU tensor of shape (len(choices),) with log-probs in the original choice order.
        """
        if not self.choices or not self.sampled_permutations:
            return torch.zeros(len(self.choices) if self.choices else 0, dtype=torch.float)
            
        num_labels = len(self.choices)
        total_probabilities = torch.zeros(num_labels, dtype=torch.float)

        perm_iterator = tqdm(
            self.sampled_permutations,
            desc="Calibrating P(z|X)",
            unit="perm",
            leave=False,
            disable=not self.verbose
        ) if self.verbose else self.sampled_permutations

        # Process permutations
        if self.batch_permutations:
            # Flatten all permutations into a single batch for efficient processing
            all_choices = []
            all_orig_indices = []
            all_prompts = []
            all_perm_indices = []
            
            for i, current_perm in enumerate(perm_iterator):
                prompt = self.format_prompt(question, current_perm)
                for choice in current_perm:
                    all_prompts.append(prompt)
                    all_choices.append(choice)
                    orig_idx = self.choices.index(choice)
                    all_orig_indices.append(orig_idx)
                    all_perm_indices.append(i)
            
            # Get log probs for all choices
            all_log_sums = self.logprob_scorer(all_prompts, all_choices)
            
            # Convert to softmax probabilities by permutation and accumulate
            perm_results = {}
            for i, (log_sum, perm_idx, orig_idx) in enumerate(zip(all_log_sums, all_perm_indices, all_orig_indices)):
                if perm_idx not in perm_results:
                    perm_results[perm_idx] = {'log_sums': [], 'choices': [], 'orig_indices': []}
                
                perm_results[perm_idx]['log_sums'].append(log_sum)
                perm_results[perm_idx]['choices'].append(all_choices[i])
                perm_results[perm_idx]['orig_indices'].append(orig_idx)
            
            # Apply softmax and accumulate for each permutation
            for perm_idx, data in perm_results.items():
                log_sums = data['log_sums']
                orig_indices = data['orig_indices']
                
                # Apply softmax
                ls_tensor = torch.tensor(log_sums, dtype=torch.float)
                perm_probs = torch.softmax(ls_tensor, dim=-1)
                
                # Accumulate probabilities in original order
                for i, orig_idx in enumerate(orig_indices):
                    total_probabilities[orig_idx] += perm_probs[i]
        else:
            # Process each permutation separately
            for current_perm in perm_iterator:
                
                # Get log probs for current permutation
                prompt = self.format_prompt(question, current_perm)
                log_sums = self.logprob_scorer(prompt, list(current_perm))
                
                # Apply softmax
                ls_tensor = torch.tensor(log_sums, dtype=torch.float)
                perm_probs = torch.softmax(ls_tensor, dim=-1)
                
                # Accumulate probabilities in original order
                for i, choice in enumerate(current_perm):
                    orig_idx = self.choices.index(choice)
                    total_probabilities[orig_idx] += perm_probs[i]

        # Convert accumulated probabilities into an average
        avg_probabilities = total_probabilities / self.num_trials

        # Ensure numerical stability
        eps = 1e-15
        avg_probabilities = torch.clamp(avg_probabilities, min=eps)
        return torch.log(avg_probabilities)

    # ------------------------------------------------------------------
    # CALIBRATION ETC.
    # ------------------------------------------------------------------

    def calibrate_p_z_given_X(self, input_text):
        """
        p(z|X)_calibrated ∝ exp( log p(z|X) - alpha * log p(z|CF) )
        Returns CPU numpy array of shape [len(choices)].
        """
        log_probs_x = self.compute_log_p_z_given_X(input_text)
        log_diff = log_probs_x - self.alpha * self.log_probs_cf
        corrected_probs = safe_exponentiate_torch(log_diff)
        return corrected_probs.detach().cpu().numpy()

    def compute_p_z(self, dataset_texts):
        """
        p(z) = average_{X in dataset} of p(z|X) (after calibration).
        Returns a NumPy array of shape [len(choices)].
        """
        num_labels = len(self.choices)
        total_probs = torch.zeros(num_labels, device=self.device)
        
        dataset_iterator = dataset_texts
        if self.verbose:
            dataset_iterator = tqdm(dataset_texts, desc="Estimating P(z) from dataset", unit="text", leave=False, disable=not self.verbose)
            
        for X in dataset_iterator:
            cp = self.calibrate_p_z_given_X(X)  # CPU numpy
            total_probs += torch.tensor(cp, device=self.device)
        p_z_gpu = total_probs / len(dataset_texts)
        return p_z_gpu.cpu().numpy()

    def compute_p_X(self, input_text: str) -> float:
        """Returns the sum of log-probabilities for the entire string via external fn."""
        return self.full_logprob_fn(input_text)

    def compute_p_X_given_z(self, p_z_X, p_X, p_z):
        """
        Bayes' rule in probability space:
          p(X|z) = (p(z|X) * p(X)) / p(z)
        """
        return (p_z_X * p_X) / p_z

    def format_prompt(self, text, choices):
        """
        Original approach to building a prompt from question + choices.
        """
        joined_choices = "\n".join(choices)
        prompt = PREFIX_TEST_PROMPT.format(question=text) + CHOICES_PROMPT.format(choices=joined_choices)
        return prompt

    def compute_average_log_px_from_sample(self, texts_sample: list[str]) -> float:
        """
        Computes the average log P(X) over a sample of texts.
        Uses self.verbose to control a tqdm progress bar.
        """
        if not texts_sample:
            logging.warning("Cannot compute average log P(X) from an empty sample. Returning 0.0.")
            return 0.0

        log_px_values = []
        
        sample_iterator = texts_sample
        if self.verbose:
            sample_iterator = tqdm(texts_sample, desc="Calculating P(X) for baseline sample", unit="text", leave=False, disable=not self.verbose)
            
        for text in sample_iterator:
            log_px_values.append(self.compute_p_X(str(text)))
        
        return np.mean(log_px_values) if log_px_values else 0.0


def initialize_probability_calibrator(
    model_identifier: str, 
    model_type: str, 
    choices: list[str], 
    num_trials: int, 
    scorer_type: str, 
    verbose: bool = False,
    batch_prompts: bool = False,
    batch_permutations: bool = True
) -> ProbabilityCalibrator:
    """Initializes and returns the ProbabilityCalibrator instance using model wrappers.

    The calibrator uses logprob scoring functions that analyze prompt+choice text sequences,
    but isolate and score only the choice tokens, not the entire sequence. This provides
    more accurate calibration focused on the choice text itself.

    Args:
        model_identifier: Name or path of the model.
        model_type: Type of model ('hf', 'together', 'vllm').
        choices: List of choice strings for the calibrator.
        num_trials: Number of permutations trials for calibration.
        scorer_type: 'single' or 'batch' to determine which scoring function from the wrapper to use.
        verbose: Whether to enable verbose logging within the calibrator.
        batch_prompts: If True, create a separate prompt for each choice; if False use one prompt.
        batch_permutations: If True, process multiple permutations in a single batch.

    Returns:
        An initialized ProbabilityCalibrator instance.
    """
    logging.info(f"Initializing ProbabilityCalibrator with model type: {model_type}, model: {model_identifier}, scorer: {scorer_type}")
    logging.info(f"Batch settings - prompts: {batch_prompts}, permutations: {batch_permutations}")

    logprob_scorer_fn = None
    full_logprob_fn = None

    if model_type == "hf":
        logging.info(f"Loading Hugging Face model: {model_identifier} for ProbabilityCalibrator setup.")
        tokenizer_instance = AutoTokenizer.from_pretrained(model_identifier)
        model_instance = AutoModelForCausalLM.from_pretrained(
            model_identifier,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model_instance.eval()

        if tokenizer_instance.pad_token is None:
            logging.info("Tokenizer does not have a pad token. Adding '<|pad|>'.")
            tokenizer_instance.add_special_tokens({"pad_token": "<|pad|>"})
            model_instance.resize_token_embeddings(len(tokenizer_instance))
        if hasattr(model_instance, 'config'): # Should always be true for HF models
            model_instance.config.pad_token_id = tokenizer_instance.pad_token_id

        single_fn, batch_fn, full_logprob_fn = build_hf_logprob_fns(model_instance, tokenizer_instance)
        logprob_scorer_fn = batch_fn if scorer_type == "batch" else single_fn

    elif model_type == "together":
        logging.info(f"Preparing TogetherAI model ({model_identifier}) for ProbabilityCalibrator setup.")
        single_fn, batch_fn, full_logprob_fn = build_together_logprob_fns(model_identifier)
        logprob_scorer_fn = batch_fn if scorer_type == "batch" else single_fn

    elif model_type == "vllm":
        logging.info(f"Preparing vLLM model ({model_identifier}) for ProbabilityCalibrator setup.")
        # build_vllm_logprob_fns handles its own LLM and tokenizer loading.
        single_fn, batch_fn, full_logprob_fn = build_vllm_logprob_fns(model_identifier)
        logprob_scorer_fn = batch_fn if scorer_type == "batch" else single_fn
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose from 'hf', 'together', 'vllm'.")

    if logprob_scorer_fn is None or full_logprob_fn is None:
        raise RuntimeError(f"Failed to obtain scorer functions for model_type '{model_type}'.")

    prob_calibrator = ProbabilityCalibrator(
        choices=choices,
        logprob_scorer=logprob_scorer_fn,
        full_logprob_fn=full_logprob_fn,
        num_trials=num_trials,
        verbose=verbose,
        batch_prompts=batch_prompts,
        batch_permutations=batch_permutations
    )
    return prob_calibrator


if __name__ == "__main__":
    """Minimal demo showing how to use the refactored ProbabilityCalibrator with
    a HuggingFace model or vLLM.  Run, e.g.:

        python -m assess_clusters --demo_hf
        
    The demo now shows the improved choice-only probability scoring logic and
    batch processing of permutations.
    """

    import argparse
    parser = argparse.ArgumentParser(description="Demo for refactored ProbabilityCalibrator with choice-only scoring")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo_hf", action="store_true", help="Run demo with a local HuggingFace model (gpt2)")
    group.add_argument("--demo_vllm", action="store_true", help="Run demo with vLLM backend (requires GPU & model)")
    group.add_argument("--demo_together", action="store_true", help="Run demo with TogetherAI backend (API key required)")
    parser.add_argument("--verbose", action="store_true", help="Show additional debug information")
    parser.add_argument("--no_batch_permutations", action="store_true", help="Disable batch processing of permutations")
    parser.add_argument("--batch_prompts", action="store_true", help="Enable separate prompt for each choice")
    args = parser.parse_args()

    # Simple toy example ---------------------------------------------------
    question = "What animal says moo?"
    choices_demo = ["Cow", "Dog", "Cat"]
    
    print(f"Running demo for question: '{question}' with choices: {choices_demo}")
    print("This demo demonstrates the choice-only probability scoring.")
    
    batch_permutations = not args.no_batch_permutations
    batch_prompts = args.batch_prompts
    print(f"Batch settings - prompts: {batch_prompts}, permutations: {batch_permutations}")

    if args.demo_hf:
        print("\nUsing HuggingFace (gpt2) backend")
        model_name = "gpt2"
        single_fn, batch_fn, full_fn = build_hf_logprob_fns(model_name)
        scorer_fn = batch_fn  # Use batch variant for efficiency
    elif args.demo_together:
        print("\nUsing TogetherAI backend")
        model_name = "togethercomputer/llama-2-7b"
        single_fn, batch_fn, full_fn = build_together_logprob_fns(model_name)
        scorer_fn = single_fn  # Only single available effectively
    else:  # demo_vllm
        print("\nUsing vLLM backend")
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        single_fn, batch_fn, full_fn = build_vllm_logprob_fns(model_name)
        scorer_fn = batch_fn

    calibrator = ProbabilityCalibrator(
        choices=choices_demo,
        logprob_scorer=scorer_fn,
        full_logprob_fn=full_fn,
        num_trials=3,
        verbose=args.verbose,
        batch_prompts=batch_prompts,
        batch_permutations=batch_permutations
    )

    # Demonstrate how choices are formatted and scored
    if args.verbose:
        print("\nDemonstrating format_prompt and choice scoring:")
        # Single prompt with all choices
        formatted_prompt = calibrator.format_prompt(question, choices_demo)
        print(f"Formatted prompt with all choices:\n{formatted_prompt}")
        
        # Individual prompts for each choice
        print("\nIndividual prompts for each choice:")
        for choice in choices_demo:
            individual_prompt = calibrator.format_prompt(question, [choice])
            print(f"Prompt for '{choice}':\n{individual_prompt}\n")
        
        print("\nFor each choice, we calculate logprobs of *just the choice tokens* in context.")
        
        # Demonstrate the API that takes prompt and choices separately
        print("\nUsing single prompt for all choices:")
        results = scorer_fn(formatted_prompt, choices_demo)
        for choice, score in zip(choices_demo, results):
            print(f"  Choice '{choice}': {score:.4f}")
            
        # Demonstrate batch API with individual prompts
        print("\nUsing individual prompts for each choice:")
        individual_prompts = [calibrator.format_prompt(question, [choice]) for choice in choices_demo]
        batch_results = scorer_fn(individual_prompts, choices_demo)
        for choice, score in zip(choices_demo, batch_results):
            print(f"  Choice '{choice}': {score:.4f}")
            
        # Demonstrate the _get_log_prob_sums method
        print("\nUsing _get_log_prob_sums with single prompt:")
        single_prompt_results = calibrator._get_log_prob_sums(question, choices_demo, batch_prompts=False)
        for choice, score in zip(choices_demo, single_prompt_results):
            print(f"  Choice '{choice}': {score:.4f}")
            
        print("\nUsing _get_log_prob_sums with individual prompts:")
        batch_prompt_results = calibrator._get_log_prob_sums(question, choices_demo, batch_prompts=True)
        for choice, score in zip(choices_demo, batch_prompt_results):
            print(f"  Choice '{choice}': {score:.4f}")

    print("\nCalibrated probabilities:")
    start_time = time.time()
    probs = calibrator.calibrate_p_z_given_X(question)
    end_time = time.time()
    for label, p in zip(choices_demo, probs):
        print(f"  P({label}|X) = {p:.3f}")
    
    print(f"\nTime taken: {end_time - start_time:.3f} seconds")
    
    print("\nThe calibrator now isolates and scores only the choice tokens (e.g., 'Cow'),")
    print("not the entire prompt+choice sequence. This provides more accurate calibration.")
    print("Additionally, it can efficiently batch-process multiple permutations for better performance.")


"""
    python src/assess_clusters.py
        --demo_vllm \
        --verbose
"""




"""
cd assess_label_hierarchy/
import compute_calibrated_probabilities as cc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from importlib import reload

model_name = "google/gemma-2-9b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision to fit in GPU memory
    device_map="auto",          # Automatically distributes model across GPUs
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prob_cal = cc.ProbabilityCalibrator(model=model, tokenizer=tokenizer, choices=[ 'apple', 'card', 'cow'], num_trials=5, batching_mode='batch')


import pandas as pd 
df = pd.read_csv('../data/qwq-32b/chess_balanced_cluster_assignments.csv.gz', index_col=0)
label_set = df['balanced_label'].unique().tolist()
prob_cal = cc.ProbabilityCalibrator(model=model, tokenizer=tokenizer, choices=label_set, num_trials=5, batching_mode='batch')

from tqdm.auto import tqdm
probs = []
for p in tqdm(df['prompt'].head(200), total=200):
    c = prob_cal.calibrate_p_z_given_X(p)
    probs.append(c)

df['calibrated_prob'] = probs
df.to_csv('../data/qwq-32b/chess_balanced_cluster_assignments_calibrated.csv.gz', index=False)
"""
