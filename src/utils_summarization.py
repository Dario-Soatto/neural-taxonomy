from typing import List
import pandas as pd
from prompts import (
    KEYWORD_DEFINITION_NODE_PROMPT, 
    KEYWORD_DEFINITION_NODE_PROMPT_WITH_EXAMPLES, 
    TreeNodeLabelResponse,
    # Summary judgment
    ARE_SUMMARIES_IN_TEXT_PROMPT,
    AreSummariesInTextResponse,
    make_summary_structure,
    # Summary generation
    GENERATE_SUMMARIES_WITH_DESCRIPTIONS_PROMPT,
    GENERATE_SUMMARIES_WITH_DESCRIPTIONS_AND_EXAMPLES_PROMPT,
    GENERATE_SUMMARIES_WITH_CHILD_NODES_PROMPT,
    GENERATE_SUMMARIES_WITH_CHILD_NODES_PROMPT_WITH_PRIOR_LABEL,
    GenerateSummariesResponse,
    GenerateSummariesWithLabelsAndDescriptionsResponse  
)
import numpy as np
from utils_openai_client import prompt_openai_model
from typing import List, Any, Callable, Tuple, Union
import numpy as np
from itertools import product
import networkx as nx
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from more_itertools import unique_everseen


def _get_sample_nodes(
    node: int, 
    tree, 
    num_samples_to_label: int = 10, 
    sentence_col: str = 'sentence'
):
    labeled_nodes_for_target_node = list(filter(lambda x: 'label' in tree.nodes[x], tree.successors(node)))
    if len(labeled_nodes_for_target_node) == 0:
        print(f"No labeled nodes found for node {node} in immediate successors, checking all descendants.")
        print("WARNING: This should only happen if the labeling is not done via reverse topological sort.")
        labeled_nodes_for_target_node = list(filter(lambda x: 'label' in tree.nodes[x], nx.descendants(tree, node)))
    if len(labeled_nodes_for_target_node) == 0:
        return None
    node_sizes = []
    for n in labeled_nodes_for_target_node:
        node = tree.nodes[n]
        if node.get('level') == 'leaf':
            node_sizes.append(1)
        else:
            node_sizes.append(node['subtree_size'])
    node_sizes = np.array(node_sizes)
    sample_nodes = np.random.choice(labeled_nodes_for_target_node, num_samples_to_label, p=node_sizes / node_sizes.sum())
    sample_labels = list(map(lambda x: f"{tree.nodes[x]['label']}{': ' + tree.nodes[x]['description'] if 'description' in tree.nodes[x] else ''}", sample_nodes))
    return sample_nodes, sample_labels


def _get_examples_for_one_node(
    node: int,
    example_df: pd.DataFrame,
    sentence_col: str = 'sentence',
    num_examples_per_node: int = 10,
    tree: nx.DiGraph = None,
    cluster_col: str = 'cluster'
):
    """
    Get examples for a single node.
    """
    orig_leaf_node_id = tree.nodes[node].get('orig_leaf_node_id')
    if orig_leaf_node_id is None:
        return []
    node_examples = None
    # Prefer index lookup if the DataFrame is indexed by cluster id.
    if orig_leaf_node_id in example_df.index:
        node_examples = example_df.loc[orig_leaf_node_id][sentence_col]
    # Fallback: use the cluster column if present.
    elif cluster_col in example_df.columns:
        node_examples = example_df.loc[example_df[cluster_col] == orig_leaf_node_id, sentence_col]
    else:
        return []

    if isinstance(node_examples, pd.Series):
        if len(node_examples) == 0:
            return []
        if len(node_examples) > num_examples_per_node:
            node_examples = node_examples.sample(num_examples_per_node).tolist()
        else:
            node_examples = node_examples.tolist()
    else:
        node_examples = [node_examples] if node_examples is not None else []
    return node_examples


def summarize_labels(
        node: int, 
        tree, num_samples_to_label: int = 10, 
        sentence_col: str = 'sentence',
) -> str:
    """
    Summarize a list of labels into a single, specific label and a description.
    """
    """
        Summarize a label with examples.
    """
    _, sample_labels = _get_sample_nodes(node, tree, num_samples_to_label, sentence_col)
    generate_definition_prompt = KEYWORD_DEFINITION_NODE_PROMPT.format(labels='\n'.join(sample_labels))
    # Get structured response using TreeNodeLabelResponse format
    return prompt_openai_model(
        prompt=generate_definition_prompt,
        model_name="gpt-4o-mini",
        temperature=0.1,
        response_format=TreeNodeLabelResponse
    )

def single_pass_summarize_labels_with_examples(
        node: int, 
        example_df: pd.DataFrame, 
        tree: nx.DiGraph, 
        num_samples_to_label: int = 10, 
        num_examples_per_node: int = 10, 
        sentence_col: str = 'sentence'
) -> str:
    """
        Summarize a label with examples.
    """
    sample_result = _get_sample_nodes(node, tree, num_samples_to_label, sentence_col)
    if sample_result is None:
        print(f"Warning: No labeled descendants available for node {node}. Using fallback label.")
        return TreeNodeLabelResponse(
            label=f"Node {node} - No Samples",
            description="Label generated without labeled descendants."
        )
    sample_nodes, sample_labels = sample_result

    # Collect examples for the sample nodes if available
    examples = []
    for sample_node_i in sample_nodes:
        examples.extend(_get_examples_for_one_node(
            sample_node_i,
            example_df,
            sentence_col,
            num_examples_per_node,
            tree,
            cluster_col='cluster'
        ))
        node_label = tree.nodes[sample_node_i].get('label', 'Unknown')
        examples.append(f"Label: {node_label}")
        examples.append("")  # Add a separator

    generate_definition_prompt = KEYWORD_DEFINITION_NODE_PROMPT_WITH_EXAMPLES.format(
        labels='\n'.join(sample_labels),
        examples='\n'.join(examples)
    )

    # Get structured response using TreeNodeLabelResponse format
    return prompt_openai_model(
        prompt=generate_definition_prompt,
        model_name="gpt-4o-mini",
        temperature=0.1,
        response_format=TreeNodeLabelResponse
    )


def judge_summary_matches(
    summary_datapoint_pairs: List[Tuple[str, str]],
    goal: str,
    model_name: str = 'gpt-4o-mini'
) -> List[int]:
    """
    Batched judge function that uses an LLM to determine whether each datapoint satisfies the summary.
    For now, make the simplifying assumption that we will only be scoring label

    Args:
        summaries: List of length-N strings, each a summary in natural language.
        datapoints: List of length-N datapoint strings (e.g., text).

    Returns:
        List of length-N binary values (0 or 1) indicating whether the summary matches the datapoint.
    """
    summary_pairs = []
    for summary, text in summary_datapoint_pairs:
        if isinstance(summary, dict):
            summary = summary['label']
        if isinstance(text, dict):
            text = text['label']
        prompt = (
            f'Summary: "{summary}"\n'
            f'Text: "{text}"\n'
        )
        summary_pairs.append(prompt)

    prompt = ARE_SUMMARIES_IN_TEXT_PROMPT.format(k=len(summary_pairs), summary_pairs='\n\n'.join(summary_pairs), goal=goal)
    output = prompt_openai_model(
        prompt=prompt, 
        model_name=model_name, 
        temperature=0.1, 
        response_format=make_summary_structure(len(summary_pairs), len(summary_pairs))
    )
    return output.responses 


def discretize_embedding_summaries(
    node: int,
    tree: nx.DiGraph,
    phi_tilde: np.ndarray,
    examples_df: pd.DataFrame,
    description_embeddings: np.ndarray,
    M: int = 5,
    num_leaf_samples_to_use_for_generation: int = 30,
    batch_size: int = 30,
    num_summary_candidates_to_generate: int = 10,
    num_datapoints_to_use_for_scoring: int = 300,
    # experimental variations
    use_descriptions_and_examples: bool = False,
    use_child_nodes: bool = False,
    use_labels_and_descriptions: bool = False,
    output_labels_and_descriptions: bool = False,
    goal: str = "the text is about the topic",
    verbose: bool = False,
    checking_model: str = 'gpt-4o-mini',
    labeling_model: str = 'gpt-4o-mini'
) -> List[str]:
    """
    Convert a continuous summary vector `phi_tilde` into a list of M discrete
    candidate summaries for a clustering task.

    Args:
        node: The node to discretize.
        tree: The tree structure. Used for getting children and other summaries.
        phi_tilde: Unit-norm vector of shape (d,), the continuous summary.
        examples_df: DataFrame of examples including: "label", "description" and "sentence" columns.
        description_embeddings: Array of shape (N, d), embedding for each datapoint. Embeddings are
            of the descriptions.
        M: Number of top summaries to return per slot.
        num_leaf_samples_to_use_for_generation: How many datapoints to sample for LLM prompting.
        batch_size: How many summaries to score at a time.
        num_summary_candidates_to_generate: How many summary candidates to generate per slot.
        num_datapoints_to_use_for_scoring: How many datapoints to sample for scoring.
        use_descriptions_and_examples: Whether to include both descriptions and example sentences
            in the LLM prompt. When True, the prompt will include both the description and an 
            example sentence for each sample, providing more context for summary generation.
        use_child_nodes: Whether to incorporate information from child nodes in the tree when
            generating summaries. When True, the LLM will consider the summaries of child nodes
            to maintain hierarchical consistency in the generated summaries.
        use_labels_and_descriptions: Whether to use both labels and descriptions when formatting 
            text. When True, text will be formatted as "label: description", otherwise only the
            label will be used.
        output_labels_and_descriptions: Whether to output summaries in a structured format with 
            separate label and description fields. When True, returns a list of dictionaries with
            'label' and 'description' keys; when False, returns a list of strings.
        goal: The goal of the summaries. To be passed into the LLM to help with generating t
            summaries that are more aligned with the goal.
        verbose: Whether to print progress information during processing.
        checking_model: The model to use for checking summary matches.
        labeling_model: The model to use for generating summaries.

    Returns:
        List of M symbolic summary strings that best align with `phi_tilde`.
    """
    # 1) Sample points uniformly for candidate generation
    full_scores = cosine_similarity(description_embeddings, phi_tilde.reshape(1, -1)).flatten()
    order = np.argsort(full_scores)[::-1] # get the most similar descriptions. 
    top_N_idx = np.random.choice(order[:num_leaf_samples_to_use_for_generation], size=int(num_leaf_samples_to_use_for_generation/2), replace=False)
    bottom_N_idx = np.random.choice(order[-num_leaf_samples_to_use_for_generation:], size=int(num_leaf_samples_to_use_for_generation/2), replace=False)

    # 1.1) get the datapoints.
    if use_labels_and_descriptions:
        text_descriptions = examples_df.apply(lambda row: f"{row['label']}: {row['description']}", axis=1).tolist()
    else:
        text_descriptions = examples_df['label'].tolist()
    text_examples = examples_df['sentences'].tolist()
    top_N_examples, top_N_descriptions, top_N_scores = [text_examples[i] for i in top_N_idx], [text_descriptions[i] for i in top_N_idx], full_scores[top_N_idx]
    bottom_N_examples, bottom_N_descriptions, bottom_N_scores = [text_examples[i] for i in bottom_N_idx], [text_descriptions[i] for i in bottom_N_idx], full_scores[bottom_N_idx]

    # 1.2) Get children and other summaries
    def format_summaries(node_list, use_labels_and_descriptions):
        # helper function to format a single summary
        def format_summary(node):
            if 'label' in tree.nodes[node]:
                if use_labels_and_descriptions:
                    return f"{tree.nodes[node]['label']}: {tree.nodes[node]['description']}"
                else:
                    return tree.nodes[node]['label']
        
        # if we're only summarizing one node, return the summary for that node
        if isinstance(node_list, int):
            return format_summary(node_list)
        
        # otherwise, return the summaries for all the nodes
        summaries = [format_summary(node) for node in node_list]
        return list(filter(lambda x: x is not None, summaries))
    
    prior_label = format_summaries(node, use_labels_and_descriptions) # useful if we're doing multiple iterations of labeling
    children_nodes = set(tree.successors(node))
    children_summaries = format_summaries(children_nodes, use_labels_and_descriptions)
    other_summaries = format_summaries(tree.nodes() - children_nodes - {node}, use_labels_and_descriptions)

    # 2) Generate candidate summaries via LLM prompt
    candidates = generate_candidate_summaries(
        low_scoring_examples=bottom_N_examples, 
        low_scoring_scores=bottom_N_scores, 
        low_scoring_descriptions=bottom_N_descriptions,
        high_scoring_examples=top_N_examples, 
        high_scoring_scores=top_N_scores, 
        high_scoring_descriptions=top_N_descriptions,
        num_summary_candidates_to_generate=num_summary_candidates_to_generate, 
        batch_size=batch_size,
        goal=goal,
        children_summaries=children_summaries,
        other_summaries=other_summaries,
        prior_label=prior_label,
        # experimental variations
        use_descriptions_and_examples=use_descriptions_and_examples,
        use_child_nodes=use_child_nodes,
        output_labels_and_descriptions=output_labels_and_descriptions,
        model_name=labeling_model
    )

    # Use itertools.product to create pairs of summaries and data points
    sampled_X_for_scoring = list(map(str, np.random.choice(
        text_descriptions, 
        size=min(num_datapoints_to_use_for_scoring, len(text_descriptions)), 
        replace=False
    )))
    summary_pairs_to_score = list(product(candidates, sampled_X_for_scoring))
    all_denotations = []
    to_iter = range(0, len(summary_pairs_to_score), batch_size)
    if verbose:
        to_iter = tqdm(to_iter, desc="Scoring summaries")
    for i in to_iter:
        batch_summary_pairs = summary_pairs_to_score[i: i + batch_size]
        denotations = judge_summary_matches(batch_summary_pairs, goal, model_name=checking_model)
        all_denotations.extend(denotations)

    full_df = (
        pd.DataFrame(summary_pairs_to_score, columns=['candidate', 'data_point'])
        .assign(candidate_label=lambda df: df['candidate'].str.get('label'))
        .assign(candidate_description=lambda df: df['candidate'].str.get('description'))
        .assign(denotation=all_denotations)
    )

    # Calculate correlation for each candidate
    candidate_groupby = 'candidate'
    if output_labels_and_descriptions:
        candidate_groupby = 'candidate_label'

    candidate_corr = (
        full_df
            .groupby(candidate_groupby)
            .apply(
                lambda df: np.corrcoef(df['denotation'], full_scores[:len(df)])[0, 1]
                if df['denotation'].std() >= 1e-8 and full_scores[:len(df)].std() >= 1e-8 else 0.0
            )
    )

    # 6) Select top-M by correlation
    candidate_corr.sort_values(ascending=False, inplace=True)
    top_M_summaries = candidate_corr.index.tolist()[:M]
    if output_labels_and_descriptions:
        top_M_summaries = [c for c in candidates if c['label'] in top_M_summaries]
    return top_M_summaries


def generate_candidate_summaries(
    low_scoring_examples: List[Any],
    low_scoring_scores: List[float],
    low_scoring_descriptions: List[str],
    high_scoring_examples: List[Any],
    high_scoring_scores: List[float],
    high_scoring_descriptions: List[str],
    num_summary_candidates_to_generate: int,
    batch_size: int,
    goal: str,
    children_summaries: List[str] = None,
    other_summaries: List[str] = None,
    prior_label: str = None,
    # experimental variations
    use_descriptions_and_examples: bool = False,
    use_child_nodes: bool = False,
    output_labels_and_descriptions: bool = False,
    model_name: str = 'gpt-4o-mini'
) -> List[str]:
    """
    LLM-driven summary generation with prompt selection based on experimental flags.
    """
    def format_examples(score_list, description_list, example_list, use_examples, negation=False):
        output_list = []
        if example_list is None:
            example_list = [None] * len(description_list)
        for i, (example, score, description) in enumerate(zip(example_list, score_list, description_list)):
            score = f"{score:.2f}"
            if negation:
                score = f"-{score}"
            if use_examples:
                output_list.append(f"Sample {i}. Description: {description}, Example: '{example}'. (score: {score})")
            else:
                output_list.append(f"Sample {i}. Description: {description} (score: {score})")
        return '\n'.join(output_list)

    low_scoring_example_str = format_examples(
        low_scoring_scores, 
        low_scoring_descriptions, 
        low_scoring_examples, 
        use_examples=use_descriptions_and_examples, 
        negation=True
    )
    high_scoring_example_str = format_examples(
        high_scoring_scores, 
        high_scoring_descriptions, 
        high_scoring_examples, 
        use_examples=use_descriptions_and_examples, 
        negation=False
    )

    # Select prompt template
    prompt_template = GENERATE_SUMMARIES_WITH_DESCRIPTIONS_PROMPT
    if use_descriptions_and_examples:
        prompt_template = GENERATE_SUMMARIES_WITH_DESCRIPTIONS_AND_EXAMPLES_PROMPT

    input_dict = {
        'low_scoring_input': low_scoring_example_str,
        'high_scoring_input': high_scoring_example_str,
        'goal': goal,
        'num_candidates': num_summary_candidates_to_generate
    }
    # only use child nodes if we have children summaries and other summaries
    if use_child_nodes and (len(children_summaries) > 0 or len(other_summaries) > 0): 
        input_dict['children_summaries'] = children_summaries
        input_dict['other_summaries'] = other_summaries
        if prior_label is None:
            prompt_template = GENERATE_SUMMARIES_WITH_CHILD_NODES_PROMPT
        else:
            prompt_template = GENERATE_SUMMARIES_WITH_CHILD_NODES_PROMPT_WITH_PRIOR_LABEL
            input_dict['prior_label_str'] = prior_label

    prompt = prompt_template.format(**input_dict)
    if output_labels_and_descriptions:
        response_format = GenerateSummariesWithLabelsAndDescriptionsResponse
    else:
        response_format = GenerateSummariesResponse

    all_summaries = []
    for _ in range(0, num_summary_candidates_to_generate, batch_size):
        responses = prompt_openai_model(
            prompt=prompt,
            model_name=model_name,
            temperature=0.1,
            response_format=response_format
        )
        responses = responses.dict()
        if 'summaries' in responses and isinstance(responses['summaries'], list):
            all_summaries.extend(responses['summaries'])
        else:
            all_summaries.append(responses['summary'])
    
    # remove duplicates
    if output_labels_and_descriptions:
        all_summaries = list(unique_everseen(all_summaries, key=lambda x: x['label']))
    else:
        all_summaries = list(set(all_summaries))
    return all_summaries


if __name__ == "__main__":
    import argparse
    import networkx as nx
    import pandas as pd
    import numpy as np
    import os
    from utils_trees import load_hierarchical_tree
    
    parser = argparse.ArgumentParser(description="Test utils_summarization functions")
    parser.add_argument("--experiment_dir", default="experiments/editorial", help="Directory with experiment outputs")
    parser.add_argument("--num_samples_to_label", default=5, help="Number of samples to label")
    parser.add_argument("--num_examples_per_node", default=3, help="Number of examples per node")
    parser.add_argument("--num_candidates", default=10, help="Number of candidates")
    parser.add_argument("--num_samples_for_candidate_generation", default=50, help="Number of samples for candidate generation")
    parser.add_argument("--num_samples_for_scoring", default=20, help="Number of samples for scoring")
    parser.add_argument("--test_summarize_labels", action="store_true", help="Test summarize_labels function")
    parser.add_argument("--test_single_pass_summarize_labels_with_examples", action="store_true", help="Test single_pass_summarize_labels_with_examples function")
    parser.add_argument("--test_discretize", action="store_true", help="Test discretize function")
    args = parser.parse_args()
    
    # Load data from previous steps
    examples_df_path = os.path.join(args.experiment_dir, "models/all_extracted_discourse_with_clusters.csv")
    examples_df = pd.read_csv(examples_df_path)
    print(f"Loaded examples data with {len(examples_df)} rows")
    
    # Update the tree loading section to use your existing tree file
    tree_path = os.path.join(args.experiment_dir, "models/agglomerative_clustering_outputs/hierarchical_tree.gml")
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"No tree files found")

    # Check file extension to determine loading method
    tree = load_hierarchical_tree(tree_path)
    leaf_nodes = list(filter(lambda x: tree.out_degree(x) == 0, tree.nodes))
    inner_nodes = list(filter(lambda x: tree.out_degree(x) > 0, nx.topological_sort(tree)))
    inner_node = list(reversed(inner_nodes))[0]
    print(f"Loaded tree with {len(tree.nodes)} nodes")
    
    # Load initial cluster labels
    cluster_labels_path = os.path.join(args.experiment_dir, "models/cluster_labels.csv")
    if os.path.exists(cluster_labels_path):
        print(f"Loading initial cluster labels from: {cluster_labels_path}")
        initial_labels_df = pd.read_csv(cluster_labels_path)
        
        # Add labels to leaf nodes in the tree
        for _, row in initial_labels_df.iterrows():
            node_id = int(row['node_id'])  # Assuming 'node_id' is the column name
            for n in tree.nodes():
                if tree.nodes[n].get('orig_leaf_node_id') == node_id:
                    nx.set_node_attributes(tree, {n: row['label']}, 'label')
                    if 'description' in row:
                        nx.set_node_attributes(tree, {n: row['description']}, 'description')
                    break
        print(f"Added labels to {len(initial_labels_df)} nodes")
    else:
        print(f"Warning: No cluster labels found at {cluster_labels_path}")
    
    # Load embeddings from cluster centroids
    cluster_centroids_path = os.path.join(args.experiment_dir, "models/cluster_centroids.npy")
    if os.path.exists(cluster_centroids_path):
        print(f"Loading embeddings from: {cluster_centroids_path}")
        cluster_embeddings = np.load(cluster_centroids_path)
        print(f"Loaded embeddings with shape: {cluster_embeddings.shape}")
    else:
        raise FileNotFoundError(f"No cluster centroids found at {cluster_centroids_path}")
        
    # Load initial labels
    initial_labels_path = os.path.join(args.experiment_dir, "editorial-discourse-initial-labeling-labeling__experiment-editorials__model_gpt-4o-mini__0_1635.json")
    if os.path.exists(initial_labels_path):
        print(f"Loading initial labels from: {initial_labels_path}")
        initial_labels = pd.read_json(initial_labels_path, lines=True)
        combined_texts = initial_labels.apply(lambda x: '"' + x['label'] + '": ' + x['description'], axis=1).tolist()
        cache_path = os.path.join(args.experiment_dir, "initial_label_embeddings_cache.npy")
        if os.path.exists(cache_path):
            print(f"Loading cached label embeddings from: {cache_path}")
            label_embeddings = np.load(cache_path)
        else:
            print("Generating embeddings for labels and descriptions...")
            # Load the trained sentence embedding model
            trained_embedding_model_path = os.path.join(args.experiment_dir, "models/editorial-sentence-similarity-model/trained-model")
            if os.path.exists(trained_embedding_model_path):
                print(f"Loading sentence embedding model from: {trained_embedding_model_path}")
                from sentence_transformers import SentenceTransformer
                trained_embedding_model = SentenceTransformer(trained_embedding_model_path)
                print("Model loaded successfully")
            else:
                raise FileNotFoundError(f"No sentence embedding model found at {trained_embedding_model_path}")
            label_embeddings = trained_embedding_model.encode(combined_texts, show_progress_bar=True)
            np.save(cache_path, label_embeddings)
    else:
        raise FileNotFoundError(f"No initial labels found at {initial_labels_path}")

    # Test summarize_labels function
    if args.test_summarize_labels:
        print("\nTesting summarize_labels:")
        summary_result = summarize_labels(inner_node, tree, num_samples_to_label=5)
        print(f"Label: {summary_result.label}")
        print(f"Description: {summary_result.description}")
    
    # Test single_pass_summarize_labels_with_examples
    if args.test_single_pass_summarize_labels_with_examples:
        print("\nTesting single_pass_summarize_labels_with_examples:")
        summary_with_examples = single_pass_summarize_labels_with_examples(
            inner_node, 
            examples_df, 
            tree, 
            num_samples_to_label=5,
            num_examples_per_node=3,
            sentence_col='sentences'
        )
        print(f"Label: {summary_with_examples.label}")
        print(f"Description: {summary_with_examples.description}")
    
    # Test discretize with a small sample
    if args.test_discretize:
        print("\nTesting discretize:")
        children_leaf_nodes = list(nx.descendants(tree, inner_node) & set(leaf_nodes))
        children_leaf_node_ids = list(map(lambda x: tree.nodes[x]['orig_leaf_node_id'], children_leaf_nodes))
        phi_tilde = cluster_embeddings[children_leaf_node_ids].mean(axis=0)
        phi_tilde = phi_tilde / np.linalg.norm(phi_tilde)
        
        top_summaries = discretize_embedding_summaries(
            phi_tilde=phi_tilde,
            text_descriptions=combined_texts,
            text_examples=examples_df['sentences'],
            text_embeddings=label_embeddings,
            M=3,
            num_samples_for_candidate_generation=30,
            num_samples_for_scoring=20,
            goal="argumentation strategies",
            verbose=True
        )
        
        print("Top summaries:")
        for i, summary in enumerate(top_summaries):
            print(f"{i+1}. {summary}")


"""
python src/utils_summarization.py --experiment_dir experiments/editorial

"""
