# pip install -U git+https://github.com/nicholishen/tooldantic.git
# from pydantic import BaseModel, create_model, field_validator, ValidationError
from typing import List, Optional, Literal
from annotated_types import Len
from typing import ClassVar, Annotated
from tooldantic import ToolBaseModel as BaseModel
import tooldantic as td


########################
#
# Similarity Prompts
#
############################

GENERIC_SIMILARITY_PROMPT = """
  I will show you {k_i} pairs of labels, all describing some aspect of text.

  Are the two labels in each pair describing similar concepts?
  Think broadly about what each label is describing. Don't pay attention to the specific topic of subject-material of each label.
  Answer with "Yes" or "No".
  Return ONLY valid JSON in this exact format (use double quotes everywhere):
  {{
    "pairs": [
      {{ "pair_idx": 1, "label": "Yes" }},
      {{ "pair_idx": 2, "label": "No" }}
    ]
  }}
"""

MATHEMATICAL_REASONING_SIMILARITY_PROMPT = """
  I will show you {k_i} pairs of of labels, all applied to different mathematical reasoning steps.

  Are the two labels in each pair describing similar reasoning steps in mathematical problem solving?
  Think broadly about what each label is describing. Don't pay attention to the specific step each label is describing.
  Answer with "Yes" or "No".
  Return ONLY valid JSON in this exact format (use double quotes everywhere):
  {{
    "pairs": [
      {{ "pair_idx": 1, "label": "Yes" }},
      {{ "pair_idx": 2, "label": "No" }}
    ]
  }}

  <example>
  1. Label 1: ```"Trial and Error": This step explores different ways to solve the problem, hoping to find a combination that simplifies it. It's a process of trying different options and discarding them when they don't lead to a simpler form.,``` Label 2: ```"Exploring Alternatives": This step performs a reasoning of exploring alternative ways to break down 2700 into factors, specifically considering the factorization 2700 = 9 * 300, and then evaluating if this leads to a different simplified form of the cube root.'```
  2. Label 1: ```"Quantitative Comparison": This step compares the cost after the interchange to the original cost, calculates the difference, and then expresses that difference as a percentage of the original cost to determine if it matches the 50% increase stated in the problem. ``` Label 2: ```"Evaluation": This step evaluates the outcome of a hypothetical scenario by comparing the result to the original, and determines that the change is not as required, thus eliminating a possible solution.'```
  3. Label 1: ```"Elimination": This step considers the centroid as a potential solution but quickly dismisses it.``` Label 2: ```"Meta-reflection": This step involves reflecting on the reasoning process itself, recognizing a past mistake, and understanding how the correct process leads to the correct solution.```
  4. Label 1: ```"Factoring": This step applies the factoring technique to simplify the equation, breaking it down into two expressions multiplied together. The goal is to identify potential solutions by recognizing that if the product of two factors is zero, at least one of the factors must be zero.``` Label 2: ```"Identifying": This step involves identifying the coefficients of the polynomial to be divided, which is a crucial step in setting up the synthetic division process. The reasoning here is focused on extracting and labeling the relevant information from the polynomial expression.```
  Answers:
    {{
      "pairs": [
        {{ "pair_idx": 1, "label": "Yes" }},
        {{ "pair_idx": 2, "label": "Yes" }},
        {{ "pair_idx": 3, "label": "No" }},
        {{ "pair_idx": 4, "label": "No" }}
      ]
    }}
</example>

  Now it's your turn:
  {samples_str}

  Answers:
"""

EMOTIONS_SIMILARITY_PROMPT = """
I will show you {k_i} pairs of of labels, all applied to different emotions.

Are the two labels in each pair describing related emotions with related impacts on the reader?
Think broadly about what each label is describing. Don't pay attention to the specific topic of subject-material of each label.
Answer with "Yes" or "No".
Return ONLY valid JSON in this exact format (use double quotes everywhere):
{{
  "pairs": [
    {{ "pair_idx": 1, "label": "Yes" }},
    {{ "pair_idx": 2, "label": "No" }}
  ]
}}

<example>
1. Label 1: "Satisfaction": The commenter feels content with their preparations. Label 2: "Polite": The commenter is trying to engage in a respectful manner.
2. Label 1: "Disapproval": The commenter disapproves of stealing. Label 2: "Frustration": The commenter is frustrated with the judgment of others without evidence.
3. Label 1: "Desire": The commenter expresses a strong craving or longing for the cake. Label 2: "Longing": Wishing for validation or recognition.
4. Label 1: "Anger": The commenter expresses strong negative feelings towards the subject. Label 2: "Humor": The commenter uses humor to relate to the situation.
  Answers:
    {{
      "pairs": [
        {{ "pair_idx": 1, "label": "Yes" }},
        {{ "pair_idx": 2, "label": "No" }},
        {{ "pair_idx": 3, "label": "Yes" }},
        {{ "pair_idx": 4, "label": "No" }}
      ]
    }}
</example>
Now it's your turn:
{samples_str}

Answers:
"""

HATE_SPEECH_SIMILARITY_PROMPT = """
I will show you {k_i} pairs of of labels, all applied to different reactions to hate speech.

Are the two labels in each pair describing similar reactions?
Think broadly about what each label is describing. Don't pay attention to the specific topic of subject-material of each label.
Answer with "Yes" or "No".
Return ONLY valid JSON in this exact format (use double quotes everywhere):
{{
  "pairs": [
    {{ "pair_idx": 1, "label": "Yes" }},
    {{ "pair_idx": 2, "label": "No" }}
  ]
}}

<example>
1. Label 1: "Hostile": Aggressively dismissing the original comment. Label 2: "Counterargument": Challenging the original narrative with additional information.
2. Label 1: "Skeptical": The response expresses doubt about the intentions of others. Label 2:   "Cynical": Expressing a cynical view on someone\'s experience.
3. Label 1: "Confrontational": Challenges the original comment\'s perspective directly. Label 2: "Hostile": Aggressively dismissing the original comment.
4. Label 1: "Critical": Judging the actions of the individual in the scenario. Label 2: "Concerned": Expressing alarm about the situation.
  Answers:
    {{
      "pairs": [
        {{ "pair_idx": 1, "label": "No" }},
        {{ "pair_idx": 2, "label": "Yes" }},
        {{ "pair_idx": 3, "label": "Yes" }},
        {{ "pair_idx": 4, "label": "No" }}
      ]
    }}
</example>
Now it's your turn:
{samples_str}

Answers:
"""

NEWS_DISCOURSE_SIMILARITY_PROMPT = """
I will show you {k_i} pairs of of labels, all applied to different sentences in a news article, categorizing the discourse function of that sentence in the overall article.

Are the two labels in each pair describing similar discourse purposes?
Think broadly about what each label is describing. Don't pay attention to the specific topic of subject-material of each label.
Answer with "Yes" or "No".
Return ONLY valid JSON in this exact format (use double quotes everywhere):
{{
  "pairs": [
    {{ "pair_idx": 1, "label": "Yes" }},
    {{ "pair_idx": 2, "label": "No" }}
  ]
}}

<example>
1. Label 1: "Additional Incident Report": Describes another conflict resulting in casualties among Taliban fighters. Label 2: "Supporting Detail": Offers a visual detail related to peacekeeping operations, giving more information about the environment in which these events occurred.
2. Label 1: "Contextual Background": Provides context about the U.S. tariffs and escalation in trade tensions. Label 2: "Main Event": Introduces the main event of the article, which is China\'s imposition of a 10\% tarrif, highlighting the escalation of the trade dispute.
3. Label 1: "Summary of Market Trends": Summarizes the overall performance of CBOT agricultural futures, linking it to weather and trade talks. Label 2: "Supply Forecast": This sentence discusses potential increases in non-OPEC oil supply, which is relevant to the overall market dynamics.
4. Label 1: "Contextual Insight": Links rapid intensification of storms to climate change through scientific studies, highlighting its increasing seriousness. Label 2: "Human Interest": Provides a quote from an individual explaining their actions, emotions and experiences during the climate-change induced storm.
  Answers:
    {{
      "pairs": [
        {{ "pair_idx": 1, "label": "Yes" }},
        {{ "pair_idx": 2, "label": "No" }},
        {{ "pair_idx": 3, "label": "Yes" }},
        {{ "pair_idx": 4, "label": "No" }}
      ]
    }}
</example>
Now it's your turn:
{samples_str}

Answers:
"""

EDITORIAL_SIMILARITY_PROMPT = """
I will show you {k_i} pairs of of labels, all applied to different sentences in an editorial or opinion article.

Are the two labels in each pair describing similar argumentation strategies or are having similar effects on the reader?
Think broadly about what each label is describing. Don't pay attention to the specific topic of subject-material of each label.
Answer with "Yes" or "No".
Return ONLY valid JSON in this exact format (use double quotes everywhere):
{{
  "pairs": [
    {{ "pair_idx": 1, "label": "Yes" }},
    {{ "pair_idx": 2, "label": "No" }}
  ]
}}

<example>
1. Label 1: "Implied Urgency": This sentence conveys a sense of urgency and hope, suggesting that the right voice could fulfill a significant need in the political landscape, thereby motivating leaders to act. Label 2: "Motivational Reminder": This sentence serves as a motivational cue for the president\'s team, reinforcing the urgency and importance of their efforts in the final phase of his presidency.
2. Label 1: "Causal Argument": This sentence argues that the restrictive policies create resentment among the Javakheti Armenians, suggesting that such animosity could be leveraged by Russia for its own strategic interests. Label 2: "Historical Context": This sentence provides historical context regarding NATO and EU expansion, framing it as a direct threat to Russia, thereby reinforcing the narrative of encirclement and vulnerability.
3. Label 1: "Statistical Evidence": The reference to the absence of convictions since 1969 serves as a powerful statistic that underscores the ongoing issue of police impunity, reinforcing the argument for the need for reform. Label 2: "Trend Evidence": Similar to the previous sentence, this presents another trend in gun ownership, specifically regarding background checks, reinforcing the argument that more people are seeking to legally own guns.
4. Label 1: "Minimization": The author downplays the severity of the actions of the Paris assassins, framing them as ordinary individuals rather than terrorists, which serves to diminish the perceived threat. Label 2: "Critique": Critically points out the limited nature of the official\'s engagement, suggesting a lack of genuine dialogue beyond a narrow focus on counter-terrorism.
  Answers:
    {{
      "pairs": [
        {{ "pair_idx": 1, "label": "Yes" }},
        {{ "pair_idx": 2, "label": "No" }},
        {{ "pair_idx": 3, "label": "Yes" }},
        {{ "pair_idx": 4, "label": "No" }}
      ]
    }}
</example>
Now it's your turn:
{samples_str}

Answers:
"""

BIOGRAPHIES_SIMILARITY_PROMPT = """
I will show you {k_i} pairs of labels, all applied to different sentences in biographical articles.

Are the two labels in each pair describing similar biographical roles or functions?
Think broadly about what each label is describing. Don't pay attention to the specific topic or subject matter of each label.
Answer with "Yes" or "No".
Return ONLY valid JSON in this exact format (use double quotes everywhere):
{{
  "pairs": [
    {{ "pair_idx": 1, "label": "Yes" }},
    {{ "pair_idx": 2, "label": "No" }}
  ]
}}

<example>
1. Label 1: "Early Life" - describes childhood or family background. Label 2: "Childhood Context" - provides early background details.
2. Label 1: "Career Milestone" - describes a notable professional achievement. Label 2: "Personal Life" - describes relationships or family matters.
Answers:
  {{
    "pairs": [
      {{ "pair_idx": 1, "label": "Yes" }},
      {{ "pair_idx": 2, "label": "No" }}
    ]
  }}
</example>
Now it's your turn:
{samples_str}

Answers:
"""

########################
#
# Initial Labeling Prompts
#
############################
EDITORIAL_INITIAL_LABELING_PROMPT = """
  You will be given an editorial or opinion article as well as a sentence from that article. 

  For each sentence, give a generic keyword label to explain the role it plays in the author's overall argumentation strategy. 
  This means identifying the function of the sentence in the context of the broader persuasive structure — what purpose it serves in advancing, supporting, framing, or organizing the argument.
  Avoid summarizing the content or repeating the surface meaning. Your goal is to interpret each unit's intentional role in shaping the argumentative flow of the article.
  Think in terms of strategic purpose: how the piece is structured to influence belief, agreement, or perception over time.
  Focus on describing what each unit is doing rhetorically, Return in a JSON in the format: 
  
  {{
    "label": LABEL,
    "description": DESCRIPTION
  }}

  <article>
  {article}
  </article>

  <sentence>
  {sentence}
  </sentence>

  Your response:
"""

BIOGRAPHIES_INITIAL_LABELING_PROMPT = """
  You will be given a biographical article excerpt as well as a sentence from that article.

  For each sentence, give a concise, generic label describing its role in a biography narrative or structure.
  Focus on the sentence's function (e.g., early life context, education, career milestone, contributions, personal life, legacy, or section header),
  not the surface topic. Avoid summarizing the content.
  Return a JSON in the format:

  {{
    "label": LABEL,
    "description": DESCRIPTION
  }}

  If the line is a section header, list marker, or metadata, label it as "Section Header" or "Metadata".

  <article>
  {article}
  </article>

  <sentence>
  {sentence}
  </sentence>

  Your response:
"""

MULTI_SENTENCE_EDITORIAL_LABELING_PROMPT = """
You will be given an editorial or opinion article as well as {k} sentences from that article. 

For each sentence, give a keyword label to explain the author's argumentation strategy in that sentence. 
This means identifying the persuasive function of the sentence in the context of the broader argumentative structure and, if appropriate, identifying the kind of evidence being used.
Avoid summarizing the content or repeating the surface meaning. Your goal is to interpret each unit's argumentative role.
Think of how the sentence tries to influence the reader's belief, agreement, or perception, focus on describing what each unit is doing rhetorically.

Return a list of {k} labels in the format: 
[{{
  'sentence_idx': SENTENCE_IDX,
  'label': LABEL,
  'description': DESCRIPTION
}},
{{
    'sentence_idx': SENTENCE_IDX,
    'label': LABEL,
    'description': DESCRIPTION
}},
...
]

<article>
{article}
</article>

<sentences>
{sentences}
</sentences>

Your response:
"""


NEWS_CLUSTER_LABELING_PROMPT = '''You are a helpful assistant. I will give you a list of news headlines and summaries. You will summarize them and 
return a single, specific topic label and a description in the following forward: "Label": Description. 
Please condense them into a single, specific label. Be precise and concise. Ignore labels that are too generic.
Please return just one 2-3 word label and one description.

Here are some examples of how I want my outputs:
<examples>
output:
"Space Industry": These articles cover missions planned either by government or private companies into space.

output:
"Heart Health": The step covers medical advances impacting cardiovascular systems.
</examples>

Now it's your turn. Here are the article headlines and summaries:
<articles>
{articles_and_summaries}
</articles>

output:
'''


class EditorialLabelingResponse(BaseModel):
    """Response format for single sentence editorial labeling."""
    label: str
    description: str


class MultiSentenceLabelingItem(BaseModel):
    """Individual sentence labeling in multi-sentence response."""
    sentence_idx: int
    label: str
    description: str


class MultiSentenceLabelingResponse(BaseModel):
    """Response format for multi-sentence editorial labeling."""
    sentences: List[MultiSentenceLabelingItem]
    justifications: List[str]


SINGLE_COMMENT_EMOTION_LABELING_PROMPT = """
  You will be given a comment extracted from Reddit. Please label all the emotions that the commenter is likely feeling and/or expressing.
  Think broadly about what the commenter is feeling and/or expressing.
  Think about the commenter's relationship to the topic of the comment.
  Return a list of labels in the format: 
  {{
    'comment_labels': [
      {{
        'label': LABEL,
        'description': DESCRIPTION
      }},
      ...
    ]
  }}

  <comment>
{comment}
  </comment>

  Your response:
"""

# Return a list of {k} labels in the format: 
# [{{
#   'comment_idx': COMMENT_IDX,
#   'comment_labels': [
#     {{
#       'label': LABEL,
#       'description': DESCRIPTION
#     }},
#     ...
#   ]
# }},
# ...
# ]
MULTI_SENTENCE_EMOTION_LABELING_PROMPT = """
  You will be given {k} comments extracted from Reddit. 
  Think broadly about what the commenter is feeling and/or expressing.
  Think about the commenter's relationship to the topic of the comment.
  For each comment, please label the emotions that the commenter is expressing.

  <comments>
  {comments}
  </comments>

  Your response:
"""


class SingleCommentLabel(BaseModel):
    """Individual comment labeling in single comment response."""
    label: str
    description: str


class SingleCommentLabelingResponse(BaseModel):
    """Response format for single comment emotion labeling."""
    comment_labels: List[SingleCommentLabel]
    justification: str


class MultiCommentLabelingItem(BaseModel):
    """Individual comment labeling in multi-comment response."""
    comment_idx: int
    comment_labels: List[SingleCommentLabel]
    justifications: List[str]


class MultiCommentLabelingResponse(BaseModel):
    """Response format for multi-comment emotion labeling."""
    comments: List[MultiCommentLabelingItem]
    description: str


class SimilarityPrompt(BaseModel):
    """Prompt for similarity task."""
    prompt: str
    label_1: str
    label_2: str
    description_1: str
    description_2: str


class SimilarityResponse(BaseModel):
    """Response format for similarity task."""
    pair_idx: int
    label: Literal['Yes', 'No']
    description: Optional[str] = None


class MultiSimilarityResponse(BaseModel):
    """Response format for multi-sentence similarity labeling."""
    pairs: List[SimilarityResponse]
    descriptions: Optional[str] = None


HATE_SPEECH_LABELING_PROMPT = """
You are a trained specialist. 
You are tasked with analyzing {k} responses to online messages. Both the messages and the responses may contain hate speech. 
For each response:

1. Examine the Content: Carefully read the response to understand its context, tone, and intent.
2. Identify Strategy: Note how the response relates to the original message.
3. Generate a Label: Assign one or more descriptive labels that capture the spirit and strategy of the response, NOT it's topic.
Be concise (1-3 words). 

<messages_and_responses>
{messages_and_responses}
</messages_and_responses>

Your response:
"""


class HateSpeechLabel(BaseModel):
    """Individual hate speech labeling in single comment response."""
    label: str
    description: str


class HateSpeechLabelingResponse(BaseModel):
    """Response format for hate speech labeling."""
    labels: List[HateSpeechLabel]
    justifications: List[str]


class MultiHateSpeechLabelingItem(BaseModel):
    """Individual hate speech labeling in multi-comment response."""
    comment_idx: int
    labels: List[HateSpeechLabel]
    justifications: str


class MultiHateSpeechLabelingResponse(BaseModel):
    """Response format for multi-hate speech labeling."""
    comments: List[MultiHateSpeechLabelingItem]


NEWS_DISCOURSE_LABELING_PROMPT = """
You are an annotator tasked with analyzing sentences in a news article. 
I will give you the full news article and ask you to write labels for {k} sentences.
For each sentence, decide what role it plays in the article's overall narrative and structure. 
Your goal is to group sentences into distinct categories based on how they contribute to understanding the main story. 
Your labels should reflect your reasoning about the function and relevance of each sentence to the article's core narrative.

<article>
{article}
</article>

Please write labels for the following sentences:
<sentence_indices>
{sentence_indices}
</sentence_indices>

Your response:
"""


class NewsDiscourseLabel(BaseModel):
    """Individual news discourse labeling in single sentence response."""
    label: str
    description: str


class NewsDiscourseLabelingResponse(BaseModel):
    """Response format for news discourse labeling."""
    sentences: List[NewsDiscourseLabel]
    justifications: str


class MultiNewsDiscourseLabelingItem(BaseModel):
    """Individual news discourse labeling in multi-sentence response."""
    sentence_idx: int
    label: str
    description: str


class MultiNewsDiscourseLabelingResponse(BaseModel):
    """Response format for multi-news discourse labeling."""
    sentences: List[MultiNewsDiscourseLabelingItem]
    justifications: List[str]


# Tree Generation Prompts
# These prompts are used in create_trees.py for generating hierarchical tree structures
# from clusters of labeled data. They help summarize and label nodes in the tree.
KEYWORD_DEFINITION_NODE_PROMPT = """
You are a helpful assistant. I will give you a list of labels I wrote. You will summarize them and 
return a single, specific label and a description in the following forward: "Label": Description. 
Please condense them into a single, specific label. Be precise and concise. Ignore labels that are too generic.
Please return just one 2-3 word label and one description.

Here are some examples of how I want my outputs:
<examples>
output:
"Initial Condition": This step sets the starting point for a problem or calculation, establishing the initial state or value of a variable.

output:
"Reorganization": The step involves reorganizing or transforming data structures, such as lists or decks, to accurately reflect changes or updates.
</examples>

Now it's your turn. Here are the labels:
<labels>
{labels}
</labels>

output:
""" 

# Please focus specifically on methods of persuasion represented by the labels -- be specific!
KEYWORD_DEFINITION_NODE_PROMPT_WITH_EXAMPLES = """
You are a helpful assistant. I will give you a list of labels for classes of data points that I wrote, along with their definitions. 
Your task is to summarize the labels into a single 1-2 word label and description, capturing the meaning of these labels.
I will also give you examples of data points in each of classes.


Here are the labels:
<labels>
{labels}
</labels>

Here are the data points:
<data points>
{examples}
</data points>

Please summarize the labels into a single 1-2 word label and a description. 
Do not return a label that is too generic. Be precise, clear, and concise. 
Use the examples to help you better understand the labels and avoid returning a label that is too generic.

Your response:
""" 


SINGLE_SUMMARY_NODE_PROMPT = """
You are a helpful assistant. I will give you a label and a definition that I wrote.

Here is the label and the definition:

<label and definition>
{label_and_definition}
</label and definition>

Please summarize it in a single, specific label that captures the meaning of the definition. 
Be precise, clear, and concise. Do not be too generic. Return just the label.

Your response:
"""

class TreeNodeLabelResponse(BaseModel):
    """Response format for single label."""
    label: str
    description: str


ARE_SUMMARIES_IN_TEXT_PROMPT = """I will show you {k} pairs of summaries and text. 
The goal of the summaries are to capture this aspect of the text: {goal}. 
For each pair, please respond with 1 or 0 to indicate whether the summary captures the meaning of the text.
Respond with 0 if the summary is too broad or too vague to NOT be satisfied by the text.
Format your responses as a list the length of the number of summaries, where the ith entry is 1 if the ith summary is satisfied and 0 otherwise.

Here are the summary pairs:
<summary_pairs>
{summary_pairs}
</summary_pairs>

Your response:
"""

class AreSummariesInTextResponse(BaseModel):
    """Response format for are summaries in text task."""
    responses: List[Literal[1, 0]]
    
    # min_items: ClassVar[int] = 1
    # max_items: ClassVar[int] = 1
    # @field_validator('responses')
    # @classmethod
    # def validate_length(cls, v):
    #     if not (cls.min_items <= len(v) <= cls.max_items):
    #         raise ValueError(f'List must have between {cls.min_items} and {cls.max_items} items')
    #     return v

from pydantic import create_model
def make_summary_structure(min_len: int, max_len: int):
    return create_model(
        'AreSummariesInTextResponse',
        responses=(Annotated[list[int], Len(min_length=min_len, max_length=max_len)], ...),
        justifications=str,
    )


GENERATE_SUMMARIES_WITH_DESCRIPTIONS_PROMPT = """
You are a helpful summarizer assistant.
Here is a corpus of descriptions I wrote for datapoints, sorted from the lowest to the highest score.

<high_scoring_descriptions>
{high_scoring_input}
</high_scoring_descriptions>

<low_scoring_descriptions>
{low_scoring_input}
</low_scoring_descriptions>

Please propose {num_candidates} different short 2-4 word summaries that summarize the descriptions with high scores as they relate to {goal}, and do NOT summarize the descriptions with lower scores.
The summaries should be concise and specific and focus on how the descriptions relate to {goal}, not their specific topics.
Each candidate summary should try to capture the meaning of ALL the descriptions with high scores. 
Do not generate vague summaries that are too generic, avoid weasle words like "Techniques", "Strategies", "Approaches", etc. Be specific about the {goal} of the text samples.
Your response is:
"""

GENERATE_SUMMARIES_WITH_DESCRIPTIONS_AND_EXAMPLES_PROMPT = """
Here is a corpus of text samples and descriptions I wrote for them, sorted from the lowest to the highest score.

<high_scoring_descriptions_and_examples>
{high_scoring_input}
</high_scoring_descriptions_and_examples>

<low_scoring_descriptions_and_examples>
{low_scoring_input}
</low_scoring_descriptions_and_examples>

Please propose {num_candidates} different short 2-4 word summaries that summarize the text samples with high scores as they relate to {goal}, and do NOT summarize the text samples with lower scores.
The summaries should be concise and specific and focus on how the text samples relate to {goal}, not their specific topics.
Each candidate summary should try to capture the meaning of ALL the text samples with high scores. 
Do not generate vague summaries that are too generic, avoid weasle words like "Techniques", "Strategies", "Approaches", etc. Be specific about the {goal} of the text samples.
Your response is:
"""

GENERATE_SUMMARIES_WITH_CHILD_NODES_PROMPT = """
I am summarizing an inner node in a tree. Here is a corpus of text samples, sorted from the lowest to the highest score by their similarity to the inner node.

<high_scoring_examples>
{high_scoring_input}
</high_scoring_examples>

<low_scoring_examples>
{low_scoring_input}
</low_scoring_examples>

Here are the children summaries for this inner node:
<children_summaries>
{children_summaries}
</children_summaries>

Here are summaries for nodes that are not children of this inner node (if you don't see any, it's okay, it just means we haven't generated any other summaries yet):
<other_summaries>
{other_summaries}
</other_summaries>

Please propose {num_candidates} different short 1-3 word summaries. I will select the best summary from your list.
The summaries should consolidate the information from the children summaries and be DIFFERENT from the other summaries.
The summaries should summarize the text samples with high scores as they relate to {goal}, and do NOT summarize the text samples with lower scores.
The summaries should be concise and specific and focus on how the text samples relate to {goal}, not their specific topics.
Each candidate summary should try to capture the meaning of ALL the text samples with high scores. 
Do not generate vague summaries that are too generic, avoid weasle words like "Techniques", "Strategies", "Approaches", etc. Be specific about the {goal} of the text samples.
Return just the summaries, no other text. Your response is:
"""


GENERATE_SUMMARIES_WITH_CHILD_NODES_PROMPT_WITH_PRIOR_LABEL = """
I am summarizing an inner node in a tree. Here is a corpus of text samples, sorted from the lowest to the highest score by their similarity to the inner node.

<high_scoring_examples>
{high_scoring_input}
</high_scoring_examples>

<low_scoring_examples>
{low_scoring_input}
</low_scoring_examples>

Here are the children summaries for this inner node:
<children_summaries>
{children_summaries}
</children_summaries>

Here are summaries for nodes that are not children of this inner node (if you don't see any, it's okay, it just means we haven't generated any other summaries yet):
<other_summaries>
{other_summaries}
</other_summaries>

Here is an existing summary for this inner node. 
I made this summary before, but I want you to improve it -- make it more specific with less weasel words. 
This label may have been based off a slightly different set of text samples than the ones I'm showing you now so please try to incorporate both sets of information.
<prior_label>
{prior_label_str}
</prior_label>

Please propose {num_candidates} different short 1-3 word summaries. I will select the best summary from your list.
The summaries should consolidate the information from the children summaries and be DIFFERENT from the other summaries.
The summaries should summarize the text samples with high scores as they relate to {goal}, and do NOT summarize the text samples with lower scores.
The summaries should be concise and specific and focus on how the text samples relate to {goal}, not their specific topics.
Each candidate summary should try to capture the meaning of ALL the text samples with high scores. 
Do not generate vague summaries that are too generic, avoid weasle words like "Techniques", "Strategies", "Approaches", etc. Be specific about the {goal} of the text samples.
Return just the summaries, no other text. Your response is:
"""

# For each summary, return a label and a description of the label.
class GenerateSummariesResponse(BaseModel):
    """Response format for generate summaries task."""
    summaries: List[str]
    justifications: List[str]

class GenerateSummariesWithLabelsAndDescriptionsResponse(BaseModel):
    """Response format for generate summaries task."""
    summaries: List[TreeNodeLabelResponse]
    justifications: List[str]
