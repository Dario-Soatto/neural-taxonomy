---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:55150
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: '"Plot Setup": This sentence introduces a key plot element involving
    a character seeking help, establishing the narrative conflict and engaging the
    audience''s interest.'
  sentences:
  - '"Contrasting Outcomes": This sentence contrasts George''s unsuccessful argument
    for the death penalty with his successful defense of Sirhan Sirhan, illustrating
    the complexities and challenges he faced in his legal career, thereby emphasizing
    his resilience and capability.'
  - '"Development Evidence": This sentence presents evidence of the storm''s intensification,
    detailing the transition from a tropical depression to Tropical Storm Danielle,
    supported by satellite imagery.'
  - '"Character Reaction": This sentence captures Banton''s immediate emotional response
    to the chaos he has caused, illustrating his fear and prompting his decision to
    flee.'
- source_sentence: '"Claim of Leadership": States Shahin''s role in leading a significant
    revolt, emphasizing his actions against the nobility and the establishment of
    a peasants'' republic.'
  sentences:
  - '"Characterization": Contrasts Shahin''s negative reputation among elites with
    his positive perception among commoners, highlighting his role as a protector
    of their interests.'
  - '"Narrative Shift": This sentence marks a turning point in the narrative, indicating
    that the initial revolt against Bashir Ahmad has evolved into a broader uprising
    against the Khazen sheikhs, highlighting the shifting allegiances and motivations
    of the peasants.'
  - '"Critical": This analysis connects the song''s lyrics to a broader commentary
    on human nature, suggesting a somber view that resonates with the song''s overall
    message.'
- source_sentence: '"Release Information": The sentence provides factual information
    about the album''s release, contextualizing its significance within the band''s
    history and contractual obligations.'
  sentences:
  - '"Structural Description": This sentence provides a detailed structural breakdown
    of the vimana''s outer wall, establishing a foundational understanding of its
    architectural design.'
  - '"Informative": Similar to the previous sentence, this one conveys factual information
    regarding the personnel involved in the album, contributing to the overall context
    without persuasion.'
  - '"Reinforcement of Theme": This sentence reinforces the theme of personal struggle
    by showing Harrison''s continued reference to the aphorism during a difficult
    period, linking it to his relationships with the other Beatles.'
- source_sentence: '"Commitment": This sentence marks Thomas''s decision to join the
    conspiracy, reflecting a shift from caution to active participation, while still
    acknowledging the hope for external support.'
  sentences:
  - '"Biographical Detail": This sentence provides biographical detail about Harrison''s
    life, specifically his residence at Friar Park, which is relevant to understanding
    the inspiration behind the song.'
  - '"Specificity": Providing specific details about the developers and architects
    involved adds depth to the argument, showcasing the intentionality behind the
    district''s development.'
  - '"Creative Inspiration": This sentence reveals the initial concept that sparked
    the episode, illustrating the imaginative process behind the narrative and engaging
    the audience''s curiosity about its development.'
- source_sentence: '"Symbolic Location": The mention of the Old Palace Yard as the
    execution site symbolizes the failure of their plot and the stark reality of their
    rebellion against the monarchy.'
  sentences:
  - '"Evidence Presentation": The sentence presents Thomas Wintour''s confession as
    a key piece of evidence, emphasizing its uniqueness and the context of its creation,
    which serves to establish its significance in understanding the Gunpowder Plot.'
  - '"Detailing": This sentence provides additional technical details about the tailplane
    and vertical stabilizer, reinforcing the thoroughness of the design modifications
    and their implications for stability.'
  - '"Development Overview": This sentence summarizes the architectural evolution
    of the district, indicating a shift in character and setting the stage for further
    discussion on its historical development.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: all nli dev trainer
      type: all-nli-dev-trainer
    metrics:
    - type: cosine_accuracy
      value: 0.7410247921943665
      name: Cosine Accuracy
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on the json dataset. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - json
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '"Symbolic Location": The mention of the Old Palace Yard as the execution site symbolizes the failure of their plot and the stark reality of their rebellion against the monarchy.',
    '"Development Overview": This sentence summarizes the architectural evolution of the district, indicating a shift in character and setting the stage for further discussion on its historical development.',
    '"Evidence Presentation": The sentence presents Thomas Wintour\'s confession as a key piece of evidence, emphasizing its uniqueness and the context of its creation, which serves to establish its significance in understanding the Gunpowder Plot.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5142, 0.4158],
#         [0.5142, 1.0000, 0.5587],
#         [0.4158, 0.5587, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet

* Dataset: `all-nli-dev-trainer`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | Value     |
|:--------------------|:----------|
| **cosine_accuracy** | **0.741** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### json

* Dataset: json
* Size: 55,150 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 23 tokens</li><li>mean: 37.06 tokens</li><li>max: 62 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 37.07 tokens</li><li>max: 62 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 37.28 tokens</li><li>max: 57 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                       | positive                                                                                                                                                                                       | negative                                                                                                                                                                                                         |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"Partnership Highlight": This sentence emphasizes Grill's entrepreneurial spirit and strategic alliances by mentioning his partnership with a seasoned director, showcasing his ambition and networking skills.</code> | <code>"Introduction": This sentence introduces Fox's involvement in the reality television show Total Divas, establishing her presence in mainstream media.</code>                             | <code>"Disappointment": This sentence conveys the futility of Thomas's efforts to gain Spanish support, highlighting the challenges faced by the conspirators and building a sense of urgency.</code>            |
  | <code>"Critique": This sentence raises a concern about character motivations, suggesting that the narrative lacks justification for certain character choices, which could affect viewer engagement.</code>                  | <code>"Praise for Lyrics": This sentence highlights the quality of Moore's lyrics, using specific descriptors to enhance the perceived value of the album's lyrical content.</code>            | <code>"Origin Story": This sentence narrates the circumstances leading to Gilligan's involvement, emphasizing his passion for the show, which serves to enhance the legitimacy of the episode's creation.</code> |
  | <code>"Narrative Transition": This sentence marks a pivotal moment in Grill's career, indicating a shift from his role in Canton back to Sweden, which sets the stage for subsequent developments in his life.</code>        | <code>"Role Assignment": This sentence illustrates the strategic role of Bayern and her sisters in simulating the Russian Baltic Fleet, showcasing their importance in naval exercises.</code> | <code>"Foreshadowing": This sentence hints at the supernatural element of Banton's shadow, creating suspense and intrigue about the consequences of this interaction.</code>                                     |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Evaluation Dataset

#### json

* Dataset: json
* Size: 6,128 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 23 tokens</li><li>mean: 37.02 tokens</li><li>max: 62 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 37.15 tokens</li><li>max: 62 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 37.12 tokens</li><li>max: 57 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                      | positive                                                                                                                                                                                                                     | negative                                                                                                                                                                                |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"Critique of Prosecution": This sentence critiques the prosecution's case against Buono, implying that their lack of evidence was a significant factor in the judicial proceedings.</code>                            | <code>"Judicial Norms": This sentence discusses the norms of judicial behavior, emphasizing that judges typically do not second-guess prosecutors, which highlights the rarity of George's actions.</code>                   | <code>"Scholarly Assertion": This sentence presents a scholarly claim regarding the perceived impact of the Roman conquest, establishing a foundation for further argumentation.</code> |
  | <code>"Acknowledgment": By acknowledging Gary Wright's contribution, the sentence recognizes the broader network of musicians that supported Harrison's work, adding depth to the collaborative aspect of the album.</code> | <code>"Establishment of Credentials": By stating Grill's journeys to China as a representative of the SOIC, this sentence builds his credibility and highlights his importance in the context of international trade.</code> | <code>"Contextualization": This sentence provides additional context by listing the sister ships of Bayern, which helps to situate her within a broader naval framework.</code>         |
  | <code>"Mythological Explanation": This sentence provides a mythological explanation for the temple's significance, linking it to the narrative of Sati and enhancing the cultural context of the worship practices.</code>  | <code>"Significance": Highlights the temple's status as a Shakti Pitha, emphasizing its importance within the context of Hindu worship.</code>                                                                               | <code>"Contextual Background": This sentence provides geographical and social context about Grill's birthplace, which is relevant for understanding his early influences.</code>        |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: epoch
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `learning_rate`: 2e-05
- `num_train_epochs`: 1
- `warmup_ratio`: 0.1
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | Validation Loss | all-nli-dev-trainer_cosine_accuracy |
|:------:|:----:|:-------------:|:---------------:|:-----------------------------------:|
| 0.0290 | 100  | 3.6295        | -               | -                                   |
| 0.0580 | 200  | 3.1769        | -               | -                                   |
| 0.0870 | 300  | 3.0402        | -               | -                                   |
| 0.1160 | 400  | 2.9987        | -               | -                                   |
| 0.1451 | 500  | 2.9825        | -               | -                                   |
| 0.1741 | 600  | 2.9793        | -               | -                                   |
| 0.2031 | 700  | 2.9739        | -               | -                                   |
| 0.2321 | 800  | 2.9368        | -               | -                                   |
| 0.2611 | 900  | 2.9127        | -               | -                                   |
| 0.2901 | 1000 | 2.877         | -               | -                                   |
| 0.3191 | 1100 | 2.8874        | -               | -                                   |
| 0.3481 | 1200 | 2.8151        | -               | -                                   |
| 0.3771 | 1300 | 2.8303        | -               | -                                   |
| 0.4062 | 1400 | 2.765         | -               | -                                   |
| 0.4352 | 1500 | 2.726         | -               | -                                   |
| 0.4642 | 1600 | 2.771         | -               | -                                   |
| 0.4932 | 1700 | 2.766         | -               | -                                   |
| 0.5222 | 1800 | 2.7085        | -               | -                                   |
| 0.5512 | 1900 | 2.6713        | -               | -                                   |
| 0.5802 | 2000 | 2.6928        | -               | -                                   |
| 0.6092 | 2100 | 2.6874        | -               | -                                   |
| 0.6382 | 2200 | 2.6531        | -               | -                                   |
| 0.6672 | 2300 | 2.6333        | -               | -                                   |
| 0.6963 | 2400 | 2.6605        | -               | -                                   |
| 0.7253 | 2500 | 2.6682        | -               | -                                   |
| 0.7543 | 2600 | 2.5598        | -               | -                                   |
| 0.7833 | 2700 | 2.5835        | -               | -                                   |
| 0.8123 | 2800 | 2.5789        | -               | -                                   |
| 0.8413 | 2900 | 2.581         | -               | -                                   |
| 0.8703 | 3000 | 2.5973        | -               | -                                   |
| 0.8993 | 3100 | 2.5836        | -               | -                                   |
| 0.9283 | 3200 | 2.5522        | -               | -                                   |
| 0.9574 | 3300 | 2.6026        | -               | -                                   |
| 0.9864 | 3400 | 2.5672        | -               | -                                   |
| 1.0    | 3447 | -             | 2.4773          | 0.7410                              |


### Framework Versions
- Python: 3.10.19
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.9.1
- Accelerate: 1.12.0
- Datasets: 4.4.2
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->