#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <experiment_dir>"
  echo "example: $0 experiments/wiki_biographies"
  exit 1
fi

exp_dir="$1"
pattern="${exp_dir}/wiki_biographies-initial-labeling-labeling__experiment-editorials__model_meta-llama-Meta-Llama-3.1-8B-Instruct__*.json"
merged="${exp_dir}/wiki_biographies-initial-labeling-labeling__experiment-editorials__model_meta-llama-Meta-Llama-3.1-8B-Instruct__merged.jsonl"

echo "Removing shard outputs matching:"
echo "  ${pattern}"
rm -f ${pattern}

echo "Removing merged file:"
echo "  ${merged}"
rm -f "${merged}"

echo "Done."
