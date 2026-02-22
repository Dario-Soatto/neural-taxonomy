#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <merged_step1_jsonl_path> [sample_n]"
  exit 1
fi

path="$1"
sample_n="${2:-3}"

python3 scripts/check_step1_responses.py --path "$path" --sample-nonnull "$sample_n"
