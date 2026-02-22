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

python3 - <<PY
import glob
import os

pattern = "${pattern}"
merged = "${merged}"

files = sorted(glob.glob(pattern))
if not files:
    raise SystemExit(f"No shard files found for pattern: {pattern}")

os.makedirs(os.path.dirname(merged), exist_ok=True)
with open(merged, "w") as out_f:
    for path in files:
        with open(path, "r") as in_f:
            for line in in_f:
                line = line.strip()
                if line:
                    out_f.write(line + "\\n")
print(f"Merged {len(files)} files into {merged}")
PY
