#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SBATCH_SCRIPT="${REPO_DIR}/src/sbatch/newsgroups_sanity_em_llm_only_jag.sh"

MODES=(clean merge_only split_only remove_only add_only)

for mode in "${MODES[@]}"; do
  echo "Submitting SANITY_MODE=${mode}"
  sbatch --export=ALL,SANITY_MODE="${mode}" "${SBATCH_SCRIPT}"
done
