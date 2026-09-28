#!/usr/bin/env bash
# Download model weights and datasets used by the project.
#
# Everything lands in gitignored dirs under the repo root: weights/ and data/.
# Each item is independent: a failure is reported and the script moves on.
#
# Usage:
#   bash scripts/download_assets.sh            # models + small/medium datasets
#   WITH_PUBMED=1 bash scripts/download_assets.sh   # also the MedRAG PubMed corpus (large)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF="${ROOT}/.venv/bin/hf"
FAILED=()

fetch() {  # fetch <repo_type> <repo_id> <dest> [extra hf args...]
  local type="$1" repo="$2" dest="$3"; shift 3
  echo ">>> ${type} ${repo} -> ${dest}"
  if ! "${HF}" download "${repo}" --repo-type "${type}" --local-dir "${dest}" "$@"; then
    FAILED+=("${repo}")
  fi
}

# --- Models ---------------------------------------------------------------
fetch model Qwen/Qwen3-0.6B            "${ROOT}/weights/Qwen3-0.6B"      # upstream-comparable baseline
fetch model Qwen/Qwen3-8B              "${ROOT}/weights/Qwen3-8B"        # realistic serving size
fetch model Qwen/Qwen3-VL-4B-Instruct  "${ROOT}/weights/Qwen3-VL-4B-Instruct"  # P3 multimodal target

# --- RAG corpora + evaluation -----------------------------------------------
fetch dataset MedRAG/textbooks         "${ROOT}/data/medrag/textbooks"
fetch dataset MedRAG/statpearls        "${ROOT}/data/medrag/statpearls"
fetch dataset MedRAG/wikipedia         "${ROOT}/data/medrag/wikipedia"
if [[ "${WITH_PUBMED:-0}" == "1" ]]; then
  fetch dataset MedRAG/pubmed          "${ROOT}/data/medrag/pubmed"
fi
mkdir -p "${ROOT}/data/mirage"
echo ">>> MIRAGE benchmark.json"
curl -fL -o "${ROOT}/data/mirage/benchmark.json" \
  https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json || FAILED+=("MIRAGE")

# --- GRPO / QA -------------------------------------------------------------
fetch dataset qiaojin/PubMedQA              "${ROOT}/data/pubmedqa"
fetch dataset GBaker/MedQA-USMLE-4-options  "${ROOT}/data/medqa"
fetch dataset openlifescienceai/medmcqa     "${ROOT}/data/medmcqa"

# --- Medical VQA -----------------------------------------------------------
fetch dataset BoKelvin/SLAKE                "${ROOT}/data/slake"
fetch dataset flaviagiammarino/vqa-rad      "${ROOT}/data/vqa-rad"
fetch dataset flaviagiammarino/path-vqa     "${ROOT}/data/path-vqa"

# --- Serving benchmark workload ----------------------------------------------
fetch dataset anon8231489123/ShareGPT_Vicuna_unfiltered "${ROOT}/data/sharegpt" \
  --include "ShareGPT_V3_unfiltered_cleaned_split.json"

echo
if ((${#FAILED[@]})); then
  echo "FAILED (${#FAILED[@]}): ${FAILED[*]}"
  exit 1
fi
echo "all assets downloaded"; du -sh "${ROOT}/weights" "${ROOT}/data"
