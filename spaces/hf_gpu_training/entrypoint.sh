#!/usr/bin/env bash
set -euo pipefail
cd /app

# Space "Variables": set TRAIN_ARGS, e.g.
#   --judge-schedule --model distilgpt2 --out-json logs/training_last.json
# or 8B:
#   --judge-schedule --load-in-4bit --model unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit --out-json logs/training_last_8b.json
IFS=' ' read -r -a ARGS <<< "${TRAIN_ARGS:---judge-preset --model distilgpt2}"
python scripts/train_reinforce_twin.py "${ARGS[@]}"

python scripts/upload_hf_space_artifacts.py

# Exit so the Space stops and GPU billing stops (do not sleep forever on a GPU).
exit 0
