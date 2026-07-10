#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

python -u -m cadelac.learning.train_panda \
  -l 1 \
  -f 0 \
  -m 0 \
  -r 0 \
  -c 0 \
  2>&1 | tee logs/exo_context_current_config_eval.log
