#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

python -u -m cadelac.learning.train_panda \
  -l 0 \
  -f 0 \
  -m 1 \
  -r 0 \
  -c 0 \
  2>&1 | tee logs/exo_context_current_config_train.log
