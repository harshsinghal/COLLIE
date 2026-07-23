#!/bin/bash
# COLLIE reason-vs-direct: train both 0.6B variants + generate eval predictions.
# Designed to run detached on the gpu.ai instance. Writes ALL_DONE at the end.
set -uo pipefail
cd /workspace
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false

echo "=== fix stale torch (recurring gpu.ai base-image issue) ==="
pip install --quiet --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip uninstall --quiet -y torchaudio || true
pip install --quiet --upgrade "transformers>=4.51" "trl>=0.17" datasets accelerate

echo "=== TRAIN reason ==="
COLLIE_TRAIN=/workspace/data/collie_reason_train.jsonl \
COLLIE_RUN_DIR=/workspace/collie-reason \
python /workspace/train_collie.py 2>&1

echo "=== TRAIN direct ==="
COLLIE_TRAIN=/workspace/data/collie_direct_train.jsonl \
COLLIE_RUN_DIR=/workspace/collie-direct \
python /workspace/train_collie.py 2>&1

echo "=== GEN reason preds ==="
COLLIE_MODEL_DIR=/workspace/collie-reason/final \
COLLIE_EVAL=/workspace/data/collie_reason_eval.jsonl \
COLLIE_PRED_OUT=/workspace/preds_reason.jsonl \
COLLIE_MAXNEW=420 \
python /workspace/gen_collie_preds.py 2>&1

echo "=== GEN direct preds ==="
COLLIE_MODEL_DIR=/workspace/collie-direct/final \
COLLIE_EVAL=/workspace/data/collie_direct_eval.jsonl \
COLLIE_PRED_OUT=/workspace/preds_direct.jsonl \
COLLIE_MAXNEW=160 \
python /workspace/gen_collie_preds.py 2>&1

echo "ALL_DONE"
