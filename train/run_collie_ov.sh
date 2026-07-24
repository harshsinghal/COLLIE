#!/bin/bash
# COLLIE open-vocab round: train reason+direct, generate on in-dist AND OOD evals.
set -uo pipefail
cd /workspace
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false

echo "=== fix stale torch (recurring gpu.ai base-image issue) ==="
pip install --quiet --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip uninstall --quiet -y torchaudio || true
pip install --quiet --upgrade "transformers>=4.51" "trl>=0.17" datasets accelerate

echo "=== TRAIN reason ==="
COLLIE_TRAIN=/workspace/data/ov_reason_train.jsonl COLLIE_RUN_DIR=/workspace/ov-reason \
python /workspace/train_collie.py 2>&1

echo "=== TRAIN direct ==="
COLLIE_TRAIN=/workspace/data/ov_direct_train.jsonl COLLIE_RUN_DIR=/workspace/ov-direct \
python /workspace/train_collie.py 2>&1

for variant in reason direct; do
  for split in eval_id eval_ood; do
    echo "=== GEN $variant $split ==="
    MAXNEW=380; [ "$variant" = "direct" ] && MAXNEW=140
    COLLIE_MODEL_DIR=/workspace/ov-$variant/final \
    COLLIE_EVAL=/workspace/data/ov_${variant}_${split}.jsonl \
    COLLIE_PRED_OUT=/workspace/preds_ov_${variant}_${split}.jsonl \
    COLLIE_MAXNEW=$MAXNEW COLLIE_GBS=16 \
    python /workspace/gen_collie_preds_batched.py 2>&1
  done
done
echo "ALL_DONE"
