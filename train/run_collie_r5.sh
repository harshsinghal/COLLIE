#!/bin/bash
# COLLIE round 5: {reason,direct} x {0.6B,1.7B}, three eval axes each.
# Models push to HF hub at end of each training (survives instance loss).
# Requires HF_TOKEN in env (written by the launcher).
set -uo pipefail
cd /workspace
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false

echo "=== fix stale torch (recurring gpu.ai base-image issue) ==="
pip install --quiet --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip uninstall --quiet -y torchaudio || true
pip install --quiet --upgrade "transformers>=4.51" "trl>=0.17" datasets accelerate

for size in 0.6B 1.7B; do
  for variant in reason direct; do
    echo "=== TRAIN $variant $size ==="
    COLLIE_BASE="Qwen/Qwen3-$size" \
    COLLIE_TRAIN=/workspace/data/ov_${variant}_train.jsonl \
    COLLIE_RUN_DIR=/workspace/r5-$variant-$size \
    COLLIE_HUB_ID="Harsh/collie-r5-$variant-${size,,}" \
    COLLIE_BS=$([ "$size" = "1.7B" ] && echo 4 || echo 8) \
    COLLIE_ACCUM=$([ "$size" = "1.7B" ] && echo 8 || echo 4) \
    python /workspace/train_collie.py 2>&1

    for split in eval_id eval_ood eval_anchor; do
      echo "=== GEN $variant $size $split ==="
      MAXNEW=380; [ "$variant" = "direct" ] && MAXNEW=140
      GBS=16; [ "$size" = "1.7B" ] && GBS=8
      COLLIE_MODEL_DIR=/workspace/r5-$variant-$size/final \
      COLLIE_EVAL=/workspace/data/ov_${variant}_${split}.jsonl \
      COLLIE_PRED_OUT=/workspace/preds_r5_${variant}_${size}_${split}.jsonl \
      COLLIE_MAXNEW=$MAXNEW COLLIE_GBS=$GBS \
      python /workspace/gen_collie_preds_batched.py 2>&1
    done
    rm -rf /workspace/r5-$variant-$size/checkpoint-* 2>/dev/null || true
  done
done
echo "ALL_DONE"
