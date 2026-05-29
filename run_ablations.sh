#!/usr/bin/env bash
# run_ablations.sh — Ablation orchestration script
# =================================================
# Runs three groups of controlled ablation experiments:
#
#   (A) Branch-removal ablation
#       Disables one multi-scale tap at a time (low / mid / high)
#       to quantify each branch's contribution.
#
#   (B) Distillation-loss decomposition
#       Isolates the KL soft-label term and the feature-MSE term
#       to measure their individual and joint contributions.
#
#   (C) Hyperparameter sensitivity (OFAT)
#       One-factor-at-a-time analysis varying α, β and T
#       independently from the default (0.5, 0.2, 5).
#
# All variants reuse the teacher checkpoint from the main run
# and train only the student_distilled model (MSFFN+KD).
#
# Usage:
#   bash run_ablations.sh A      # branch-removal only
#   bash run_ablations.sh B      # loss-decomposition only
#   bash run_ablations.sh C      # hyperparameter sensitivity only
#   bash run_ablations.sh AB     # A + B (recommended minimum)
#   bash run_ablations.sh ALL    # all three groups

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-AB}"
EPOCHS_PER_RUN="${EPOCHS_PER_RUN:-30}"
BASE_SEED="${BASE_SEED:-42}"

# ── Locate teacher checkpoint ────────────────────────────
TEACHER_CKPT=""
for p in "checkpoints/teacher_best.weights.h5" \
         "checkpoints/seed42/teacher_best.weights.h5"; do
  if [ -f "$p" ]; then TEACHER_CKPT="$p"; break; fi
done
if [ -z "$TEACHER_CKPT" ]; then
  echo "Error: no teacher checkpoint found in checkpoints/."
  echo "Please run the full experiment first to train the teacher."
  exit 1
fi

BASE_METRICS=""
for p in "results/metrics.json" "results/seed42/metrics.json"; do
  if [ -f "$p" ]; then BASE_METRICS="$p"; break; fi
done

echo "=============================================="
echo " Ablation Runner  (mode=$MODE)"
echo " Teacher checkpoint: $TEACHER_CKPT"
echo " Epochs per variant: $EPOCHS_PER_RUN"
echo "=============================================="

# ── Helper: run a single ablation variant ────────────────
run_one() {
  local TAG="$1"; shift
  local RUN_TAG="ablation_$TAG"
  local RESULT_DIR="results/$RUN_TAG"
  local CKPT_DIR="checkpoints/$RUN_TAG"

  # Skip if already completed
  if [ -f "$RESULT_DIR/metrics.json" ]; then
    local done=$(python3 -c "
import json
with open('$RESULT_DIR/metrics.json') as f: d = json.load(f)
print('yes' if 'student_distilled' in d else 'no')
" 2>/dev/null || echo "no")
    if [ "$done" = "yes" ]; then
      echo "[skip] $TAG — already completed"
      return
    fi
  fi

  echo ""
  echo ">>> [$TAG] started at $(date)"
  mkdir -p "$CKPT_DIR" "$RESULT_DIR"

  # Symlink teacher weights to avoid re-training
  ln -sf "$(pwd)/$TEACHER_CKPT" "$CKPT_DIR/teacher_best.weights.h5"

  # Copy non-distilled metrics so the script skips teacher + baselines
  if [ -n "$BASE_METRICS" ]; then
    python3 -c "
import json, os
with open('$BASE_METRICS') as f: src = json.load(f)
dst_path = '$RESULT_DIR/metrics.json'
dst = json.load(open(dst_path)) if os.path.exists(dst_path) else {}
for k in src:
    if k != 'student_distilled':
        dst[k] = src[k]
with open(dst_path, 'w') as f: json.dump(dst, f)
"
  fi

  env RUN_TAG="$RUN_TAG" SEED="$BASE_SEED" EPOCHS="$EPOCHS_PER_RUN" \
    SKIP_HEAVY_SOTA=1 ONLY_TRAIN=student_distilled \
    "$@" \
    python3 run_experiment.py

  echo ">>> [$TAG] completed at $(date)"
}

# ── (A) Branch-removal ablation ──────────────────────────
if [[ "$MODE" == "A" || "$MODE" == "AB" || "$MODE" == "ALL" ]]; then
  echo ""
  echo "===== (A) Branch-removal ablation ====="
  run_one "branch_no_low"  ABLATE_BRANCH=no_low
  run_one "branch_no_mid"  ABLATE_BRANCH=no_mid
  run_one "branch_no_high" ABLATE_BRANCH=no_high
fi

# ── (B) Distillation-loss decomposition ──────────────────
if [[ "$MODE" == "B" || "$MODE" == "AB" || "$MODE" == "ALL" ]]; then
  echo ""
  echo "===== (B) Distillation-loss decomposition ====="
  run_one "distill_kl_only"   DISTILL_MODE=kl_only
  run_one "distill_feat_only" DISTILL_MODE=feat_only
fi

# ── (C) Hyperparameter sensitivity (OFAT) ────────────────
if [[ "$MODE" == "C" || "$MODE" == "ALL" ]]; then
  echo ""
  echo "===== (C) Hyperparameter sensitivity (OFAT) ====="
  # Default is α=0.5, β=0.2, T=5 (the main experiment).
  # Vary one factor at a time:
  run_one "hp_a03_b02_t5" ALPHA=0.3 BETA=0.2 TEMPERATURE=5
  run_one "hp_a07_b02_t5" ALPHA=0.7 BETA=0.2 TEMPERATURE=5
  run_one "hp_a05_b01_t5" ALPHA=0.5 BETA=0.1 TEMPERATURE=5
  run_one "hp_a05_b03_t5" ALPHA=0.5 BETA=0.3 TEMPERATURE=5
  run_one "hp_a05_b02_t3" ALPHA=0.5 BETA=0.2 TEMPERATURE=3
  run_one "hp_a05_b02_t7" ALPHA=0.5 BETA=0.2 TEMPERATURE=7
fi

echo ""
echo "=============================================="
echo " Ablation experiments complete ($MODE)"
echo " Results: results/ablation_*/"
echo "=============================================="
