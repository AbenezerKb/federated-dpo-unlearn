#!/usr/bin/env bash
# Federated DPO Unlearning Experiment Runner
# Phase 1: Federated DPO alignment (train)
# Phase 2: Federated unlearning (one client requests deletion)
# Phase 3: Evaluation on TOFU metrics
# Loops over all 5 objectives (or specify one)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-configs/default.yaml}"
OBJECTIVES="${2:-all}"
OUTPUT_DIR="${3:-outputs}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"

mkdir -p "$RUN_DIR"

echo "============================================"
echo "Federated DPO Unlearning Experiment"
echo "============================================"
echo "Config:     $CONFIG"
echo "Objectives: $OBJECTIVES"
echo "Output:     $RUN_DIR"
echo "Started:    $(date)"
echo "============================================"

# Check GPU availability
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"

if [ "$OBJECTIVES" = "all" ]; then
    OBJECTIVE_LIST=("dpo" "npo" "simnpo" "simpo" "fugas")
else
    IFS=',' read -ra OBJECTIVE_LIST <<< "$OBJECTIVES"
fi

# Phase 1: Federated DPO alignment (shared across all objectives)
ALIGNMENT_CKPT="${RUN_DIR}/alignment_checkpoint"

echo ""
echo "============================================"
echo "PHASE 1: Federated DPO Alignment"
echo "============================================"

if [ ! -d "$ALIGNMENT_CKPT" ]; then
    python3 -m scripts.run_phase \
        --config "$CONFIG" \
        --phase alignment \
        --output-dir "$ALIGNMENT_CKPT" \
        2>&1 | tee "${RUN_DIR}/phase1_alignment.log"
    echo "Alignment checkpoint saved to: $ALIGNMENT_CKPT"
else
    echo "Alignment checkpoint found, skipping Phase 1"
fi

# Phase 2 & 3: Unlearning + Evaluation for each objective
for objective in "${OBJECTIVE_LIST[@]}"; do
    echo ""
    echo "============================================"
    echo "PHASE 2: Unlearning with objective=$objective"
    echo "============================================"

    OBJ_DIR="${RUN_DIR}/${objective}"
    mkdir -p "$OBJ_DIR"

    python3 -m scripts.run_phase \
        --config "$CONFIG" \
        --phase unlearning \
        --objective "$objective" \
        --alignment-checkpoint "$ALIGNMENT_CKPT" \
        --output-dir "$OBJ_DIR" \
        2>&1 | tee "${OBJ_DIR}/phase2_unlearning.log"

    echo ""
    echo "============================================"
    echo "PHASE 3: Evaluation ($objective)"
    echo "============================================"

    python3 -m scripts.run_phase \
        --config "$CONFIG" \
        --phase eval \
        --objective "$objective" \
        --model-checkpoint "${OBJ_DIR}/unlearned_model" \
        --output-dir "$OBJ_DIR" \
        2>&1 | tee "${OBJ_DIR}/phase3_eval.log"

    echo "Results for $objective saved to: $OBJ_DIR"
done

# Aggregate results
echo ""
echo "============================================"
echo "Aggregating Results"
echo "============================================"

python3 -m scripts.aggregate_results \
    --run-dir "$RUN_DIR" \
    --objectives "${OBJECTIVE_LIST[*]}" \
    2>&1 | tee "${RUN_DIR}/aggregate.log"

echo ""
echo "============================================"
echo "Experiment Complete"
echo "============================================"
echo "Results: $RUN_DIR"
echo "Finished: $(date)"
