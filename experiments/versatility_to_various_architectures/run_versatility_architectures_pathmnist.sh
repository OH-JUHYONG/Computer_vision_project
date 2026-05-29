#!/bin/bash
# Table 3-style PathMNIST experiment:
# train several scratch classifier backbones on Original and DistDiff 5x data.

set -euo pipefail

DATASET=${DATASET:-pathmnist}
BACKBONES=${BACKBONES:-"resnet50 resnext50 wideresnet50 mobilenetv2"}
LR=${LR:-0.1}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"1 2 3"}
EXPAND_NUM=${EXPAND_NUM:-5}
EPOCHS=${EPOCHS:-100}
WORKERS=${WORKERS:-0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
CPU_THREADS=${CPU_THREADS:-2}

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MODEL_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
cd "${MODEL_ROOT}"

export OMP_NUM_THREADS=${CPU_THREADS}
export MKL_NUM_THREADS=${CPU_THREADS}
export OPENBLAS_NUM_THREADS=${CPU_THREADS}
export NUMEXPR_NUM_THREADS=${CPU_THREADS}
export VECLIB_MAXIMUM_THREADS=${CPU_THREADS}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export PYTHONUNBUFFERED=1

mkdir -p logs
RESULT_ROOT="experiments/versatility_to_various_architectures/results"
mkdir -p "${RESULT_ROOT}"

EXP_DIR="data/${DATASET}_expansion/save/distdiff_batch_${EXPAND_NUM}x"
EXPECTED=$((900 * EXPAND_NUM))

if [ ! -d "${EXP_DIR}" ]; then
    echo "[ERROR] Missing DistDiff expansion directory: ${EXP_DIR}" >&2
    exit 1
fi

count=$(find "${EXP_DIR}" -name "*.png" | wc -l)
if [ "${count}" -ne "${EXPECTED}" ]; then
    echo "[ERROR] ${EXP_DIR}: expected ${EXPECTED} PNGs, found ${count}" >&2
    exit 1
fi

echo "============================================================"
echo "  PathMNIST architecture versatility experiment 시작"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  backbones: ${BACKBONES}"
echo "  expand_num: ${EXPAND_NUM}"
echo "============================================================"

ensure_not_partial() {
    local ckpt=$1
    if [ -d "${ckpt}" ] && [ ! -f "${ckpt}/results.yaml" ]; then
        echo "[ERROR] Partial checkpoint directory exists without results.yaml: ${ckpt}" >&2
        exit 1
    fi
}

run_original_seed() {
    local backbone=$1
    local seed=$2
    local ckpt="${RESULT_ROOT}/${backbone}_original_lr${LR}/seed${seed}"
    local log="logs/arch_${backbone}_original_seed${seed}.log"

    if [ -f "${ckpt}/results.yaml" ]; then
        echo "[SKIP] ${backbone} original seed${seed}: ${ckpt}/results.yaml exists"
        return
    fi
    ensure_not_partial "${ckpt}"

    echo "[RUN] ${backbone} original seed${seed} -> ${log}"
    CUDA_VISIBLE_DEVICES=${GPU} python src/train.py \
        -a "${backbone}" \
        -d "${DATASET}" \
        --checkpoint "${ckpt}" \
        --data_dir data \
        --manualSeed "${seed}" \
        --workers "${WORKERS}" \
        --train-batch-size "${TRAIN_BATCH_SIZE}" \
        --val-batch-size "${VAL_BATCH_SIZE}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" > "${log}" 2>&1
}

run_distdiff_seed() {
    local backbone=$1
    local seed=$2
    local ckpt="${RESULT_ROOT}/${backbone}_distdiff_batch_${EXPAND_NUM}x_lr${LR}/seed${seed}"
    local log="logs/arch_${backbone}_distdiff_${EXPAND_NUM}x_seed${seed}.log"

    if [ -f "${ckpt}/results.yaml" ]; then
        echo "[SKIP] ${backbone} DistDiff ${EXPAND_NUM}x seed${seed}: ${ckpt}/results.yaml exists"
        return
    fi
    ensure_not_partial "${ckpt}"

    echo "[RUN] ${backbone} DistDiff ${EXPAND_NUM}x seed${seed} -> ${log}"
    CUDA_VISIBLE_DEVICES=${GPU} python src/train_expanded_data_concat_original.py \
        -a "${backbone}" \
        -d "${DATASET}" \
        --checkpoint "${ckpt}" \
        --data_dir data \
        --manualSeed "${seed}" \
        --data_expanded_dir "${EXP_DIR}" \
        --workers "${WORKERS}" \
        --train-batch-size "${TRAIN_BATCH_SIZE}" \
        --val-batch-size "${VAL_BATCH_SIZE}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" > "${log}" 2>&1
}

for backbone in ${BACKBONES}; do
    for seed in ${SEEDS}; do
        run_original_seed "${backbone}" "${seed}"
    done
    for seed in ${SEEDS}; do
        run_distdiff_seed "${backbone}" "${seed}"
    done
done

echo "============================================================"
echo "  PathMNIST architecture versatility experiment 완료"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo
echo "=== Final results ==="
for backbone in ${BACKBONES}; do
    python src/parse_logs.py "${RESULT_ROOT}/${backbone}_original_lr${LR}" --multi
    python src/parse_logs.py "${RESULT_ROOT}/${backbone}_distdiff_batch_${EXPAND_NUM}x_lr${LR}" --multi
done
