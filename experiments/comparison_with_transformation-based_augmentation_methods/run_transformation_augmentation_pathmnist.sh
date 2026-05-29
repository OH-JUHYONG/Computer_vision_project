#!/bin/bash
# Table 1-style PathMNIST experiment:
# compare transformation-based augmentation on Original, Naive SD, and DistDiff 5x data.

set -euo pipefail

DATASET=${DATASET:-pathmnist}
MODEL=${MODEL:-resnet50}
LR=${LR:-0.1}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"1 2 3"}
TRANSFORMS=${TRANSFORMS:-"default autoaug randaug cutout gridmask mixup cutmix"}
SETTINGS=${SETTINGS:-"original distdiff"}
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
RESULT_ROOT="experiments/comparison_with_transformation-based_augmentation_methods/results"
mkdir -p "${RESULT_ROOT}"

EXPECTED=$((900 * EXPAND_NUM))

expanded_dir_for_setting() {
    local setting=$1
    case "${setting}" in
        original)
            echo ""
            ;;
        distdiff)
            echo "data/${DATASET}_expansion/save/distdiff_batch_${EXPAND_NUM}x"
            ;;
        sd|naive_sd)
            echo "data/${DATASET}_expansion/save/sd_batch_${EXPAND_NUM}x"
            ;;
        *)
            echo "[ERROR] Unknown setting: ${setting}. Use original, distdiff, or sd." >&2
            exit 1
            ;;
    esac
}

for setting in ${SETTINGS}; do
    exp_dir=$(expanded_dir_for_setting "${setting}")
    if [ -n "${exp_dir}" ]; then
        if [ ! -d "${exp_dir}" ]; then
            echo "[ERROR] Missing expansion directory for ${setting}: ${exp_dir}" >&2
            exit 1
        fi

        count=$(find "${exp_dir}" -name "*.png" | wc -l)
        if [ "${count}" -ne "${EXPECTED}" ]; then
            echo "[ERROR] ${exp_dir}: expected ${EXPECTED} PNGs, found ${count}" >&2
            exit 1
        fi
    fi
done

echo "============================================================"
echo "  PathMNIST transformation augmentation comparison 시작"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  transforms: ${TRANSFORMS}"
echo "  settings: ${SETTINGS}"
echo "  expand_num: ${EXPAND_NUM}"
echo "============================================================"

ensure_not_partial() {
    local ckpt=$1
    if [ -d "${ckpt}" ] && [ ! -f "${ckpt}/results.yaml" ]; then
        echo "[ERROR] Partial checkpoint directory exists without results.yaml: ${ckpt}" >&2
        exit 1
    fi
}

run_one() {
    local setting=$1
    local transform=$2
    local seed=$3
    local ckpt="${RESULT_ROOT}/${setting}_${transform}_${EXPAND_NUM}x_lr${LR}/seed${seed}"
    local log="logs/transform_${setting}_${transform}_${EXPAND_NUM}x_seed${seed}.log"

    if [ -f "${ckpt}/results.yaml" ]; then
        echo "[SKIP] ${setting} ${transform} seed${seed}: ${ckpt}/results.yaml exists"
        return
    fi
    ensure_not_partial "${ckpt}"

    echo "[RUN] ${setting} ${transform} seed${seed} -> ${log}"
    exp_dir=$(expanded_dir_for_setting "${setting}")
    if [ "${setting}" = "original" ]; then
        CUDA_VISIBLE_DEVICES=${GPU} python src/train_transform.py \
            -a "${MODEL}" \
            -d "${DATASET}" \
            --checkpoint "${ckpt}" \
            --data_dir data \
            --manualSeed "${seed}" \
            --expand_num "${EXPAND_NUM}" \
            --transform_type "${transform}" \
            --workers "${WORKERS}" \
            --train-batch-size "${TRAIN_BATCH_SIZE}" \
            --val-batch-size "${VAL_BATCH_SIZE}" \
            --lr "${LR}" \
            --epochs "${EPOCHS}" > "${log}" 2>&1
    else
        CUDA_VISIBLE_DEVICES=${GPU} python src/train_transform.py \
            -a "${MODEL}" \
            -d "${DATASET}" \
            --checkpoint "${ckpt}" \
            --data_dir data \
            --manualSeed "${seed}" \
            --data_expanded_dir "${exp_dir}" \
            --expand_num "${EXPAND_NUM}" \
            --transform_type "${transform}" \
            --workers "${WORKERS}" \
            --train-batch-size "${TRAIN_BATCH_SIZE}" \
            --val-batch-size "${VAL_BATCH_SIZE}" \
            --lr "${LR}" \
            --epochs "${EPOCHS}" > "${log}" 2>&1
    fi
}

for setting in ${SETTINGS}; do
    for transform in ${TRANSFORMS}; do
        for seed in ${SEEDS}; do
            run_one "${setting}" "${transform}" "${seed}"
        done
    done
done

echo "============================================================"
echo "  PathMNIST transformation augmentation comparison 완료"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo
echo "=== Final results ==="
for setting in ${SETTINGS}; do
    for transform in ${TRANSFORMS}; do
        python src/parse_logs.py "${RESULT_ROOT}/${setting}_${transform}_${EXPAND_NUM}x_lr${LR}" --multi || true
    done
done
