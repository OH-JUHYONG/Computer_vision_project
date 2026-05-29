#!/bin/bash
# Table 2-style PathMNIST stronger-classifier experiment:
# fine-tune LAION-pretrained CLIP-ViT-B/32 on Original, Naive SD 5x, and DistDiff 5x.

set -euo pipefail

DATASET=${DATASET:-pathmnist}
MODEL=${MODEL:-open_clip_vit_b32}
LR=${CLIP_LR:-1e-4}
OPTIMIZER=${OPTIMIZER:-adamw}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
GRAD_CLIP=${GRAD_CLIP:-1.0}
GPU=${CLIP_GPU:-0}
SEEDS=${CLIP_SEEDS:-"1 2 3"}
EXPAND_NUM=${EXPAND_NUM:-5}
EPOCHS=${EPOCHS:-100}
WORKERS=${WORKERS:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
CPU_THREADS=${CPU_THREADS:-2}
SETTINGS=${SETTINGS:-"original sd distdiff"}
FORCE_RERUN=${FORCE_RERUN:-False}

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
RESULT_ROOT="experiments/comparison_with_stronger_classification_models/results"
mkdir -p "${RESULT_ROOT}"

check_expansion_count() {
    local dir=$1
    local expected=$2
    local count
    if [ ! -d "${dir}" ]; then
        echo "[ERROR] Missing expansion directory: ${dir}" >&2
        exit 1
    fi
    count=$(find "${dir}" -name "*.png" | wc -l)
    if [ "${count}" -ne "${expected}" ]; then
        echo "[ERROR] ${dir}: expected ${expected} PNGs, found ${count}" >&2
        exit 1
    fi
}

check_expansion_count "data/${DATASET}_expansion/save/sd_batch_${EXPAND_NUM}x" "$((900 * EXPAND_NUM))"
check_expansion_count "data/${DATASET}_expansion/save/distdiff_batch_${EXPAND_NUM}x" "$((900 * EXPAND_NUM))"

echo "============================================================"
echo "  PathMNIST stronger classification models experiment 시작"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  model: ${MODEL} pretrained"
echo "  pretrained: OpenCLIP ViT-B-32/laion2b_s34b_b79k"
echo "  optimizer: ${OPTIMIZER}, lr: ${LR}, weight_decay: ${WEIGHT_DECAY}, grad_clip: ${GRAD_CLIP}"
echo "  settings: ${SETTINGS}"
echo "  expand_num: ${EXPAND_NUM}"
echo "============================================================"

maybe_clear_checkpoint() {
    local ckpt=$1
    if [ "${FORCE_RERUN}" = "True" ] || [ "${FORCE_RERUN}" = "true" ] || [ "${FORCE_RERUN}" = "1" ]; then
        if [ -d "${ckpt}" ]; then
            echo "[FORCE_RERUN] Removing ${ckpt}"
            rm -rf "${ckpt}"
        fi
    fi
}

ensure_not_partial() {
    local ckpt=$1
    if [ -d "${ckpt}" ] && [ ! -f "${ckpt}/results.yaml" ]; then
        echo "[ERROR] Partial checkpoint directory exists without results.yaml: ${ckpt}" >&2
        exit 1
    fi
}

run_original_seed() {
    local seed=$1
    local ckpt="${RESULT_ROOT}/original_${OPTIMIZER}_lr${LR}_wd${WEIGHT_DECAY}/seed${seed}"
    local log="logs/stronger_${MODEL}_original_seed${seed}.log"

    maybe_clear_checkpoint "${ckpt}"
    if [ -f "${ckpt}/results.yaml" ]; then
        echo "[SKIP] Original seed${seed}: ${ckpt}/results.yaml exists"
        return
    fi
    ensure_not_partial "${ckpt}"

    echo "[RUN] Original seed${seed} -> ${log}"
    CUDA_VISIBLE_DEVICES=${GPU} python src/train.py \
        -a "${MODEL}" \
        -d "${DATASET}" \
        --checkpoint "${ckpt}" \
        --data_dir data \
        --manualSeed "${seed}" \
        --pretrained \
        --workers "${WORKERS}" \
        --train-batch-size "${TRAIN_BATCH_SIZE}" \
        --val-batch-size "${VAL_BATCH_SIZE}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --optimizer "${OPTIMIZER}" \
        --grad-clip "${GRAD_CLIP}" \
        --epochs "${EPOCHS}" > "${log}" 2>&1
}

run_expanded_seed() {
    local method=$1
    local exp=$2
    local seed=$3
    local ckpt="${RESULT_ROOT}/${method}_batch_${EXPAND_NUM}x_${OPTIMIZER}_lr${LR}_wd${WEIGHT_DECAY}/seed${seed}"
    local log="logs/stronger_${MODEL}_${method}_${EXPAND_NUM}x_seed${seed}.log"

    maybe_clear_checkpoint "${ckpt}"
    if [ -f "${ckpt}/results.yaml" ]; then
        echo "[SKIP] ${method} seed${seed}: ${ckpt}/results.yaml exists"
        return
    fi
    ensure_not_partial "${ckpt}"

    echo "[RUN] ${method} seed${seed} -> ${log}"
    CUDA_VISIBLE_DEVICES=${GPU} python src/train_expanded_data_concat_original.py \
        -a "${MODEL}" \
        -d "${DATASET}" \
        --checkpoint "${ckpt}" \
        --data_dir data \
        --manualSeed "${seed}" \
        --data_expanded_dir "data/${DATASET}_expansion/${exp}" \
        --pretrained \
        --workers "${WORKERS}" \
        --train-batch-size "${TRAIN_BATCH_SIZE}" \
        --val-batch-size "${VAL_BATCH_SIZE}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --optimizer "${OPTIMIZER}" \
        --grad-clip "${GRAD_CLIP}" \
        --epochs "${EPOCHS}" > "${log}" 2>&1
}

for setting in ${SETTINGS}; do
    case "${setting}" in
        original)
            for seed in ${SEEDS}; do
                run_original_seed "${seed}"
            done
            ;;
        sd|naive_sd)
            for seed in ${SEEDS}; do
                run_expanded_seed "sd" "save/sd_batch_${EXPAND_NUM}x" "${seed}"
            done
            ;;
        distdiff)
            for seed in ${SEEDS}; do
                run_expanded_seed "distdiff" "save/distdiff_batch_${EXPAND_NUM}x" "${seed}"
            done
            ;;
        *)
            echo "[ERROR] Unknown setting: ${setting}. Use original, sd, or distdiff." >&2
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  PathMNIST stronger classification models experiment 완료"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo
echo "=== Final results ==="
for setting in ${SETTINGS}; do
    case "${setting}" in
        original)
            python src/parse_logs.py "${RESULT_ROOT}/original_${OPTIMIZER}_lr${LR}_wd${WEIGHT_DECAY}" --multi || true
            ;;
        sd|naive_sd)
            python src/parse_logs.py "${RESULT_ROOT}/sd_batch_${EXPAND_NUM}x_${OPTIMIZER}_lr${LR}_wd${WEIGHT_DECAY}" --multi || true
            ;;
        distdiff)
            python src/parse_logs.py "${RESULT_ROOT}/distdiff_batch_${EXPAND_NUM}x_${OPTIMIZER}_lr${LR}_wd${WEIGHT_DECAY}" --multi || true
            ;;
    esac
done
