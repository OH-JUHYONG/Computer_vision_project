#!/bin/bash
# Figure 4-style PathMNIST experiment:
# fine-tune pretrained CLIP-ViT-B/32 on Original, Naive SD 1/2/5/10x,
# and DistDiff 1/2/5/10x.

set -euo pipefail

DATASET=${DATASET:-pathmnist}
MODEL=${MODEL:-open_clip_vit_b32}
LR=${CLIP_LR:-0.1}
GPU=${CLIP_GPU:-0}
SEEDS=${CLIP_SEEDS:-"1 2 3"}
RATIOS=${CLIP_RATIOS:-"1 2 5 10"}
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

echo "============================================================"
echo "  PathMNIST CLIP-ViT-B/32 scale fine-tuning 시작"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  workdir: $(pwd)"
echo "  gpu: ${GPU}"
echo "  lr: ${LR}, epochs: ${EPOCHS}"
echo "  seeds: ${SEEDS}"
echo "  ratios: ${RATIOS}"
echo "  workers: ${WORKERS}, cpu_threads: ${CPU_THREADS}"
echo "  train_batch_size: ${TRAIN_BATCH_SIZE}, val_batch_size: ${VAL_BATCH_SIZE}"
echo "============================================================"

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

for ratio in ${RATIOS}; do
    check_expansion_count "data/pathmnist_expansion/save/sd_batch_${ratio}x" "$((900 * ratio))"
    check_expansion_count "data/pathmnist_expansion/save/distdiff_batch_${ratio}x" "$((900 * ratio))"
done

ensure_not_partial() {
    local ckpt=$1
    if [ -d "${ckpt}" ] && [ ! -f "${ckpt}/results.yaml" ]; then
        echo "[ERROR] Partial checkpoint directory exists without results.yaml: ${ckpt}" >&2
        echo "        Let the running job finish, or inspect/remove this seed directory before resuming." >&2
        exit 1
    fi
}

run_original_seed() {
    local seed=$1
    local ckpt="checkpoint/${DATASET}/${MODEL}_pretrained_lr${LR}/seed${seed}"
    local log="logs/clip_scale_original_seed${seed}.log"

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
        --epochs "${EPOCHS}" > "${log}" 2>&1
}

run_expanded_seed() {
    local method=$1
    local exp=$2
    local seed=$3
    local ckpt="checkpoint/${DATASET}/${MODEL}_pretrained_${exp}_lr${LR}/seed${seed}"
    local log_method="${method// /_}"
    local log="logs/clip_scale_${log_method}_seed${seed}.log"

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
        --epochs "${EPOCHS}" > "${log}" 2>&1
}

for seed in ${SEEDS}; do
    run_original_seed "${seed}"
done

for ratio in ${RATIOS}; do
    for seed in ${SEEDS}; do
        run_expanded_seed "sd_${ratio}x" "save/sd_batch_${ratio}x" "${seed}"
    done
done

for ratio in ${RATIOS}; do
    for seed in ${SEEDS}; do
        run_expanded_seed "distdiff_${ratio}x" "save/distdiff_batch_${ratio}x" "${seed}"
    done
done

echo "============================================================"
echo "  PathMNIST CLIP-ViT-B/32 scale fine-tuning 완료"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo
echo "=== Final results ==="
python src/parse_logs.py "checkpoint/${DATASET}/${MODEL}_pretrained_lr${LR}" --multi
for ratio in ${RATIOS}; do
    python src/parse_logs.py "checkpoint/${DATASET}/${MODEL}_pretrained_save/sd_batch_${ratio}x_lr${LR}" --multi
done
for ratio in ${RATIOS}; do
    python src/parse_logs.py "checkpoint/${DATASET}/${MODEL}_pretrained_save/distdiff_batch_${ratio}x_lr${LR}" --multi
done
