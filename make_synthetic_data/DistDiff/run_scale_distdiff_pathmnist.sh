#!/bin/bash
# Generate PathMNIST DistDiff synthetic data for 1/2/5/10x.

set -euo pipefail

DATASET=${DATASET:-pathmnist}
RATIOS=${DISTDIFF_RATIOS:-"1 2 5 10"}
GPU=${DISTDIFF_GPU:-0}
TOTAL_SPLIT=${DISTDIFF_TOTAL_SPLIT:-1}
GUIDE_MODEL=${GUIDE_MODEL:-resnet50}
GUIDANCE_TYPE="transform_guidance"
SCALE=${SCALE:-7.5}
GUIDANCE_STEP=${GUIDANCE_STEP:-10}
GUIDANCE_PERIOD=${GUIDANCE_PERIOD:-2}
CONSTRAINT_VALUE=${CONSTRAINT_VALUE:-0.2}
K=${K:-3}
RHO=${RHO:-10.0}
STRENGTH=${STRENGTH:-0.2}
EXPECTED_BASE=${EXPECTED_BASE:-900}
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

GUIDE_MODEL_WEIGHT="checkpoint/${DATASET}/${GUIDE_MODEL}_unpretrained_lr0.1/seed1/model_best.pth.tar"

echo "============================================================"
echo "  PathMNIST DistDiff synthetic generation 시작"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  workdir: $(pwd)"
echo "  gpu: ${GPU}, total_split: ${TOTAL_SPLIT}"
echo "  ratios: ${RATIOS}"
echo "  strength: ${STRENGTH}, guidance_step: ${GUIDANCE_STEP}"
echo "============================================================"

if [ ! -f "${GUIDE_MODEL_WEIGHT}" ]; then
    echo "[ERROR] Missing guide checkpoint: ${GUIDE_MODEL_WEIGHT}" >&2
    exit 1
fi

for ratio in ${RATIOS}; do
    out_dir="data/${DATASET}_expansion/save/distdiff_batch_${ratio}x"
    expected=$((EXPECTED_BASE * ratio))
    mkdir -p "${out_dir}"
    current=$(find "${out_dir}" -name "*.png" 2>/dev/null | wc -l)

    echo
    echo "[${ratio}x] current: ${current} / ${expected}"
    if [ "${current}" -eq "${expected}" ]; then
        echo "[SKIP] DistDiff ${ratio}x already complete: ${out_dir}"
        continue
    fi
    if [ "${current}" -gt "${expected}" ]; then
        echo "[ERROR] DistDiff ${ratio}x has too many PNGs: ${current} / ${expected}" >&2
        exit 1
    fi

    for split in $(seq 0 $((TOTAL_SPLIT - 1))); do
        log_file="logs/distdiff_generate_${ratio}x_split${split}.log"
        echo "[RUN] DistDiff ${ratio}x split ${split}/${TOTAL_SPLIT} -> ${log_file}"
        CUDA_VISIBLE_DEVICES=${GPU} python src/generate_data.py \
            --guidance_type="${GUIDANCE_TYPE}" \
            -a "${GUIDE_MODEL}" \
            -d "${DATASET}" \
            --output_dir "${out_dir}" \
            --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
            --gradient_checkpointing \
            --K "${K}" \
            --train_batch_size 1 \
            --optimize_targets "global_prototype-local_prototype" \
            --strength "${STRENGTH}" \
            --num_images_per_prompt "${ratio}" \
            --guidance_step "${GUIDANCE_STEP}" \
            --guidance_period "${GUIDANCE_PERIOD}" \
            --encoder_weight_path "${GUIDE_MODEL_WEIGHT}" \
            --guidance_scale "${SCALE}" \
            --constraint_value "${CONSTRAINT_VALUE}" \
            --rho "${RHO}" \
            --total_split "${TOTAL_SPLIT}" \
            --split "${split}" > "${log_file}" 2>&1
    done

    final_count=$(find "${out_dir}" -name "*.png" 2>/dev/null | wc -l)
    echo "[${ratio}x] final: ${final_count} / ${expected}"
    if [ "${final_count}" -ne "${expected}" ]; then
        echo "[ERROR] DistDiff ${ratio}x count mismatch. Check split logs." >&2
        exit 1
    fi
done

echo
echo "============================================================"
echo "  PathMNIST DistDiff synthetic generation 완료"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
