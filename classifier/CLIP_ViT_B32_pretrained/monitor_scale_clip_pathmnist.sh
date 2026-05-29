#!/bin/bash
# Live monitor for Figure 4-style CLIP-ViT-B/32 PathMNIST scale fine-tuning.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MODEL_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
cd "${MODEL_ROOT}"

DATASET=${DATASET:-pathmnist}
MODEL=${MODEL:-open_clip_vit_b32}
LR=${CLIP_LR:-0.1}
RATIOS=${CLIP_RATIOS:-"1 2 5 10"}
INTERVAL=${INTERVAL:-30}

count_results() {
    local base=$1
    if [ -d "${base}" ]; then
        find "${base}" -path "*/seed*/results.yaml" -type f | wc -l
    else
        echo 0
    fi
}

show_seed_status() {
    local base=$1
    for seed in 1 2 3; do
        local result_file="${base}/seed${seed}/results.yaml"
        if [ -f "${result_file}" ]; then
            local acc
            acc=$(awk '/best_accuracy:/ {print $2}' "${result_file}")
            printf "    seed%s: done, best_accuracy=%s\n" "${seed}" "${acc}"
        elif [ -d "${base}/seed${seed}" ]; then
            printf "    seed%s: running or incomplete\n" "${seed}"
        else
            printf "    seed%s: pending\n" "${seed}"
        fi
    done
}

while true; do
    clear
    echo "============================================================"
    echo "  PathMNIST CLIP-ViT-B/32 scale fine-tuning 진행 상황"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo

    echo "[process]"
    ps -eo pid,ppid,stat,etime,cmd | grep -E "run_scale_clip_pathmnist|run_table2_clip_pathmnist|train.py|train_expanded_data_concat_original|open_clip_vit_b32" | grep -v grep || echo "  실행 중인 CLIP 학습 프로세스 없음"
    echo

    total=0
    original_base="checkpoint/${DATASET}/${MODEL}_pretrained_lr${LR}"
    original_count=$(count_results "${original_base}")
    total=$((total + original_count))
    echo "[Original] ${original_count}/3 results.yaml"
    show_seed_status "${original_base}"

    echo
    echo "[Naive SD]"
    for ratio in ${RATIOS}; do
        base="checkpoint/${DATASET}/${MODEL}_pretrained_save/sd_batch_${ratio}x_lr${LR}"
        count=$(count_results "${base}")
        total=$((total + count))
        printf "  sd_%sx: %d / 3\n" "${ratio}" "${count}"
        show_seed_status "${base}"
    done

    echo
    echo "[DistDiff]"
    for ratio in ${RATIOS}; do
        base="checkpoint/${DATASET}/${MODEL}_pretrained_save/distdiff_batch_${ratio}x_lr${LR}"
        count=$(count_results "${base}")
        total=$((total + count))
        printf "  distdiff_%sx: %d / 3\n" "${ratio}" "${count}"
        show_seed_status "${base}"
    done

    target=$((3 + 2 * 4 * 3))
    echo
    echo "[total] ${total} / ${target} results.yaml"

    echo
    echo "[master log tail]"
    if [ -f logs/clip_scale_master.log ]; then
        tail -n 18 logs/clip_scale_master.log
    elif [ -f logs/table2_clip_master.log ]; then
        tail -n 18 logs/table2_clip_master.log
    else
        echo "  CLIP master log 없음"
    fi

    echo
    latest_log=$(find logs -maxdepth 1 \( -name "clip_scale_*.log" -o -name "table2_clip_*.log" \) -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}' || true)
    echo "[latest run log tail]"
    if [ -n "${latest_log}" ]; then
        echo "  ${latest_log}"
        tail -n 24 "${latest_log}"
    else
        echo "  CLIP run log 없음"
    fi

    echo
    echo "다음 갱신까지 ${INTERVAL}s. 종료: Ctrl-C"
    sleep "${INTERVAL}"
done
