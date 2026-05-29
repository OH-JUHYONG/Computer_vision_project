#!/bin/bash
# Live monitor for Figure 4-style ResNet-50 scratch PathMNIST scale training.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MODEL_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
cd "${MODEL_ROOT}"

DATASET=${DATASET:-pathmnist}
MODEL=${MODEL:-resnet50}
LR=${RESNET_LR:-0.1}
RATIOS=${RESNET_RATIOS:-"1 2 5 10"}
INTERVAL=${INTERVAL:-30}
RESULT_ROOT="experiments/scaling_in_number_of_data/results"

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
    echo "  PathMNIST ResNet-50 scratch scale training 진행 상황"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo

    echo "[process]"
    ps -eo pid,ppid,stat,etime,cmd | grep -E "run_scale_resnet50_pathmnist|src/train.py -a resnet50|src/train_expanded_data_concat_original.py -a resnet50" | grep -v grep || echo "  실행 중인 ResNet 학습 프로세스 없음"
    echo

    total=0
    original_base="${RESULT_ROOT}/original_only"
    original_count=$(count_results "${original_base}")
    total=$((total + original_count))
    echo "[Original] ${original_count}/3 results.yaml"
    show_seed_status "${original_base}"

    echo
    echo "[Naive SD]"
    for ratio in ${RATIOS}; do
        base="${RESULT_ROOT}/sd_batch_${ratio}x_lr${LR}"
        count=$(count_results "${base}")
        total=$((total + count))
        printf "  sd_%sx: %d / 3\n" "${ratio}" "${count}"
        show_seed_status "${base}"
    done

    echo
    echo "[DistDiff]"
    for ratio in ${RATIOS}; do
        base="${RESULT_ROOT}/distdiff_batch_${ratio}x_lr${LR}"
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
    if [ -f logs/resnet_scale_master.log ]; then
        tail -n 18 logs/resnet_scale_master.log
    else
        echo "  logs/resnet_scale_master.log 없음"
    fi

    echo
    latest_log=$(find logs -maxdepth 1 -name "resnet_scale_*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}' || true)
    echo "[latest run log tail]"
    if [ -n "${latest_log}" ]; then
        echo "  ${latest_log}"
        tail -n 24 "${latest_log}"
    else
        echo "  ResNet run log 없음"
    fi

    echo
    echo "다음 갱신까지 ${INTERVAL}s. 종료: Ctrl-C"
    sleep "${INTERVAL}"
done
