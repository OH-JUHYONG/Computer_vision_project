#!/bin/bash
# Live monitor for PathMNIST architecture versatility experiment.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MODEL_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
cd "${MODEL_ROOT}"

DATASET=${DATASET:-pathmnist}
BACKBONES=${BACKBONES:-"resnet50 resnext50 wideresnet50 mobilenetv2"}
LR=${LR:-0.1}
EXPAND_NUM=${EXPAND_NUM:-5}
INTERVAL=${INTERVAL:-30}
RESULT_ROOT="experiments/versatility_to_various_architectures/results"

count_results() {
    local base=$1
    if [ -d "${base}" ]; then
        find "${base}" -path "*/seed*/results.yaml" -type f | wc -l
    else
        echo 0
    fi
}

while true; do
    clear
    echo "============================================================"
    echo "  PathMNIST architecture versatility 진행 상황"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo

    ps -eo pid,ppid,stat,etime,cmd | grep -E "run_versatility_architectures_pathmnist|src/train.py|src/train_expanded_data_concat_original.py" | grep -v grep || echo "  실행 중인 architecture 학습 프로세스 없음"
    echo

    total=0
    target=0
    for backbone in ${BACKBONES}; do
        echo "[${backbone}]"
        original_base="${RESULT_ROOT}/${backbone}_original_lr${LR}"
        distdiff_base="${RESULT_ROOT}/${backbone}_distdiff_batch_${EXPAND_NUM}x_lr${LR}"
        original_count=$(count_results "${original_base}")
        distdiff_count=$(count_results "${distdiff_base}")
        total=$((total + original_count + distdiff_count))
        target=$((target + 6))
        printf "  original:        %d / 3\n" "${original_count}"
        printf "  distdiff_%sx:    %d / 3\n" "${EXPAND_NUM}" "${distdiff_count}"
        echo
    done

    echo "[total] ${total} / ${target} results.yaml"
    echo
    latest_log=$(find logs -maxdepth 1 -name "arch_*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}' || true)
    if [ -n "${latest_log}" ]; then
        echo "[latest log] ${latest_log}"
        tail -n 24 "${latest_log}"
    else
        echo "  architecture log 없음"
    fi

    echo
    echo "다음 갱신까지 ${INTERVAL}s. 종료: Ctrl-C"
    sleep "${INTERVAL}"
done
