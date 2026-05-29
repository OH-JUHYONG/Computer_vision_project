#!/bin/bash
# Live monitor for PathMNIST transformation augmentation comparison.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MODEL_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
cd "${MODEL_ROOT}"

DATASET=${DATASET:-pathmnist}
MODEL=${MODEL:-resnet50}
LR=${LR:-0.1}
EXPAND_NUM=${EXPAND_NUM:-5}
TRANSFORMS=${TRANSFORMS:-"default autoaug randaug cutout gridmask mixup cutmix"}
SETTINGS=${SETTINGS:-"original distdiff"}
SEEDS=${SEEDS:-"1 2 3"}
INTERVAL=${INTERVAL:-30}
RESULT_ROOT="experiments/comparison_with_transformation-based_augmentation_methods/results"

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
    echo "  PathMNIST transformation augmentation 진행 상황"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo

    ps -eo pid,ppid,stat,etime,cmd | grep -E "run_transformation_augmentation_pathmnist|src/train_transform.py" | grep -v grep || echo "  실행 중인 transformation 학습 프로세스 없음"
    echo

    total=0
    transform_count=0
    for _ in ${TRANSFORMS}; do
        transform_count=$((transform_count + 1))
    done
    setting_count=0
    for _ in ${SETTINGS}; do
        setting_count=$((setting_count + 1))
    done
    seed_count=0
    for _ in ${SEEDS}; do
        seed_count=$((seed_count + 1))
    done
    for setting in ${SETTINGS}; do
        echo "[${setting}]"
        for transform in ${TRANSFORMS}; do
            base="${RESULT_ROOT}/${setting}_${transform}_${EXPAND_NUM}x_lr${LR}"
            count=$(count_results "${base}")
            total=$((total + count))
            printf "  %-8s %d / %d\n" "${transform}" "${count}" "${seed_count}"
        done
        echo
    done

    echo "[total] ${total} / $((setting_count * transform_count * seed_count)) results.yaml"
    echo
    latest_log=$(find logs -maxdepth 1 -name "transform_*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}' || true)
    if [ -n "${latest_log}" ]; then
        echo "[latest log] ${latest_log}"
        tail -n 24 "${latest_log}"
    else
        echo "  transform log 없음"
    fi

    echo
    echo "다음 갱신까지 ${INTERVAL}s. 종료: Ctrl-C"
    sleep "${INTERVAL}"
done
