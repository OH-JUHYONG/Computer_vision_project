#!/bin/bash
# Live monitor for PathMNIST stronger-classifier experiment.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MODEL_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
cd "${MODEL_ROOT}"

DATASET=${DATASET:-pathmnist}
MODEL=${MODEL:-open_clip_vit_b32}
LR=${CLIP_LR:-1e-4}
OPTIMIZER=${OPTIMIZER:-adamw}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
EXPAND_NUM=${EXPAND_NUM:-5}
SETTINGS=${SETTINGS:-"original sd distdiff"}
SEEDS=${CLIP_SEEDS:-"1 2 3"}
INTERVAL=${INTERVAL:-30}
RESULT_ROOT="experiments/comparison_with_stronger_classification_models/results"

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
    for seed in ${SEEDS}; do
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
    echo "  PathMNIST stronger classification models 진행 상황"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo

    ps -eo pid,ppid,stat,etime,cmd | grep -E "run_stronger_classification_models_pathmnist|src/train.py|src/train_expanded_data_concat_original.py|open_clip_vit_b32" | grep -v grep || echo "  실행 중인 stronger classifier 프로세스 없음"
    echo

    seed_count=0
    for _ in ${SEEDS}; do
        seed_count=$((seed_count + 1))
    done

    total=0
    setting_count=0
    for setting in ${SETTINGS}; do
        setting_count=$((setting_count + 1))
        case "${setting}" in
            original)
                label="Original"
                base="${RESULT_ROOT}/original_${OPTIMIZER}_lr${LR}_wd${WEIGHT_DECAY}"
                ;;
            sd|naive_sd)
                label="Naive SD ${EXPAND_NUM}x"
                base="${RESULT_ROOT}/sd_batch_${EXPAND_NUM}x_${OPTIMIZER}_lr${LR}_wd${WEIGHT_DECAY}"
                ;;
            distdiff)
                label="DistDiff ${EXPAND_NUM}x"
                base="${RESULT_ROOT}/distdiff_batch_${EXPAND_NUM}x_${OPTIMIZER}_lr${LR}_wd${WEIGHT_DECAY}"
                ;;
            *)
                echo "[ERROR] Unknown setting: ${setting}"
                exit 1
                ;;
        esac
        count=$(count_results "${base}")
        total=$((total + count))
        echo "[${label}] ${count}/${seed_count}"
        show_seed_status "${base}"
        echo
    done

    echo "[total] ${total} / $((setting_count * seed_count)) results.yaml"
    echo

    latest_log=$(find logs -maxdepth 1 -name "stronger_*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}' || true)
    if [ -n "${latest_log}" ]; then
        echo "[latest log] ${latest_log}"
        tail -n 24 "${latest_log}"
    else
        echo "  stronger classifier log 없음"
    fi

    echo
    echo "다음 갱신까지 ${INTERVAL}s. 종료: Ctrl-C"
    sleep "${INTERVAL}"
done
