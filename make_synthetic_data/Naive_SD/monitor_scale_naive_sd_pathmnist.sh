#!/bin/bash
# Live monitor for PathMNIST naive Stable Diffusion synthetic generation.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MODEL_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
cd "${MODEL_ROOT}"

DATASET=${DATASET:-pathmnist}
RATIOS=${SD_RATIOS:-"1 2 5 10"}
EXPECTED_BASE=${EXPECTED_BASE:-900}
INTERVAL=${INTERVAL:-30}

while true; do
    clear
    echo "============================================================"
    echo "  PathMNIST Naive SD synthetic generation 진행 상황"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo

    echo "[process]"
    ps -eo pid,ppid,stat,etime,cmd | grep -E "run_scale_naive_sd_pathmnist|generate_data.py|naive_sd_generate" | grep -v grep || echo "  실행 중인 Naive SD generation 프로세스 없음"
    echo

    total=0
    target=0
    echo "[generated images]"
    for ratio in ${RATIOS}; do
        dir="data/${DATASET}_expansion/save/sd_batch_${ratio}x"
        expected=$((EXPECTED_BASE * ratio))
        target=$((target + expected))
        if [ -d "${dir}" ]; then
            count=$(find "${dir}" -name "*.png" 2>/dev/null | wc -l)
        else
            count=0
        fi
        total=$((total + count))
        pct=$(awk "BEGIN { printf \"%.1f\", (${count}/${expected})*100 }")
        printf "  sd_%sx: %d / %d (%s%%)\n" "${ratio}" "${count}" "${expected}" "${pct}"
    done
    echo "  total: ${total} / ${target}"

    echo
    echo "[master log tail]"
    if [ -f logs/naive_sd_generate_master.log ]; then
        tail -n 18 logs/naive_sd_generate_master.log
    else
        echo "  logs/naive_sd_generate_master.log 없음"
    fi

    echo
    latest_log=$(find logs -maxdepth 1 -name "naive_sd_generate_*x_split*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}' || true)
    echo "[latest run log tail]"
    if [ -n "${latest_log}" ]; then
        echo "  ${latest_log}"
        tail -n 24 "${latest_log}"
    else
        echo "  Naive SD generation log 없음"
    fi

    echo
    echo "다음 갱신까지 ${INTERVAL}s. 종료: Ctrl-C"
    sleep "${INTERVAL}"
done
