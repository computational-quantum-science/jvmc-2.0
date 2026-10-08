#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

N_BATCHES_LIST=(1 2 4 8 16)
QUANTITIES=(force qgt)
# Every sweep writes to a new folder, so that results of different runs are never mixed
OUT_DIR="results_$(date +%Y%m%d_%H%M%S)"

run() {
    echo "=== $* ==="
    if ! python batched_jacobian.py "$@" --out_dir "$OUT_DIR"; then
        echo "    FAILED ($*) -- continuing"
    fi
}

for quantity in "${QUANTITIES[@]}"; do
    run dense --n_batches 1 --quantity "$quantity"

    for n_batches in "${N_BATCHES_LIST[@]}"; do
        run batched --n_batches "$n_batches" --quantity "$quantity"
    done
done

python plot_jacobian.py "$OUT_DIR"
