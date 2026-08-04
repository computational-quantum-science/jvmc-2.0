
#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

N_BATCHES_LIST=(1 2 4 8)
QUANTITIES=(force qgt)

for quantity in "${QUANTITIES[@]}"; do
    echo "=== quantity=$quantity  jacobian=dense  n_batches=1 ==="
    python batched_jacobian.py dense --n_batches 1 --quantity "$quantity"

    for n_batches in "${N_BATCHES_LIST[@]}"; do
        echo "=== quantity=$quantity  jacobian=batched  n_batches=$n_batches ==="
        if ! python batched_jacobian.py batched --n_batches "$n_batches" --quantity "$quantity"; then
            echo "    FAILED (quantity=$quantity jacobian=batched n_batches=$n_batches) -- continuing"
        fi
    done
done
