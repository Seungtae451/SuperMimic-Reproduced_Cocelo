#!/usr/bin/env bash
# Train a fresh V10 policy on the accepted compact library with both RTX 5080s.
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

run_name=v10_seed_320k_2x5080_20261024
log_dir="$root/logs/$run_name"
if [[ -e "$log_dir/config.json" || -e "$log_dir/ckpt_latest.pt" ]]; then
    echo "Refusing to overwrite an existing training run: $log_dir" >&2
    exit 1
fi
if [[ ! -f data/v10/library/accepted_compact.npy ]]; then
    echo "Accepted V10 motion library is missing" >&2
    exit 1
fi

mkdir -p "$log_dir/matplotlib"
exec >>"$log_dir/train.log" 2>&1
echo "$(date --iso-8601=seconds) starting fresh 320,000-iteration V10 training"

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$root/src"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$log_dir/matplotlib"

exec "$root/.venv/bin/torchrun" --standalone --nproc_per_node=2 \
    -m supermimic.scripts.train \
    --robot v10 \
    --data.library data/v10/library/accepted_compact.npy \
    --run-name "$run_name" \
    --env.num-envs 1024 \
    --policy-trainer.batch-size 2048 \
    --wm-trainer.batch-size 1024 \
    --schedule.total-iterations 320000 \
    --schedule.checkpoint-interval 4000
