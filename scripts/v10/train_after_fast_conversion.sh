#!/usr/bin/env bash
# Wait for the bounded G1->V10 conversion, audit its final snapshot, then train
# one V10 policy on both local RTX 5080s for at least 12 hours.
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
run_name=v10_seed_fast_full_2x5080_20261024
log_dir="$root/logs/$run_name"
mkdir -p "$log_dir"
exec >>"$log_dir/pipeline.log" 2>&1
echo "$(date --iso-8601=seconds) waiting for G1->V10 conversion"

converter=v10-fast-g1-convert-8-v2.service
while [[ $(systemctl --user show "$converter" -p ActiveState --value) == active ]]; do
    sleep 30
done
result=$(systemctl --user show "$converter" -p Result --value)
exit_status=$(systemctl --user show "$converter" -p ExecMainStatus --value)
if [[ "$result" != success || "$exit_status" != 0 ]] || \
   ! journalctl --user -u "$converter" --no-pager | grep -F 'Complete:' >/dev/null; then
    echo "$(date --iso-8601=seconds) converter did not complete normally: result=$result exit=$exit_status"
    exit 1
fi
echo "$(date --iso-8601=seconds) conversion complete; auditing final library"

data/v10/fast_g1/venv/bin/python -B scripts/v10/audit_fast_g1_library.py \
    --output data/v10/manifests/fast_g1_final_audit.json

# The older SOMA source is only removed if there is too little room for logs
# and checkpoints. The completed G1 source and accepted V10 library are kept.
free_kb=$(df -Pk "$root" | awk 'NR==2 {print $4}')
if (( free_kb < 25 * 1024 * 1024 )); then
    old_source="$root/data/v10/downloads/soma_uniform.tar.gz"
    if [[ -f "$old_source" ]]; then
        echo "$(date --iso-8601=seconds) low disk space; removing old direct human-to-V10 source: $old_source"
        rm -- "$old_source"
    fi
fi
free_kb=$(df -Pk "$root" | awk 'NR==2 {print $4}')
if (( free_kb < 10 * 1024 * 1024 )); then
    echo "$(date --iso-8601=seconds) only $free_kb KiB free; refusing to start training"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$root/src"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$log_dir/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Each torchrun rank pins itself to one visible GPU. Start with useful batch
# sizes; if CUDA runs out of memory during startup, retry at smaller sizes.
for setting in '1024 2048 1024' '512 1024 512' '256 512 256'; do
    read -r envs policy_batch wm_batch <<< "$setting"
    echo "$(date --iso-8601=seconds) starting two-GPU training: envs=$envs policy_batch=$policy_batch wm_batch=$wm_batch"
    attempt_log="$log_dir/attempt_${envs}.log"
    attempt_start=$SECONDS
    resume_args=()
    if [[ -f "$log_dir/ckpt_latest.pt" ]]; then
        resume_args=(--resume "$log_dir/ckpt_latest.pt" --schedule.warmup-iterations 500)
        echo "$(date --iso-8601=seconds) resuming from latest checkpoint with replay refill"
    fi
    set +e
    "$root/.venv/bin/torchrun" --standalone --nproc_per_node=2 \
        -m supermimic.scripts.train \
        --robot v10 \
        --data.library data/v10/library/accepted_compact.npy \
        --run-name "$run_name" \
        --env.num-envs "$envs" \
        --policy-trainer.batch-size "$policy_batch" \
        --wm-trainer.batch-size "$wm_batch" \
        --schedule.total-iterations 1000000 \
        --schedule.max-hours 12.5 \
        --schedule.checkpoint-interval 1000 \
        "${resume_args[@]}" \
        2>&1 | tee "$attempt_log"
    train_status=${PIPESTATUS[0]}
    set -e
    elapsed=$((SECONDS - attempt_start))
    if (( train_status == 0 )); then
        if (( elapsed < 12 * 3600 )); then
            echo "$(date --iso-8601=seconds) training ended before 12 hours ($elapsed s); investigate"
            exit 1
        fi
        echo "$(date --iso-8601=seconds) training completed after $elapsed s"
        exit 0
    fi
    echo "$(date --iso-8601=seconds) training attempt exited $train_status after $elapsed s"
    if ! grep -Eiq 'out of memory|CUDNN_STATUS_ALLOC_FAILED|cudaMalloc.*failed' "$attempt_log"; then
        exit "$train_status"
    fi
    echo "$(date --iso-8601=seconds) retrying with smaller per-GPU workload after startup OOM"
done
echo "$(date --iso-8601=seconds) all two-GPU memory configurations failed"
exit 1
