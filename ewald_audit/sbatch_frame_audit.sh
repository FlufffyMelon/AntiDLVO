#!/bin/bash
#SBATCH --job-name=ew_audit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=ewald_audit/logs/frame_%A_%a.out
#SBATCH --error=ewald_audit/logs/frame_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

setup_cuda_libs() {
    export CUDA_HOME="${CUDA_HOME:-$HOME/cuda-12.3}"
    if [ ! -e "$CUDA_HOME/lib64/libnvrtc.so.12" ]; then
        echo "ERROR: $CUDA_HOME/lib64/libnvrtc.so.12 not found; CuPy will fail." >&2
    fi
    export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}
setup_cuda_libs
export AUDIT_MEM_BUDGET=2.0e8

LIST=ewald_audit/run_list.txt
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
echo "task $SLURM_ARRAY_TASK_ID -> $RUN"

.venv/bin/python ewald_audit/scripts/frame_audit.py "$RUN" \
    --frames 2000 3000 4000 --n-moves 150 --gpu 0
