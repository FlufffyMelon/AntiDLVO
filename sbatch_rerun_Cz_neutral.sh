#!/bin/bash
#SBATCH --job-name=zw_Cz_rerun
#SBATCH --cpus-per-task=16
#SBATCH --gpus 1
#SBATCH --time=24:00:00
#SBATCH --array=0-1

# Re-run of the 7 missing points of the Cz sweep for the neutral system
# (ions + zwitterionic dipoles, explicit charged walls).
#
# Everything is at base values except pz and Cz:
#   H = 3 nm, sigma = -20 uC/cm^2, C = 0.01 M, n_steps = 1e6.
#
# The 7 points are split over two array tasks, one GPU each, because every
# node currently has one GPU taken by another user and `--gpus 2` on a single
# node would stay pending. Step-2 measurements (20k steps, 7 runs sharing one
# GPU) gave 15.0 it/s for the heaviest point (Cz = 0.7, 3465 atoms), i.e.
# ~18.5 h for 1e6 steps; with only 4 runs per GPU there is a comfortable
# margin under the 24 h MaxTime of the `normal` partition. n_steps is NOT
# reduced.
#
# Array task 1 additionally re-runs the already-computed point pz = 17,
# Cz = 0.1 with the new placement for 200k steps into results_tmp/, as a
# sanity check that the new initial configuration reproduces the old results.

set -euo pipefail

# -----------------------------
# User-set variables
# -----------------------------

# Base config (unchanged; the only non-default setting passed below is the
# initial-placement mode)
CONFIG_FILE="configs_prod/ions_dipole.yaml"

# Fixed parameters for every run
H0=3
SIGMA0="-20"
C0="0.01"
N_STEPS=1000000

# Per-run stdout/stderr (inside results*, which is not synced back)
LOG_DIR="results_tmp/rerun_Cz_logs"

# run_calc starts two background jobs per calculation (the run itself and a
# timing poller), so the cap is twice the number of concurrent runs.
RUNS_PER_TASK=4
MAX_PARALLEL=$((2 * RUNS_PER_TASK))

# The 7 points, split into two groups of comparable cost (each group gets one
# Cz = 0.7, one 0.5, one 0.3 and one light run).
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
    0) SPECS=("17 0.7" "17 0.5" "17 0.3" "27.6 0.1") ;;
    1) SPECS=("27.6 0.7" "27.6 0.5" "27.6 0.3") ;;
    *) echo "Unexpected SLURM_ARRAY_TASK_ID=$TASK_ID" >&2; exit 1 ;;
esac

# -----------------------------
# Helpers
# -----------------------------

# Convert a float like 27.6 into a label like 27p6 for file/dir names
float_label() {
    local v="$1"
    echo "$v" | sed 's/\./p/g'
}

# The venv has cupy-cuda12x, which dlopens libnvrtc.so.12 at the first kernel
# compilation. A non-interactive sbatch shell does not read ~/.bashrc, so nothing
# puts a CUDA 12 lib dir on the path and CuPy dies right after startup.
#
# The only CUDA 12.x on this cluster is $HOME/cuda-12.3. The `cuda` modules under
# /opt/nvidia/hpc_sdk/modulefiles are unusable here: cuda/12.0 points at
# /usr/local/cuda-12.0, which does not exist on the compute nodes, and cuda/13.3
# provides libnvrtc.so.13 (CUDA 13), which cupy-cuda12x cannot load.
setup_cuda_libs() {
    export CUDA_HOME="${CUDA_HOME:-$HOME/cuda-12.3}"
    if [ ! -e "$CUDA_HOME/lib64/libnvrtc.so.12" ]; then
        echo "ERROR: $CUDA_HOME/lib64/libnvrtc.so.12 not found; CuPy will fail." >&2
        exit 1
    fi
    export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    source .venv/bin/activate
    echo "CUDA_HOME=$CUDA_HOME"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
}

# Semaphore to cap background jobs to MAX_PARALLEL
bg_limit() {
    local cap="$1"
    while :; do
        local running
        running=$(jobs -r -p | wc -l)
        if [ "$running" -lt "$cap" ]; then
            break
        fi
        wait -n || true
    done
}

# Timing infrastructure
TIMING_FILE="/tmp/timing_data_$$"
TIMING_PIDS_FILE="/tmp/timing_pids_$$"
> "$TIMING_FILE"
> "$TIMING_PIDS_FILE"

record_job_start() {
    local pid="$1"
    local name="$2"
    local start_time
    start_time=$(date +%s.%N)
    echo "$pid|$name|$start_time" >> "$TIMING_PIDS_FILE"
}

record_job_end() {
    local pid="$1"
    local end_time
    end_time=$(date +%s.%N)

    local start_line
    start_line=$(grep "^${pid}|" "$TIMING_PIDS_FILE" | head -1)
    if [ -n "$start_line" ]; then
        local name
        local start_time
        IFS='|' read -r _ name start_time <<< "$start_line"
        local duration
        duration=$(awk "BEGIN {printf \"%.6f\", $end_time - $start_time}")
        echo "$duration|$name" >> "$TIMING_FILE"
        sed -i "/^${pid}|/d" "$TIMING_PIDS_FILE"
    fi
}

cleanup_timing() {
    rm -f "$TIMING_FILE" "$TIMING_PIDS_FILE"
}
trap cleanup_timing EXIT

# Launch one calculation. Arguments: pz, Cz, n_steps, results_dir
run_calc() {
    local PZ="$1"
    local CZ_val="$2"
    local steps="$3"
    local results_dir="$4"

    local pz_label
    pz_label=$(float_label "$PZ")
    local name="ions_dipole_Cz_pz_${pz_label}_H_${H0}_sigma_${SIGMA0}_C_${C0}_Cz_${CZ_val}"

    source .venv/bin/activate

    ./run.sh "$CONFIG_FILE" \
        --pz "$PZ" \
        --H "$H0" \
        --sigma "$SIGMA0" \
        --C "$C0" \
        --Cz "$CZ_val" \
        --system.non_static_init.mode random_hs \
        --simulation.n_steps "$steps" \
        --experiment.name "$name" \
        --experiment.results_dir "$results_dir" \
        > "${LOG_DIR}/${name}_$(basename "$results_dir").log" 2>&1 &

    local job_pid=$!
    record_job_start "$job_pid" "$name"

    (
        while kill -0 "$job_pid" 2>/dev/null; do
            sleep 0.5
        done
        record_job_end "$job_pid"
    ) &

    echo "launched  pz=$PZ  Cz=$CZ_val  steps=$steps  pid=$job_pid  -> $results_dir/$name"
}

# -----------------------------
# Main
# -----------------------------

mkdir -p "$LOG_DIR"

SCRIPT_START_TIME=$(date +%s.%N)

echo "=========================================="
echo "Cz rerun, array task $TASK_ID on $(hostname)"
echo "Start: $(date)"
echo "Runs: ${SPECS[*]}"
echo "=========================================="

setup_cuda_libs
nvidia-smi || true

for spec in "${SPECS[@]}"; do
    set -- $spec
    PZ="$1"
    CZ_val="$2"
    pz_label=$(float_label "$PZ")
    bg_limit "$MAX_PARALLEL"
    run_calc "$PZ" "$CZ_val" "$N_STEPS" "results_prod_neutral/pz_${pz_label}/Cz"
done

# Optional validation run: the already-computed point pz = 17, Cz = 0.1 with the
# new placement, short, into results_tmp (never into results_prod_neutral).
if [ "$TASK_ID" = "1" ]; then
    bg_limit "$MAX_PARALLEL"
    run_calc "17" "0.1" 200000 "results_tmp/init_compare"
fi

# Do not let a single failed run abort the timing report (set -e would).
wait || echo "WARNING: at least one background job exited non-zero"
sleep 0.1

SCRIPT_END_TIME=$(date +%s.%N)
SCRIPT_TOTAL_TIME=$(awk "BEGIN {printf \"%.6f\", $SCRIPT_END_TIME - $SCRIPT_START_TIME}")

echo ""
echo "=========================================="
echo "Array task $TASK_ID finished."
echo "=========================================="
echo ""

format_time() {
    local seconds="$1"
    local hours
    local minutes
    local secs
    local remainder
    hours=$(awk "BEGIN {printf \"%.0f\", int($seconds / 3600)}")
    remainder=$(awk "BEGIN {printf \"%.6f\", $seconds % 3600}")
    minutes=$(awk "BEGIN {printf \"%.0f\", int($remainder / 60)}")
    secs=$(awk "BEGIN {printf \"%.2f\", $remainder % 60}")
    printf "%02d:%02d:%05.2f" "$hours" "$minutes" "$secs"
}

if [ -s "$TIMING_FILE" ]; then
    echo "PER-RUN WALL TIMES"
    echo "=================="
    while IFS='|' read -r duration name; do
        printf "%-60s %s\n" "$name" "$(format_time "$duration")"
    done < "$TIMING_FILE"
    echo ""
fi

echo "Total wall time: $(format_time "$SCRIPT_TOTAL_TIME")"
echo "End: $(date)"
echo ""
