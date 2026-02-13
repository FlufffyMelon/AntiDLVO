set -euo pipefail

# -----------------------------
# User-set variables
# -----------------------------

# Set the dipole moment at the very beginning (run this script twice: 17 and 27.6)
# PZ="17"
PZ="27.6"

# Base config to use
CONFIG_FILE="configs_prod/ions_dipole.yaml"

# Degree of parallelism (adjusted for 32 cores)
MAX_PARALLEL=64

# Defaults (match values in the config, i.e., those marked in parentheses)
H0=3
SIGMA0="-20"
C0="0.01"
CZ0="0.1"

# Sweeps (include defaults as specified)
H_SWEEP=(3 5 7 9 11)
SIGMA_SWEEP=(-30 -25 -20 -15 -10 -5 0)
C_SWEEP=(0 0.01 0.04 0.07 0.1)
CZ_SWEEP=(0 0.1 0.3 0.5 0.7)

# -----------------------------
# Helpers
# -----------------------------

# Convert a float like 27.6 into a label like 27p6 for file/dir names
float_label() {
    local v="$1"
    echo "$v" | sed 's/\./p/g'
}

# Semaphore to cap background jobs to MAX_PARALLEL
bg_limit() {
    local cap="$1"
    while :; do
        # Count running background jobs for this shell
        local running
        running=$(jobs -r -p | wc -l)
        if [ "$running" -lt "$cap" ]; then
            break
        fi
        # Wait for any job to finish
        wait -n || true
    done
}

# GPU device counter (alternates between 0 and 1)
# Use a file to persist counter across subshells
GPU_COUNTER_FILE="/tmp/gpu_counter_$$"
echo "0" > "$GPU_COUNTER_FILE"

get_gpu_device() {
    local counter
    counter=$(cat "$GPU_COUNTER_FILE")
    local device=$((counter % 2))
    counter=$((counter + 1))
    echo "$counter" > "$GPU_COUNTER_FILE"
    echo "$device"
}

# Cleanup function to remove counter file on exit
cleanup_gpu_counter() {
    rm -f "$GPU_COUNTER_FILE"
}
trap cleanup_gpu_counter EXIT

# Timing infrastructure
TIMING_FILE="/tmp/timing_data_$$"
TIMING_PIDS_FILE="/tmp/timing_pids_$$"
> "$TIMING_FILE"  # Initialize empty file
> "$TIMING_PIDS_FILE"  # Initialize empty file

# Record start time for a job
record_job_start() {
    local pid="$1"
    local name="$2"
    local start_time
    start_time=$(date +%s.%N)
    echo "$pid|$name|$start_time" >> "$TIMING_PIDS_FILE"
}

# Record end time and calculate duration for a job
record_job_end() {
    local pid="$1"
    local end_time
    end_time=$(date +%s.%N)

    # Find the start time for this PID
    local start_line
    start_line=$(grep "^${pid}|" "$TIMING_PIDS_FILE" | head -1)
    if [ -n "$start_line" ]; then
        local name
        local start_time
        IFS='|' read -r _ name start_time <<< "$start_line"
        local duration
        duration=$(awk "BEGIN {printf \"%.6f\", $end_time - $start_time}")
        echo "$duration|$name" >> "$TIMING_FILE"
        # Remove the PID entry
        sed -i "/^${pid}|/d" "$TIMING_PIDS_FILE"
    fi
}

# Cleanup timing files on exit
cleanup_timing() {
    rm -f "$TIMING_FILE" "$TIMING_PIDS_FILE"
}
trap cleanup_timing EXIT

# Launch one calculation with overrides and a descriptive experiment name + results dir
run_calc() {
    local sweep_kind="$1"   # e.g., H, sigma, C, Cz
    local H_val="$2"
    local SIGMA_val="$3"
    local C_val="$4"
    local CZ_val="$5"

    local pz_label
    pz_label=$(float_label "$PZ")

    # Build a compact, navigable name embedding parameters
    local name="ions_dipole_${sweep_kind}_pz_${pz_label}_H_${H_val}_sigma_${SIGMA_val}_C_${C_val}_Cz_${CZ_val}"

    # Place results under a hierarchy by pz and sweep kind
    local results_dir="results_prod/pz_${pz_label}/${sweep_kind}"

    # Ensure results_dir exists so logs/config backups are easy to find (runner will still create its own timestamped subfolder)
    # mkdir -p "$results_dir"

    # Get GPU device (alternates between 0 and 1)
    local gpu_dev
    # gpu_dev=$(get_gpu_device)
    gpu_dev=1

    source .venv/bin/activate

    # Start the job and capture its PID
    ./run.sh "$CONFIG_FILE" \
        --ewald.gpu_device "$gpu_dev" \
        --pz "$PZ" \
        --H "$H_val" \
        --sigma "$SIGMA_val" \
        --C "$C_val" \
        --Cz "$CZ_val" \
        --experiment.name "$name" \
        --experiment.results_dir "$results_dir" &

    local job_pid=$!

    # Record start time for this job
    record_job_start "$job_pid" "$name"

    # Set up a background process to poll for job completion and record its end time
    (
        # Poll until the process finishes
        while kill -0 "$job_pid" 2>/dev/null; do
            sleep 0.5
        done
        record_job_end "$job_pid"
    ) &

    # --ewald.gpu_device "$gpu_dev" \
}

# -----------------------------
# Main: submit grid with fixed defaults except the varied param
# -----------------------------

# Record overall script start time
SCRIPT_START_TIME=$(date +%s.%N)

echo "Starting grid for pz=$PZ with up to $MAX_PARALLEL parallel jobs..."
echo "Script start time: $(date)"

# H sweep (vary H; others at defaults)
for H_val in "${H_SWEEP[@]}"; do
    # bg_limit "$MAX_PARALLEL"
    run_calc "H" "$H_val" "$SIGMA0" "$C0" "$CZ0"
done

# sigma sweep (vary sigma; others at defaults)
for SIGMA_val in "${SIGMA_SWEEP[@]}"; do
    # bg_limit "$MAX_PARALLEL"
    run_calc "sigma" "$H0" "$SIGMA_val" "$C0" "$CZ0"
done

# C sweep (vary C; others at defaults)
for C_val in "${C_SWEEP[@]}"; do
    # bg_limit "$MAX_PARALLEL"
    run_calc "C" "$H0" "$SIGMA0" "$C_val" "$CZ0"
done

# Cz sweep (vary Cz; others at defaults)
for CZ_val in "${CZ_SWEEP[@]}"; do
    # bg_limit "$MAX_PARALLEL"
    run_calc "Cz" "$H0" "$SIGMA0" "$C0" "$CZ_val"
done

# Wait for all background jobs (including timing recorders)
wait

# Small delay to ensure all file writes are flushed
sleep 0.1

# Record overall script end time
SCRIPT_END_TIME=$(date +%s.%N)
SCRIPT_TOTAL_TIME=$(awk "BEGIN {printf \"%.6f\", $SCRIPT_END_TIME - $SCRIPT_START_TIME}")

echo ""
echo "=========================================="
echo "All calculations for pz=$PZ completed."
echo "=========================================="
echo ""

# Calculate and display timing statistics
if [ -s "$TIMING_FILE" ]; then
    # Read all durations into an array and calculate statistics
    durations=()
    names=()
    while IFS='|' read -r duration name; do
        durations+=("$duration")
        names+=("$name")
    done < "$TIMING_FILE"

    num_jobs=${#durations[@]}

    if [ "$num_jobs" -gt 0 ]; then
        # Calculate sum for mean
        sum=0
        min="${durations[0]}"
        max="${durations[0]}"

        for duration in "${durations[@]}"; do
            sum=$(awk "BEGIN {printf \"%.6f\", $sum + $duration}")
            # Compare using awk for floating point
            if awk "BEGIN {exit !($duration < $min)}"; then
                min="$duration"
            fi
            if awk "BEGIN {exit !($duration > $max)}"; then
                max="$duration"
            fi
        done

        mean=$(awk "BEGIN {printf \"%.2f\", $sum / $num_jobs}")

        # Format times in human-readable format
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

        echo "TIMING STATISTICS"
        echo "================="
        echo "Total number of tasks: $num_jobs"
        echo ""
        echo "Total wall time: $(format_time "$SCRIPT_TOTAL_TIME")"
        echo "Mean task time:  $(format_time "$mean")"
        echo "Min task time:   $(format_time "$min")"
        echo "Max task time:   $(format_time "$max")"
        echo ""
        echo "Script end time: $(date)"
    else
        echo "Warning: No timing data collected."
    fi
else
    echo "Warning: No timing data file found."
fi

echo ""


