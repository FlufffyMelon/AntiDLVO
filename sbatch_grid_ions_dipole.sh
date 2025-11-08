#!/bin/bash
#SBATCH --job-name=python
#SBATCH --cpus-per-task=24
#SBATCH -w node4

set -euo pipefail

# -----------------------------
# User-set variables
# -----------------------------

# Set the dipole moment at the very beginning (run this script twice: 17 and 27.6)
# PZ="17"   # options: 17 or 27.6
PZ="27.6"   # options: 17 or 27.6

# Base config to use
CONFIG_FILE="configs_prod/ions_dipole.yaml"

# Degree of parallelism (must match --cpus-per-task above)
MAX_PARALLEL=24

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

    ./run.sh "$CONFIG_FILE" \
        --pz "$PZ" \
        --H "$H_val" \
        --sigma "$SIGMA_val" \
        --C "$C_val" \
        --Cz "$CZ_val" \
        --experiment.name "$name" \
        --experiment.results_dir "$results_dir" &
}

# -----------------------------
# Main: submit grid with fixed defaults except the varied param
# -----------------------------

echo "Starting grid for pz=$PZ with up to $MAX_PARALLEL parallel jobs..."

# H sweep (vary H; others at defaults)
for H_val in "${H_SWEEP[@]}"; do
    bg_limit "$MAX_PARALLEL"
    run_calc "H" "$H_val" "$SIGMA0" "$C0" "$CZ0"
done

# sigma sweep (vary sigma; others at defaults)
for SIGMA_val in "${SIGMA_SWEEP[@]}"; do
    bg_limit "$MAX_PARALLEL"
    run_calc "sigma" "$H0" "$SIGMA_val" "$C0" "$CZ0"
done

# C sweep (vary C; others at defaults)
for C_val in "${C_SWEEP[@]}"; do
    bg_limit "$MAX_PARALLEL"
    run_calc "C" "$H0" "$SIGMA0" "$C_val" "$CZ0"
done

# Cz sweep (vary Cz; others at defaults)
for CZ_val in "${CZ_SWEEP[@]}"; do
    bg_limit "$MAX_PARALLEL"
    run_calc "Cz" "$H0" "$SIGMA0" "$C0" "$CZ_val"
done

# Wait for all background jobs
wait

echo "All calculations for pz=$PZ completed."


